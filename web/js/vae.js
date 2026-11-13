// VAE 디코드 — 분할 그래프(head/tail)가 있으면 tail을 가로 띠로 나눠 돌린다.
//
// ORT WebGPU는 한 번의 run() 안에서 해제된 버퍼를 재사용하지 않아 정점 메모리가 중간 텐서 합만큼 커진다
// (1024² +7~8GB, 1280² +10.8GB → VRAM 상한에서 페이징, VAE 5.1s). run을 띠 단위로 쪼개면 run마다 버퍼가
// 반납되므로 정점이 띠 크기에 묶인다. tail은 국소 연산만 있어(수용영역 반경 14.7 latent 실측) halo 16으로
// 잘라 붙이면 단일 호출과 비트 동일하다(1280²·1536² 실측). 분할 그래프는 export/split_vae.py 산출물.
import { LAT_C } from "./constants.js";
import { log } from "./dom.js";
import { sess } from "./loader.js";
import { HAS_F16, disposeRun, disposeTensor, makeF16Tensor, ort } from "./ort.js";

const VAE_HALO = 16; // latent 단위, 수용영역 반경 117.5px(14.7 latent)보다 크게
const VAE_STRIP_AREA = 128 * 128; // 띠 하나의 입력 latent 면적 상한 (띠 2개 이상일 때)
// 띠 없이 한 번에 돌리는 면적 상한. 이보다 크면 최소 2개로 나눈다 — 1024²급(832×1216 등 포함)이 한 번에 돌면 일시 VRAM이 +7~8GB라
// 정점이 14.8GB까지 올라가고, 여유가 적은 카드에서는 페이징으로 VAE가 수 초씩 느려진다. 2개로 나누면 +3.4GB(정점 11.1GB), 그림은 비트 동일,
// VAE 시간은 halo를 겹쳐 계산하는 만큼 늘어난다(2026-11-13 실측). 768² 이하와 1280² 이상은 종전과 같다.
const VAE_SINGLE_AREA = 80 * 128;
// head의 mid-block 어텐션(latent 해상도 전역)을 쿼리 청크별 run으로 나누는 기준. 한 run 안에선 8청크의 score·prob가
// 전부 살아 있어 1536²에서 청크당 680MB×2 × 8이 쌓인다. 1024² 이하는 한 run이 더 빠르다.
const VAE_ATTN_SPLIT_TOKENS = 128 * 128;
const VAE_ATTN_CHUNKS = 8; // 원래 그래프의 Split(num_outputs=8)과 같게 — ceil(N/8)행씩, 마지막만 짧음

// head를 돌려 mid-block 출력(tail 입력)을 CPU로 받는다. 큰 해상도는 head_a → 청크별 attn → head_b.
async function runHead(vaeIn, H, W) {
  if (!(sess.vaeHeadA && H * W > VAE_ATTN_SPLIT_TOKENS)) {
    const ho = await sess.vaeHead.run({ latent: vaeIn });
    return { mid: ho[Object.keys(ho)[0]], release: () => disposeRun(ho) };
  }
  // head_a 출력은 전부 GPU 버퍼(loader에서 지정): kᵀ·v·residual은 GPU에 둔 채 다음 run에 넘기고, q만 내려 청크로 자른다.
  const a = await sess.vaeHeadA.run({ latent: vaeIn });
  const [qT, kT, vT, resid] = sess.vaeHeadA.outputNames.map((n) => a[n]);
  const q = await qT.getData(),
    C = qT.dims[3],
    N = qT.dims[2];
  disposeTensor(qT);
  const out = new Float32Array(N * C),
    rows = Math.ceil(N / VAE_ATTN_CHUNKS);
  for (let r0 = 0; r0 < N; r0 += rows) {
    const n = Math.min(rows, N - r0);
    const qc = new ort.Tensor("float32", q.slice(r0 * C, (r0 + n) * C), [1, 1, n, C]);
    const o = await sess.vaeAttn.run({ q: qc, kT, v: vT });
    out.set(o.out.data, r0 * C);
    disposeTensor(qc);
    disposeRun(o);
  }
  disposeTensor(kT);
  disposeTensor(vT);
  const [attnName, residName] = sess.vaeHeadB.inputNames.filter((n) => n !== "latent");
  const attnT = new ort.Tensor("float32", out, [1, 1, N, C]);
  const bo = await sess.vaeHeadB.run({ [attnName]: attnT, [residName]: resid, latent: vaeIn });
  disposeTensor(attnT);
  disposeTensor(resid);
  return { mid: bo[Object.keys(bo)[0]], release: () => disposeRun(bo) };
}

// 입력 면적이 상한 이하가 되는 가장 적은 띠 수(면적이 VAE_SINGLE_AREA를 넘으면 2개부터)와 각 띠의 [코어 시작, 끝, 입력 시작, 끝]
function planStrips(H, W) {
  for (let n = H * W <= VAE_SINGLE_AREA ? 1 : 2; ; n++) {
    const S = Math.ceil(H / n),
      strips = [];
    for (let y0 = 0; y0 < H; y0 += S) {
      const y1 = Math.min(H, y0 + S);
      strips.push([y0, y1, n === 1 ? 0 : Math.max(0, y0 - VAE_HALO), n === 1 ? H : Math.min(H, y1 + VAE_HALO)]);
    }
    if (n >= H || strips.every(([, , i0, i1]) => (i1 - i0) * W <= VAE_STRIP_AREA)) return strips;
  }
}

// xf: fp32 latent (LAT_C·H·W). 반환 img는 fp32 (1,3,8H,8W) [-1,1]. 판독 후 release()로 텐서를 반환한다.
async function decodeLatent(xf, H, W) {
  const vaeIn = makeF16Tensor(xf, [1, LAT_C, 1, H, W]);
  // WebNN VAE(loader.js loadWebnnVae): 그 크기 전용 fp16 그래프를 한 번에 돌린다. 실행이 실패하면 내리고 아래 WebGPU 경로로 디코드한다.
  const wn = sess.vaeWebnn;
  if (wn && wn.lat.h === H && wn.lat.w === W) {
    try {
      const vo = await wn.run({ latent: vaeIn });
      return {
        img: vo[wn.outputNames[0]].data,
        release() {
          disposeTensor(vaeIn);
          disposeRun(vo);
        },
      };
    } catch (e) {
      console.error(e);
      log(`VAE(WebNN) 실행 실패 — ${e.message}. 기존 경로(WebGPU)로 디코드`, "err");
      sess.vaeWebnn = null;
      wn.release().catch(() => {});
    }
  }
  if (!(sess.vaeHead && sess.vaeTail)) {
    const vo = await sess.vae.run({ latent: vaeIn });
    return {
      img: vo[Object.keys(vo)[0]].data,
      release() {
        disposeTensor(vaeIn);
        disposeRun(vo);
      },
    };
  }
  const head = await runHead(vaeIn, H, W);
  const mid = head.mid;
  const A = mid.data,
    C = mid.dims[1];
  const midName = sess.vaeTail.inputNames.find((n) => n !== "latent");
  const Wp = W * 8,
    img = new Float32Array(3 * H * 8 * Wp);
  for (const [y0, y1, i0, i1] of planStrips(H, W)) {
    const hin = i1 - i0;
    const crop = new Float32Array(C * hin * W);
    for (let c = 0; c < C; c++) crop.set(A.subarray(c * H * W + i0 * W, c * H * W + i1 * W), c * hin * W);
    const midT = new ort.Tensor("float32", crop, [1, C, 1, hin, W]);
    const zeros = HAS_F16 ? new Float16Array(LAT_C * hin * W) : new Uint16Array(LAT_C * hin * W);
    const shapeT = new ort.Tensor("float16", zeros, [1, LAT_C, 1, hin, W]); // tail은 latent를 Shape로만 읽음
    const to = await sess.vaeTail.run({ [midName]: midT, latent: shapeT });
    const part = to[Object.keys(to)[0]].data,
      skip = (y0 - i0) * 8,
      rows = (y1 - y0) * 8;
    for (let ch = 0; ch < 3; ch++)
      img.set(part.subarray((ch * hin * 8 + skip) * Wp, (ch * hin * 8 + skip + rows) * Wp), (ch * H * 8 + y0 * 8) * Wp);
    disposeTensor(midT);
    disposeTensor(shapeT);
    disposeRun(to);
  }
  disposeTensor(vaeIn);
  head.release();
  return { img, release() {} };
}

export { decodeLatent, planStrips };
