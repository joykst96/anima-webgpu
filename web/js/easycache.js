// EasyCache — ComfyUI comfy_extras/nodes_easycache.py(EasyCacheHolder + easycache_forward_wrapper) 포팅.
// 모델 호출(DiT run) 단위로, 입력 변화율로 출력 변화를 예측해 누적 변화가 임계 미만이면 호출을 건너뛰고
// 마지막으로 계산한 (출력 − 입력) 차이를 현재 입력에 더해 쓴다. 판단은 긍정 프롬프트 출력으로 하고, 건너뛰면 CFG 쪽도 함께 건너뛴다.
// 입력 x = latent(σ로 나누지 않음, CONST), 출력 = DiT v 예측 — ComfyUI DIFFUSION_MODEL 래퍼가 보는 값과 같다.
// ComfyUI와 다른 점:
//  - 마지막 모델 호출은 항상 계산한다. Turbo 8스텝은 마지막 σ(0.3)가 end 구간(σ 0.136) 밖이라 원본대로면 마지막 스텝도
//    건너뛰는데, 그러면 최종 잡음이 덜 빠져 그림 전체에 자글자글한 질감이 남는다(실측). 베이스 30스텝은 원래 end 구간이라 무관.
//  - 해상도가 바뀌면(SPEED 단계 전환) 상태를 처음처럼 비운다(ComfyUI reset은 변화율을 0으로 둬 다음 호출을 바로 건너뛴다).
import { SHIFT } from "./constants.js";

// ComfyUI 노드 기본값 (subsample 8은 노드에 고정)
const EASYCACHE = { threshold: 0.2, startPct: 0.15, endPct: 0.95, sub: 8 };

// ModelSamplingDiscreteFlow.percent_to_sigma (shift 적용)
const pctToSigma = (p) => (p <= 0 ? 1 : p >= 1 ? 0 : (SHIFT * (1 - p)) / (1 + (SHIFT - 1) * (1 - p)));

// [C, H, W] 텐서의 공간 ::sub, ::sub 표본 (ComfyUI subsample — 채널은 전부)
function subsample(a, C, H, W, s) {
  const h = Math.ceil(H / s),
    w = Math.ceil(W / s),
    out = new Float32Array(C * h * w);
  let o = 0;
  for (let c = 0; c < C; c++)
    for (let i = 0; i < H; i += s) for (let j = 0; j < W; j += s) out[o++] = a[c * H * W + i * W + j];
  return out;
}

function meanAbsDiff(a, b) {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += Math.abs(a[k] - b[k]);
  return s / a.length;
}

function meanAbs(a) {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += Math.abs(a[k]);
  return s / a.length;
}

// lastSigma: 스케줄의 마지막 모델 호출 σ — 이 호출은 항상 계산한다 (아래 설명)
function createEasyCache(C, lastSigma, opts = EASYCACHE) {
  const startT = pctToSigma(opts.startPct),
    endT = pctToSigma(opts.endPct);
  let rate, cum, xPrev, outPrev, outNorm, diffs, shape, pending;
  const reset = () => {
    rate = null; // relative_transformation_rate
    cum = 0; // cumulative_change_rate
    xPrev = outPrev = outNorm = null;
    diffs = null; // [긍정, 부정|null] 출력 − 입력
    shape = null;
  };
  reset();
  const ec = {
    skipped: [], // 건너뛴 호출 번호 (1부터)
    calls: 0,
    // 호출 전: 건너뛸 수 있으면 [v, vu|null] 추정값, 아니면 null (→ 실제로 계산한 뒤 after 호출)
    before(x, sigma, H, W) {
      ec.calls++;
      pending = null;
      if (sigma <= endT || sigma <= lastSigma) return null; // 끝 구간·마지막 스텝: 계산만, 상태 갱신 없음
      const active = sigma <= startT;
      let inChange = null;
      if (active) {
        const key = `${H}x${W}`;
        if (shape !== key) {
          if (shape !== null) reset();
          shape = key;
        }
        if (xPrev) inChange = meanAbsDiff(subsample(x, C, H, W, opts.sub), xPrev);
        if (outNorm !== null && rate !== null) {
          cum += (rate * inChange) / outNorm;
          if (cum < opts.threshold && diffs) {
            ec.skipped.push(ec.calls);
            return diffs.map((d) => d && Float32Array.from(x, (xv, k) => xv + d[k]));
          }
          cum = 0;
        }
      }
      pending = { active, inChange, H, W };
      return null;
    },
    // 계산 후: 차이 캐시와 변화율 갱신. v·vu는 CFG 결합 전 값.
    after(x, v, vu) {
      if (!pending) return;
      const { active, inChange, H, W } = pending;
      diffs = [v, vu].map((o) => o && Float32Array.from(o, (ov, k) => ov - x[k]));
      if (!active) return;
      const vs = subsample(v, C, H, W, opts.sub);
      if (outNorm !== null && inChange !== null) rate = meanAbsDiff(vs, outPrev) / inChange;
      xPrev = subsample(x, C, H, W, opts.sub);
      outPrev = vs;
      outNorm = meanAbs(v);
    },
  };
  return ec;
}

export { EASYCACHE, createEasyCache };
