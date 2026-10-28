// SPEED(스펙트럼 점진 샘플링): DCT 확장 + 멀티해상도 euler
import { gaussianFill } from "./sampling.js";

// ═════════ SPEED / SPD (Spectral Progressive Diffusion) ═════════
// 저해상도에서 denoise 시작 → 특정 σ에서 DCT 스펙트럼 확장으로 해상도를
// 키우며 고주파를 노이즈로 채움. Anima는 σ를 t로 직접 사용.
//
// 출처(verbatim 포팅, 감사):
//   - ComfyUI-Spectrum-KSampler (spd.py) — 베이스
//     https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler
//   - ComfyUI-SPEED — 교차 참고
//     https://github.com/ruwwww/ComfyUI-SPEED
//   - 논문: Xiao et al., arXiv:2605.18736
// 수식은 numpy 골든값으로 검증됨(오차 1e-15).
const _DCT_CACHE = new Map();
function _dctMatrix(n) {
  const c = _DCT_CACHE.get(n);
  if (c) return c;
  const m = new Float64Array(n * n);
  const s0 = 1 / Math.sqrt(n),
    sk = Math.sqrt(2 / n);
  for (let k = 0; k < n; k++) {
    const sc = k === 0 ? s0 : sk,
      b = k * n;
    for (let j = 0; j < n; j++)
      m[b + j] = Math.cos((Math.PI * k * (2 * j + 1)) / (2 * n)) * sc;
  }
  _DCT_CACHE.set(n, m);
  return m;
}
// dct2 = Dh @ x @ Dw.T  (채널별, x는 길이 C*H*W flat)
function _dct2(x, C, H, W) {
  const Dh = _dctMatrix(H),
    Dw = _dctMatrix(W),
    plane = H * W;
  const out = new Float64Array(C * plane);
  for (let c = 0; c < C; c++) {
    const xb = c * plane;
    const t1 = new Float64Array(plane); // Dh @ ch
    for (let i = 0; i < H; i++)
      for (let k = 0; k < H; k++) {
        const a = Dh[i * H + k];
        if (a === 0) continue;
        for (let j = 0; j < W; j++)
          t1[i * W + j] += a * x[xb + k * W + j];
      }
    for (
      let i = 0;
      i < H;
      i++ // (t1) @ Dw.T
    )
      for (let r = 0; r < W; r++) {
        let s = 0;
        const rb = r * W;
        for (let k = 0; k < W; k++) s += t1[i * W + k] * Dw[rb + k];
        out[xb + i * W + r] = s;
      }
  }
  return out;
}
// idct2 = Dh.T @ x @ Dw
function _idct2(x, C, H, W) {
  const Dh = _dctMatrix(H),
    Dw = _dctMatrix(W),
    plane = H * W;
  const out = new Float64Array(C * plane);
  for (let c = 0; c < C; c++) {
    const xb = c * plane;
    const t1 = new Float64Array(plane); // Dh.T @ ch
    for (let k = 0; k < H; k++)
      for (let i = 0; i < H; i++) {
        const a = Dh[k * H + i];
        if (a === 0) continue;
        for (let j = 0; j < W; j++)
          t1[i * W + j] += a * x[xb + k * W + j];
      }
    for (
      let i = 0;
      i < H;
      i++ // (t1) @ Dw
    )
      for (let j = 0; j < W; j++) {
        let s = 0;
        for (let k = 0; k < W; k++) s += t1[i * W + k] * Dw[k * W + j];
        out[xb + i * W + j] = s;
      }
  }
  return out;
}
const _snap = (v, m) => Math.max(m, Math.round(v / m) * m);

// (C,H,W) latent → 저주파 (C,h,w). 반환 {data,h,w}
function _dctLowpassInit(x, C, H, W, scale, patch) {
  const xi = _dct2(x, C, H, W);
  const h = Math.min(_snap(H * scale, patch), H);
  const w = Math.min(_snap(W * scale, patch), W);
  const crop = new Float64Array(C * h * w);
  for (let c = 0; c < C; c++)
    for (let y = 0; y < h; y++) {
      const src = c * H * W + y * W,
        dst = c * h * w + y * w;
      for (let xx = 0; xx < w; xx++) crop[dst + xx] = xi[src + xx];
    }
  return { data: _idct2(crop, C, h, w), h, w };
}

// 저해상도 (C,hLo,wLo) → 고해상도. noise 내부 생성(rand). 반환 {data,h,w,sigmaAligned}
function _spectralExpand(
  xLo,
  C,
  hLo,
  wLo,
  sigmaVal,
  scaleLo,
  scaleHi,
  Hfull,
  Wfull,
  patch,
  rand,
) {
  const xi = _dct2(xLo, C, hLo, wLo);
  const hHi = Math.max(_snap(Hfull * scaleHi, patch), hLo);
  const wHi = Math.max(_snap(Wfull * scaleHi, patch), wLo);
  const r = scaleHi / scaleLo;
  const sigmaAligned = (r * sigmaVal) / (1 + (r - 1) * sigmaVal);
  const kappa = r / (1 + (r - 1) * sigmaVal);

  const noise = new Float64Array(C * hHi * wHi);
  const nf = new Float32Array(C * hHi * wHi);
  gaussianFill(nf, rand);
  for (let i = 0; i < noise.length; i++) noise[i] = nf[i];

  const xiNew = new Float64Array(C * hHi * wHi);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < hLo; y++) {
      const src = c * hLo * wLo + y * wLo,
        dst = c * hHi * wHi + y * wHi;
      for (let xx = 0; xx < wLo; xx++) xiNew[dst + xx] = xi[src + xx];
    }
    for (let y = 0; y < hHi; y++) {
      const dstRow = c * hHi * wHi + y * wHi;
      const xStart = y < hLo ? wLo : 0;
      for (let xx = xStart; xx < wHi; xx++)
        xiNew[dstRow + xx] += sigmaVal * noise[dstRow + xx];
    }
  }
  const xNew = _idct2(xiNew, C, hHi, wHi);
  for (let i = 0; i < xNew.length; i++) xNew[i] *= kappa;
  return { data: xNew, h: hHi, w: wHi, sigmaAligned };
}

// SPEED 멀티해상도 euler. denoise(xf32, sigma, curH, curW) 호출.
// stages: 오름차순 scale 리스트(1.0 끝). transSig: 핸드오프 σ(len=stages-1).
// onStep(i, s, sn, den, curH, curW) — i는 전체 누적 인덱스.
async function sampleSpeed(denoise, x0, sigmas, rand, onStep, opts) {
  const { stages, transSig, fullH, fullW, C, patch } = opts;
  let curScale = stages[0];
  let curH = _snap(fullH * curScale, patch),
    curW = _snap(fullW * curScale, patch);

  // 초기 저해상도로 다운 (스케일<1이면)
  let x;
  if (curScale < 1.0) {
    const low = _dctLowpassInit(x0, C, fullH, fullW, curScale, patch);
    x = new Float32Array(low.data.length);
    for (let i = 0; i < x.length; i++) x[i] = low.data[i];
    curH = low.h;
    curW = low.w;
  } else {
    x = x0;
  }

  const sig = sigmas.slice(); // 가변 복사 (re-space)
  let stageIdx = 0;
  const n = sig.length - 1;

  for (let i = 0; i < n; i++) {
    // 해상도 핸드오프: σ가 임계 이하로 떨어지면 확장
    while (stageIdx < transSig.length && sig[i] <= transSig[stageIdx]) {
      const scaleLo = stages[stageIdx],
        scaleHi = stages[stageIdx + 1];
      const old = sig[i];
      // x를 Float64로 올려 확장
      const xLo = new Float64Array(x.length);
      for (let k = 0; k < x.length; k++) xLo[k] = x[k];
      const ex = _spectralExpand(
        xLo,
        C,
        curH,
        curW,
        old,
        scaleLo,
        scaleHi,
        fullH,
        fullW,
        patch,
        rand,
      );
      x = new Float32Array(ex.data.length);
      for (let k = 0; k < x.length; k++) x[k] = ex.data[k];
      curH = ex.h;
      curW = ex.w;
      curScale = scaleHi;
      // 남은 σ re-space (Sec 4.3)
      const nv = ex.sigmaAligned;
      if (old > 0 && nv !== old)
        for (let j = i + 1; j <= n; j++) sig[j] = nv * (sig[j] / old);
      sig[i] = nv;
      stageIdx++;
    }

    const s = sig[i],
      sn = sig[i + 1];
    const den = await denoise(x, s, curH, curW);
    if (sn === 0) x.set(den);
    else
      for (let k = 0; k < x.length; k++)
        x[k] += ((sn - s) * (x[k] - den[k])) / s;
    if ((await onStep(i, s, sn, den, curH, curW)) === false) return null;
  }
  return x; // 풀 해상도 latent
}

export { sampleSpeed };
