// Spectrum — 건너뛰는 스텝에서 DiT 블록 전체를 생략하고, 저장해 둔 특징으로 그 스텝의 특징을 예측해 머리(출력층)만 돌린다.
//   논문: Han et al., "Spectrum" (arXiv:2603.01623). 규칙·수식의 원본: ComfyUI-Spectrum-KSampler (sorryhyun, MIT)
//   https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler — 같은 규칙을 다시 쓴 comfyui-anima-kludge의 spectrum.py(MIT)를 옮겼다.
//
// 특징 = final_layer에 들어가는 토큰 특징(마지막 블록 출력). DiT 그래프를 "본체(+특징 출력)"와 "머리(특징, timesteps → v_pred)"로
// 나눠 둔 경우에만 쓸 수 있다(export/webnn/spectrum_split.py). v_pred를 직접 외삽하는 것과는 다르다 — 머리는 스텝마다 다른 σ로
// 특징을 변조하므로, 예측한 특징을 그 스텝의 σ로 머리에 통과시켜야 한다.
//
// 스케줄은 내용과 무관한 고정 규칙이다: 앞 warmup개·뒤 tail개 스텝은 항상 실제 계산, 그 사이는 "실제 1회 → 예측 floor(window)−1회"를
// 반복하며 실제 계산 때마다 window가 flex만큼 커진다. 기본값이면 30스텝 = 실제 17 + 예측 13.
//
// 예측식(스텝 번호 t, 총 스텝 수 N):
//   τ(t) = 2t/N − 1,  φ(t) = [T_0(τ) … T_m(τ)] (체비쇼프 1종),  X = 저장된 스텝들의 φ를 쌓은 행렬
//   체비쇼프 릿지 회귀:  ĥ_cheb = Σ_i c_i·h_i,  c = X·(XᵀX + λI)⁻¹·φ(t*)
//   1차 테일러:         ĥ_taylor = (1 + k)·h_n − k·h_{n−1},  k = (t* − t_n) / (t_n − t_{n−1})
//   ĥ = blend·ĥ_cheb + (1 − blend)·ĥ_taylor
// 실행 형태는 둘이다(queue.js가 고른다).
//   GPU: 특징을 본체가 낸 GPU 텐서(fp16) 그대로 최근 K개 들고 있다가 "가중합 머리" 그래프에 넣는다. 여기서는 계수 c만 계산한다(weights).
//        JS가 특징 값을 만지지 않는다. K(그래프의 입력 수, 16)가 실제 스텝 수보다 작아지면 오래된 것부터 버린다(원본의 history_size와 같은 뜻).
//   CPU: 가중합 머리가 없을 때. 특징(fp32)을 읽어 와 모멘트 S_j = Σ_i T_j(τ_i)·h_i (j = 0…m)를 누적하고(store), 예측 때
//        ĥ_cheb = Σ_j a_j·S_j,  a = (XᵀX + λI)⁻¹·φ(t*) 로 합산해(predict) 머리에 올린다. 위 가중합과 같은 식이고 이력을 버리지 않는다.
//        1024² 30스텝 기준 GPU 15.0초, CPU 17.8초(끄면 24.0초).
const SPECTRUM = { warmup: 6, window: 2.0, flex: 0.25, tail: 3, degree: 3, lam: 0.1, blend: 0.3 };

function chebyshev(tau, m) {
  const t = [1, tau];
  for (let k = 2; k <= m; k++) t.push(2 * tau * t[k - 1] - t[k - 2]);
  return t.slice(0, m + 1);
}
// 작은 선형계 A·z = b (가우스 소거, 부분 피벗). A는 (m+1)×(m+1) 양의 정부호.
function solve(A, b) {
  const n = b.length,
    M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const z = new Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * z[k];
    z[r] = s / M[r][r];
  }
  return z;
}
// a = (XᵀX + λI)⁻¹·φ(t*) 와 테일러 계수 k. ts = 실제로 계산한 스텝 번호들(오름차순).
function forecastCoeffs(ts, tStar, N, P = SPECTRUM) {
  const m = P.degree,
    phi = (t) => chebyshev((2 * t) / N - 1, m);
  const A = Array.from({ length: m + 1 }, (_, i) => Array.from({ length: m + 1 }, (_, j) => (i === j ? P.lam : 0)));
  for (const t of ts) {
    const f = phi(t);
    for (let i = 0; i <= m; i++) for (let j = 0; j <= m; j++) A[i][j] += f[i] * f[j];
  }
  const n = ts.length;
  return { a: solve(A, phi(tStar)), k: (tStar - ts[n - 1]) / Math.max(ts[n - 1] - ts[n - 2], 1e-8), phi };
}
// 이력 전체에 대한 가중치 c (ĥ = Σ c_i·h_i). 실행에는 쓰지 않는다 — 원본과의 대조용(tools/spectrum-check.mjs).
function forecastWeights(ts, tStar, N, P = SPECTRUM) {
  const { a, k, phi } = forecastCoeffs(ts, tStar, N, P);
  const n = ts.length;
  return ts.map((t, i) => {
    const cheb = phi(t).reduce((s, v, j) => s + v * a[j], 0);
    if (P.blend >= 1) return cheb;
    return P.blend * cheb + (1 - P.blend) * (i === n - 1 ? 1 + k : i === n - 2 ? -k : 0);
  });
}

// 샘플링 1회마다 새로 만든다. N = 스텝 수(시그마 개수 − 1). 갈래(branch) = "cond" / "uncond" — 특징을 따로 예측한다.
function createSpectrum(N, P = SPECTRUM) {
  const m = P.degree;
  let prev = -1,
    mode = "actual",
    window = P.window,
    consec = 0;
  const ts = []; // 실제로 계산한 스텝 번호 (갈래 공통)
  const branches = {}; // branch → { S: Float32Array[m+1], h: [직전, 그 전], out }
  const st = { actual: 0, predicted: [] };
  const ready = () => ts.length >= Math.max(2, m + 2);
  return {
    stats: st,
    // 최근 K개 실제 스텝에 대한 가중치 c (ĥ = Σ c_i·h_i). 특징을 GPU 텐서로 들고 있을 때: 호출부가 같은 K개 특징을 가중합 그래프에 넣는다.
    // K가 지금까지의 실제 스텝 수 이상이면 store/predict(모멘트 형태)와 같은 값이다.
    weights(i, K) {
      const recent = ts.slice(-K);
      return forecastWeights(recent, i, N, P);
    },
    // 스텝 i를 실제로 계산할지("actual") 예측할지("cached"). 스텝당 한 번만 부른다.
    beginStep(i) {
      if (mode === "actual") {
        if (prev >= P.warmup) window = Math.round((window + P.flex) * 1000) / 1000;
        consec = 0;
      } else consec += 1;
      prev = i;
      const cache = i >= P.warmup && i < N - P.tail && ready() && (consec + 1) % Math.max(1, Math.floor(window)) !== 0;
      mode = cache ? "cached" : "actual";
      if (cache) st.predicted.push(i + 1);
      else {
        st.actual++;
        ts.push(i);
      }
      return mode;
    },
    // 실제 스텝의 특징을 저장한다(복사하므로 feat 버퍼는 바로 놓아도 된다). beginStep(i)가 "actual"을 돌려준 스텝에서만,
    // 다음 beginStep 전에 부른다(그 스텝 번호의 체비쇼프 값으로 누적한다).
    store(branch, feat) {
      const n = feat.length;
      const b = (branches[branch] ??= {
        S: Array.from({ length: m + 1 }, () => new Float32Array(n)),
        h: [new Float32Array(n), new Float32Array(n)],
        out: null,
      });
      const T = chebyshev((2 * prev) / N - 1, m);
      b.h.reverse(); // h[0] = 가장 최근 특징
      const hl = b.h[0];
      if (m === 3) {
        const [S0, S1, S2, S3] = b.S,
          [, t1, t2, t3] = T; // T_0 = 1
        for (let x = 0; x < n; x++) {
          const f = feat[x];
          S0[x] += f;
          S1[x] += t1 * f;
          S2[x] += t2 * f;
          S3[x] += t3 * f;
          hl[x] = f;
        }
        return;
      }
      for (let j = 0; j <= m; j++) {
        const Sj = b.S[j],
          tj = T[j];
        for (let x = 0; x < n; x++) Sj[x] += tj * feat[x];
      }
      hl.set(feat);
    },
    // 예측 스텝 i의 특징. 돌려준 배열은 다음 predict 호출까지만 유효하다(갈래별 버퍼 재사용).
    predict(branch, i) {
      const b = branches[branch];
      const { a, k } = forecastCoeffs(ts, i, N, P);
      const n = b.h[0].length,
        out = (b.out ??= new Float32Array(n));
      const w = Math.min(1, P.blend),
        c1 = (1 - w) * (1 + k),
        c0 = -(1 - w) * k;
      const [h1, h0] = b.h;
      if (m === 3) {
        const [S0, S1, S2, S3] = b.S,
          a0 = w * a[0],
          a1 = w * a[1],
          a2 = w * a[2],
          a3 = w * a[3];
        for (let x = 0; x < n; x++) out[x] = a0 * S0[x] + a1 * S1[x] + a2 * S2[x] + a3 * S3[x] + c1 * h1[x] + c0 * h0[x];
        return out;
      }
      out.fill(0);
      for (let j = 0; j <= m; j++) {
        const Sj = b.S[j],
          aj = w * a[j];
        for (let x = 0; x < n; x++) out[x] += aj * Sj[x];
      }
      if (w < 1) for (let x = 0; x < n; x++) out[x] += c1 * h1[x] + c0 * h0[x];
      return out;
    },
  };
}

// N스텝에서 예측으로 건너뛰는 스텝 번호들(0부터). 스케줄이 내용과 무관해서 미리 알 수 있다 — 비면 Spectrum을 켤 이유가 없다.
function predictedSteps(N, P = SPECTRUM) {
  const sp = createSpectrum(N, P);
  for (let i = 0; i < N; i++) sp.beginStep(i);
  return sp.stats.predicted.map((s) => s - 1);
}

export { SPECTRUM, createSpectrum, forecastWeights, predictedSteps };
