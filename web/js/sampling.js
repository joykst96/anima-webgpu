// 시드 RNG, 시그마 스케줄, 샘플러(euler, er_sde) — run_pipeline.py 포팅
import { SHIFT } from "./constants.js";

// ───────── 시드 RNG + 가우시안 ─────────
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussianFill(arr, rand) {
  for (let i = 0; i < arr.length; i += 2) {
    const u1 = Math.max(rand(), 1e-12),
      u2 = rand();
    const r = Math.sqrt(-2 * Math.log(u1)),
      th = 2 * Math.PI * u2;
    arr[i] = r * Math.cos(th);
    if (i + 1 < arr.length) arr[i + 1] = r * Math.sin(th);
  }
}

// ───────── 스케줄러 (run_pipeline.py 포팅) ─────────
const shiftSigma = (t) => (SHIFT * t) / (1 + (SHIFT - 1) * t);
function simpleSchedule(steps) {
  const table = Array.from({ length: 1000 }, (_, j) =>
    shiftSigma((j + 1) / 1000),
  );
  const ss = table.length / steps,
    sigs = [];
  for (let x = 0; x < steps; x++)
    sigs.push(table[table.length - 1 - Math.floor(x * ss)]);
  sigs.push(0);
  return sigs;
}
// ComfyUI normal_scheduler (ModelSamplingDiscreteFlow, multiplier 1 → timestep(σ) = σ):
// σ_max(=1)~σ_min(=shift(1/1000))를 timestep으로 선형 분할한 뒤 sigma(t) = shift(t)를 한 번 더 적용하고 0을 붙인다.
// (σ에 shift가 두 번 걸리는 게 ComfyUI 동작 그대로다. 종전 구현 shift(1 − i/n)은 simple과 같은 값이었다)
function normalSchedule(steps) {
  const start = shiftSigma(1),
    end = shiftSigma(1 / 1000);
  const sigs = [];
  for (let i = 0; i < steps; i++) {
    const t = steps === 1 ? start : start + ((end - start) * i) / (steps - 1); // torch.linspace(start, end, steps)
    sigs.push(shiftSigma(t));
  }
  sigs.push(0);
  return sigs;
}
// ───────── 샘플러 (run_pipeline.py 포팅) ─────────
async function sampleEuler(denoise, x, sigmas, rand, onStep) {
  for (let i = 0; i < sigmas.length - 1; i++) {
    const s = sigmas[i],
      sn = sigmas[i + 1];
    const den = await denoise(x, s);
    if (sn === 0) x.set(den);
    else
      for (let k = 0; k < x.length; k++)
        x[k] += ((sn - s) * (x[k] - den[k])) / s;
    if ((await onStep(i, s, sn, den)) === false) return null; // 중단
  }
  return x;
}

async function sampleErSde(
  denoise,
  x,
  sigmas,
  rand,
  onStep,
  sNoise = 1.0,
  maxStage = 3,
) {
  sigmas = sigmas.slice();
  if (sigmas[0] >= 1.0) sigmas[0] = shiftSigma(1 - 1e-4); // logit 발산 방지
  const ns = (l) => l * (Math.exp(Math.pow(l, 0.3)) + 10);
  const lam = sigmas.map((s) => s / (1 - s)); // CONST: λ = σ/(1-σ)
  const NI = 200;
  let oldDen = null,
    oldDenD = null;
  const noise = new Float32Array(x.length);

  for (let i = 0; i < sigmas.length - 1; i++) {
    const den = await denoise(x, sigmas[i]);
    const stage = Math.min(maxStage, i + 1);
    if (sigmas[i + 1] === 0) {
      x.set(den);
    } else {
      const ls = lam[i],
        lt = lam[i + 1];
      const aS = sigmas[i] / ls,
        aT = sigmas[i + 1] / lt; // = 1-σ
      const rA = aT / aS,
        r = ns(lt) / ns(ls);
      const c1 = rA * r,
        c2 = aT * (1 - r);
      for (let k = 0; k < x.length; k++) x[k] = c1 * x[k] + c2 * den[k];

      if (stage >= 2) {
        const dt = lt - ls,
          step = -dt / NI;
        let sInt = 0,
          sUInt = 0;
        for (let p = 0; p < NI; p++) {
          const pos = lt + p * step,
            sp = ns(pos);
          sInt += 1 / sp;
          sUInt += (pos - ls) / sp;
        }
        sInt *= step;
        sUInt *= step;
        const dCoef = 1 / (ls - lam[i - 1]);
        const c3 = aT * (dt + sInt * ns(lt));
        const denD = new Float32Array(x.length);
        for (let k = 0; k < x.length; k++) {
          denD[k] = (den[k] - oldDen[k]) * dCoef;
          x[k] += c3 * denD[k];
        }
        if (stage >= 3) {
          const uCoef = 1 / ((ls - lam[i - 2]) / 2);
          const c4 = aT * ((dt * dt) / 2 + sUInt * ns(lt));
          for (let k = 0; k < x.length; k++)
            x[k] += c4 * (denD[k] - oldDenD[k]) * uCoef;
        }
        oldDenD = denD;
      }
      if (sNoise > 0) {
        const v = Math.max(lt * lt - ls * ls * r * r, 0);
        const c5 = aT * sNoise * Math.sqrt(v);
        gaussianFill(noise, rand);
        for (let k = 0; k < x.length; k++) x[k] += c5 * noise[k];
      }
    }
    oldDen = den;
    if ((await onStep(i, sigmas[i], sigmas[i + 1], den)) === false)
      return null; // 중단
  }
  return x;
}

// ComfyUI k_diffusion/sampling.py sample_cfgpp_ud10_ab (_sample_cfgpp_history: history_weight 0.25, zero_weight 1.0,
// zero_order 1, uncond_history_weight 0.1) 포팅 — 베이스 Anima용(ComfyUI 3ac5d79, CFG 2.0 권장).
// CFG++ Euler + 가변 간격 AB2 이력 + 마지막 스텝 σ=0 외삽. 결정적(잡음 없음).
// denoise 결과의 .uncond = 부정 프롬프트 denoised (CFG 1이면 긍정과 같음 — ComfyUI post_cfg와 같은 규칙).
// CONST(flow): half-logSNR λ = −logit(σ) → α = σ·e^λ = 1 − σ.
async function sampleCfgppUd10Ab(denoise, x, sigmas, rand, onStep) {
  const HISTORY_W = 0.25,
    UNCOND_HISTORY_W = 0.1;
  // linear_multistep_coeff(2, σ, i, j): 1차 라그랑주 기저의 [σ_i, σ_i+1] 적분 (ComfyUI는 quad, 1차라 닫힌 식과 같다)
  const ab2 = (i, j) => {
    const a = j === 0 ? sigmas[i - 1] : sigmas[i],
      b = j === 0 ? sigmas[i] : sigmas[i - 1];
    const s0 = sigmas[i] - a,
      s1 = sigmas[i + 1] - a;
    return (s1 * s1 - s0 * s0) / 2 / (b - a);
  };
  const derivs = [],
    denHist = [];
  let oldUncondD = null;
  for (let i = 0; i < sigmas.length - 1; i++) {
    const s = sigmas[i],
      sn = sigmas[i + 1],
      dt = sn - s;
    const den = await denoise(x, s);
    const unc = den.uncond || den;
    const aS = 1 - s,
      aT = 1 - sn;
    const curUncondD = new Float32Array(x.length);
    for (let k = 0; k < x.length; k++) curUncondD[k] = (x[k] - aS * unc[k]) / s;
    const ratio = i > 0 ? (UNCOND_HISTORY_W * dt) / (s - sigmas[i - 1]) : 0;
    const d = new Float32Array(x.length),
      eul = new Float32Array(x.length);
    for (let k = 0; k < x.length; k++) {
      const ud = ratio ? curUncondD[k] + ratio * (curUncondD[k] - oldUncondD[k]) : curUncondD[k];
      eul[k] = aT * den[k] + sn * ud - x[k];
      d[k] = eul[k] / dt;
    }
    derivs.push(d);
    if (derivs.length > 2) derivs.shift();
    if (derivs.length === 1) for (let k = 0; k < x.length; k++) x[k] += eul[k];
    else {
      const c0 = ab2(i, 0),
        c1 = ab2(i, 1),
        [dPrev, dCur] = derivs;
      for (let k = 0; k < x.length; k++) {
        const hist = c0 * dCur[k] + c1 * dPrev[k];
        x[k] += eul[k] + HISTORY_W * (hist - eul[k]);
      }
    }
    if (sn === 0 && denHist.length) {
      // σ=0 외삽 (zero_order 1, zero_weight 1 → x를 예측값으로 교체)
      const prev = denHist[denHist.length - 1],
        inv = 1 / (s - sigmas[i - 1]);
      for (let k = 0; k < x.length; k++) x[k] = den[k] - s * (den[k] - prev[k]) * inv;
    }
    denHist.push(den);
    if (denHist.length > 2) denHist.shift();
    oldUncondD = curUncondD;
    if ((await onStep(i, s, sn, den)) === false) return null; // 중단
  }
  return x;
}

export { gaussianFill, mulberry32, normalSchedule, sampleCfgppUd10Ab, sampleErSde, sampleEuler, simpleSchedule };
