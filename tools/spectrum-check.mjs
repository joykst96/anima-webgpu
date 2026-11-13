// web/js/spectrum.js를 Node에서 돌려 원 구현(ComfyUI comfyui-anima-kludge spectrum.py의 numpy 재서술)과 대조한다. 모델 불필요.
//   node tools/spectrum-check.mjs
// 기준: tools/spectrum-ref-30steps.json — 30스텝·기본 파라미터에서 스텝별 모드, 창, 이력, 가중치 c(소수 9자리).
// 확인하는 것: ① 스텝별 실제/예측 배치와 창 값이 같다 ② 가중치 c가 절대 오차 1e-8 이내 ③ 20·50스텝의 실제/예측 수
//              ④ 페이지가 쓰는 모멘트 형태(특징 누적)의 예측이 가중합 형태(Σ c_i·h_i)와 같다(특징 크기를 실제와 비슷하게 준 난수)
import fs from "node:fs";
import path from "node:path";
import { SPECTRUM, createSpectrum, forecastWeights, predictedSteps } from "../web/js/spectrum.js";

const ref = JSON.parse(fs.readFileSync(path.join(import.meta.dirname, "spectrum-ref-30steps.json"), "utf8"));
let fail = 0;
const check = (ok, msg) => { if (!ok) { fail++; console.log("FAIL", msg); } };

// ①② 스케줄과 가중치
{
  const N = ref.params.steps, sp = createSpectrum(N);
  let window = SPECTRUM.window, prevMode = "actual", prev = -1, maxErr = 0, nPred = 0;
  const ts = [];
  for (const row of ref.rows) {
    if (prevMode === "actual" && prev >= SPECTRUM.warmup) window = Math.round((window + SPECTRUM.flex) * 1000) / 1000; // 기준의 창과 비교용
    const mode = sp.beginStep(row.step);
    check(mode === row.mode, `step ${row.step}: mode ${mode} ≠ ${row.mode}`);
    check(Math.abs(window - row.window) < 1e-9, `step ${row.step}: window ${window} ≠ ${row.window}`);
    if (mode === "cached") {
      check(JSON.stringify(ts) === JSON.stringify(row.hist_t), `step ${row.step}: 이력 ${ts} ≠ ${row.hist_t}`);
      const c = forecastWeights(ts, row.step, N);
      c.forEach((v, i) => { maxErr = Math.max(maxErr, Math.abs(v - row.c[i])); });
      check(Math.abs(c.reduce((a, b) => a + b, 0) - row.sum_c) < 1e-8, `step ${row.step}: Σc`);
      nPred++;
    } else ts.push(row.step);
    prevMode = mode; prev = row.step;
  }
  check(maxErr < 1e-8, `가중치 최대 오차 ${maxErr}`);
  console.log(`30스텝: 실제 ${sp.stats.actual} · 예측 ${nPred} (기준 17 · 13), 가중치 c 최대 절대 오차 ${maxErr.toExponential(2)}`);
  check(sp.stats.actual === 17 && nPred === 13, "30스텝 실제/예측 수");
}
// ③ 다른 스텝 수 (원본 로그: 20 → 14 full / 6 predicted, 50 → 22 / 28, 8스텝은 예측 없음)
for (const [N, full, pred] of [[20, 14, 6], [50, 22, 28], [8, 8, 0]]) {
  const sp = createSpectrum(N);
  for (let i = 0; i < N; i++) sp.beginStep(i);
  console.log(`${N}스텝: 실제 ${sp.stats.actual} · 예측 ${sp.stats.predicted.length} (기준 ${full} · ${pred})`);
  check(sp.stats.actual === full && sp.stats.predicted.length === pred, `${N}스텝 실제/예측 수`);
}
// ④ 모멘트 형태 = 가중합 형태. 특징은 fp32, 크기는 실제 특징(rms 약 1,200, 최대 약 4만)과 비슷하게.
{
  const N = 30, n = 20000, sp = createSpectrum(N);
  let seed = 7; const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296 - 0.5);
  const base = Float64Array.from({ length: n }, () => rnd() * 4000), drift = Float64Array.from({ length: n }, () => rnd() * 300);
  const hist = [], ts = []; let worst = 0;
  for (let i = 0; i < N; i++) {
    const mode = sp.beginStep(i);
    if (mode === "actual") {
      const h = Float32Array.from({ length: n }, (_, x) => base[x] + drift[x] * i + 40000 * (x === 3) + rnd() * 50);
      sp.store("cond", h); hist.push(h); ts.push(i);
    } else {
      const got = sp.predict("cond", i), c = forecastWeights(ts, i, N);
      let num = 0, den = 0;
      for (let x = 0; x < n; x++) { let e = 0; for (let k = 0; k < c.length; k++) e += c[k] * hist[k][x]; num += (got[x] - e) ** 2; den += e * e; }
      worst = Math.max(worst, Math.sqrt(num / den));
    }
  }
  console.log(`모멘트 형태 vs 가중합 형태: 상대 오차 최대 ${worst.toExponential(2)} (fp32 누적)`);
  check(worst < 1e-5, `모멘트 형태 오차 ${worst}`);
}
check(predictedSteps(8).length === 0 && predictedSteps(30).join() === "6,8,10,12,14,15,17,18,20,21,23,24,26", "predictedSteps");
console.log(fail ? `실패 ${fail}건` : "모두 통과");
process.exit(fail ? 1 : 0);
