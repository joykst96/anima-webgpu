// 골든 하네스 — 리팩터링 전후 동작 동등성 판정용 (브라우저 콘솔에서 로드).
//   const h = await import("/__tools/harness.mjs?v=1");
//   h.start(h.CASES)            // 백그라운드 실행 → window.__golden
//   window.__golden.status / .results
//   await h.compareToGolden()   // tools/golden.json과 비교 (없으면 처음 한 번 await h.saveGolden()으로 이 환경 기준 저장)
//
// 페이지 코드에 손대지 않고 UI(DOM)만으로 구동한다. 판정은 생성 이미지의 픽셀 SHA-256.
// 생성 외 기능은 별도 프로브(자동완성 결과 목록)로 비교한다.

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// 숨겨진 페이지에서도 스로틀되지 않는 양보 (MessageChannel 태스크 반복)
async function yieldFor(ms) {
  const t0 = performance.now(), ch = new MessageChannel();
  while (performance.now() - t0 < ms) await new Promise((r) => { ch.port1.onmessage = r; ch.port2.postMessage(0); });
}
// 페이지가 실제로 로드한 ORT(?ort= 버전 포함)와 같은 모듈 인스턴스
const ORT_URL = "/js/ort.js";

// ── ORT 호출 시간 계측 (페이지와 같은 모듈 인스턴스를 import해 prototype을 감싼다) ──
export async function installTiming() {
  const { ort } = await import(ORT_URL);
  window.__bench ??= { runs: [], creates: [] };
  const P = ort.InferenceSession.prototype;
  if (P.__patched) return;
  const origRun = P.run;
  P.run = async function (...a) {
    const n = this.inputNames || [];
    const kind = n.includes("timesteps") ? "dit" : n.includes("qwen_hidden_states") ? "adapter"
      : n.includes("input_ids") ? "te" : n.includes("latent") ? "vae" : "?";
    const t0 = performance.now();
    const r = await origRun.apply(this, a);
    window.__bench.runs.push({ kind, ms: performance.now() - t0, at: t0 });
    return r;
  };
  const origCreate = ort.InferenceSession.create;
  ort.InferenceSession.create = async function (...a) {
    const t0 = performance.now();
    const s = await origCreate.apply(this, a);
    window.__bench.creates.push({ inputs: s.inputNames.join(","), ms: performance.now() - t0 });
    return s;
  };
  P.__patched = true;
}

// ── 결정적 합성 LoRA (.safetensors) ──
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rand, n, std) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const r = Math.sqrt(-2 * Math.log(Math.max(rand(), 1e-12))) * std, th = 2 * Math.PI * rand();
    out[i] = r * Math.cos(th);
    if (i + 1 < n) out[i + 1] = r * Math.sin(th);
  }
  return out;
}
function encode(f32, dtype) {
  if (dtype === "F32") return new Uint8Array(f32.buffer);
  if (dtype === "F16") return new Uint8Array(new Float16Array(f32).buffer);
  // BF16: 상위 16비트 절단
  const u32 = new Uint32Array(f32.buffer), u16 = new Uint16Array(f32.length);
  for (let i = 0; i < f32.length; i++) u16[i] = u32[i] >>> 16;
  return new Uint8Array(u16.buffer);
}
function safetensors(tensors) {
  const header = {}, chunks = [];
  let off = 0;
  for (const [name, { dtype, shape, bytes }] of tensors) {
    header[name] = { dtype, shape, data_offsets: [off, off + bytes.length] };
    chunks.push(bytes);
    off += bytes.length;
  }
  let hj = JSON.stringify(header);
  hj += " ".repeat((8 - ((8 + hj.length) % 8)) % 8);
  const hb = new TextEncoder().encode(hj);
  const out = new Uint8Array(8 + hb.length + off);
  new DataView(out.buffer).setBigUint64(0, BigInt(hb.length), true);
  out.set(hb, 8);
  let p = 8 + hb.length;
  for (const c of chunks) { out.set(c, p); p += c.length; }
  return out;
}
const LORA_MODS = /^b([0-5])_(self_attn|cross_attn|mlp)_/; // 블록 0~5의 attn/mlp만 (adaLN 제외)
export async function makeSyntheticLora({ seed, rank, dtype, name }) {
  // 모듈 목록·모양은 모든 Anima 2B 슬롯 그래프가 같다 — 배포 목록(config.js)의 첫 슬롯 그래프 것을 쓴다
  const { MODELS } = await import("/js/config.js");
  const slot = Object.values(MODELS).find((m) => m.ditLora).ditLora;
  const mani = await (await fetch(slot + ".lora_manifest.json")).json();
  const rand = mulberry32(seed), tensors = [];
  for (const e of mani.entries) {
    if (!LORA_MODS.test(e.key)) continue;
    const base = "diffusion_model." + e.key.replace(/^b(\d+)_/, "blocks_$1_");
    tensors.push([base + ".lora_down.weight", { dtype, shape: [rank, e.K], bytes: encode(gauss(rand, rank * e.K, 1 / Math.sqrt(e.K)), dtype) }]);
    tensors.push([base + ".lora_up.weight", { dtype, shape: [e.N, rank], bytes: encode(gauss(rand, e.N * rank, 0.02), dtype) }]);
    tensors.push([base + ".alpha", { dtype: "F32", shape: [], bytes: encode(Float32Array.of(rank / 2), "F32") }]);
  }
  // 리포트 경로(스킵 집계)도 타게 하는 키: TE LoRA 1개, 미지원(lokr) 1개
  tensors.push(["lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight", { dtype: "F32", shape: [1, 4], bytes: encode(new Float32Array(4), "F32") }]);
  tensors.push(["diffusion_model.blocks.0.self_attn.q_proj.lokr_w1", { dtype: "F32", shape: [2, 2], bytes: encode(new Float32Array(4), "F32") }]);
  return new File([safetensors(tensors)], name, { type: "application/octet-stream" });
}

// ── UI 구동 ──
const doneCount = () => ($("log").innerText.match(/완료 —|생성 실패|생성 중단됨/g) || []).length;
function setVal(id, v) {
  const el = $(id);
  el.value = v;
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
}
function setChk(id, on) {
  const el = $(id);
  if (el.checked !== on) { el.checked = on; el.dispatchEvent(new Event("change", { bubbles: true })); }
}
// DOM 변화로 깨어나는 대기. 숨겨진 페이지에선 타이머가 분 단위로 몰려 실행되므로
// setTimeout 폴링 대신 MutationObserver를 쓴다(타임아웃 가드만 타이머).
function waitFor(pred, timeoutMs, what) {
  if (pred()) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const mo = new MutationObserver(() => {
      if (pred()) { mo.disconnect(); clearTimeout(to); resolve(); }
    });
    mo.observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true });
    const to = setTimeout(() => { mo.disconnect(); reject(new Error(`timeout: ${what}`)); }, timeoutMs);
  });
}

// Image.decode()는 패널이 숨겨지면 보류된다 → 렌더링과 무관한 createImageBitmap으로 디코드
async function decodePixels(blob) {
  const im = await createImageBitmap(blob);
  const c = new OffscreenCanvas(im.width, im.height), cx = c.getContext("2d");
  cx.drawImage(im, 0, 0);
  return cx.getImageData(0, 0, im.width, im.height);
}
async function pixelSha(data) {
  const d = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(d)].slice(0, 8).map((b) => b.toString(16).padStart(2, "0")).join("");
}
async function newestHash() {
  const img = document.querySelector("#gallery .thumb img");
  const png = await (await fetch(img.src)).blob();
  const px = await decodePixels(png);
  const r = { wh: `${px.width}x${px.height}`, sha: await pixelSha(px.data) };
  Object.defineProperty(r, "png", { value: png }); // 기준 저장용 (JSON 직렬화엔 안 나옴)
  return r;
}

async function setLoras(list) {
  // 기존 행 전부 제거 후 필요한 만큼 추가
  for (const row of [...$("loraList").children]) row.querySelector("button").click();
  for (const { file, weight } of list) {
    $("btnLoraAdd").click();
    const row = $("loraList").lastElementChild;
    const input = row.querySelector('input[type="file"]');
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
    const range = row.querySelector('input[type="range"]');
    range.value = weight;
    range.dispatchEvent(new Event("input", { bubbles: true }));
    range.dispatchEvent(new Event("change", { bubbles: true }));
  }
  const btn = $("btnLoraApply");
  // 페이지의 file.arrayBuffer() 완료 대기. 행 삭제가 이미 버튼을 켰을 수 있어 DOM으로는 구분 불가 →
  // 스로틀되지 않는 MessageChannel 양보로 시간을 보낸다.
  await yieldFor(500);
  if (btn.disabled) return $("loraStat").textContent; // 변경 없음
  btn.click();
  await waitFor(() => btn.textContent !== "반영 중..." && !$("btnGen").disabled, 300000, "LoRA 반영");
  return $("loraStat").textContent;
}

const P0 = "masterpiece, best quality, score_7, safe, 1girl, solo, silver hair, long hair, blue eyes, school uniform, cherry blossoms, outdoors, smile";
const N0 = "worst quality, low quality, score_1, score_2, score_3, artist name, nsfw";
const BASE = { prompt: P0, negative: N0, w: 1024, h: 1024, steps: 8, cfg: 1, sampler: "euler", scheduler: "simple", seed: 12345, negpip: true, speed: false, loras: [] };
export const CASES = [
  { name: "turbo-euler-negpip" },
  { name: "er_sde-weights", sampler: "er_sde", prompt: "masterpiece, best quality, score_7, safe, 1girl, solo, (silver hair:1.3), long hair, ((blue eyes)), school uniform, cherry blossoms, outdoors, \\(smile\\)" },
  { name: "cfg2-uncond", cfg: 2, negpip: false },
  { name: "portrait-no-negpip", w: 832, h: 1216, negpip: false, seed: 777 },
  { name: "normal-scheduler", scheduler: "normal", seed: 31337 },
  { name: "speed", speed: true },
  { name: "lora-single", loras: [{ synth: { seed: 1, rank: 4, dtype: "F16", name: "synthA.safetensors" }, weight: 0.8 }] },
  { name: "lora-multi", loras: [{ synth: { seed: 1, rank: 4, dtype: "F16", name: "synthA.safetensors" }, weight: 0.7 }, { synth: { seed: 2, rank: 8, dtype: "BF16", name: "synthB.safetensors" }, weight: 0.5 }] },
  { name: "lora-cleared", loras: [] },
];

export async function runCase(c0) {
  const c = { ...BASE, ...c0 };
  let loraStat = null;
  const wantLora = JSON.stringify(c.loras.map((l) => [l.synth, l.weight]));
  if (window.__goldenLoraKey !== wantLora) {
    const list = [];
    for (const l of c.loras) list.push({ file: await makeSyntheticLora(l.synth), weight: l.weight });
    loraStat = await setLoras(list);
    window.__goldenLoraKey = wantLora;
  }
  setChk("speedChk", c.speed); // 켜면 1024·1:1로 리셋되므로 해상도보다 먼저
  if (!c.speed) { setVal("width", c.w); setVal("height", c.h); }
  setVal("steps", c.steps);
  setVal("cfg", c.cfg); // cfg 리스너가 NegPip 자동 체크를 건드리므로 negpip보다 먼저
  setChk("negpipChk", c.negpip);
  $("sampler").value = c.sampler;
  $("scheduler").value = c.scheduler;
  setVal("seed", c.seed);
  setVal("batch", 1);
  $("prompt").value = c.prompt;
  $("negative").value = c.negative;
  $("previewChk").checked = false;

  window.__bench.runs.length = 0;
  const before = doneCount(), t0 = performance.now(), wall0 = Date.now();
  $("btnGen").click();
  await waitFor(() => doneCount() > before, 300000, `생성 ${c.name}`);
  const ms = performance.now() - t0;
  const lastLine = $("log").innerText.split("\n").filter((l) => /완료 —|생성 실패|생성 중단됨/.test(l)).pop();
  if (!/완료 —/.test(lastLine)) throw new Error(`${c.name}: ${lastLine}`);
  const runs = window.__bench.runs;
  const sum = (k) => runs.filter((r) => r.kind === k).reduce((a, r) => a + r.ms, 0);
  const img = await newestHash();
  const r = {
    name: c.name, wh: img.wh, sha: img.sha, ms: Math.round(ms),
    dit: Math.round(sum("dit")), ditCalls: runs.filter((r) => r.kind === "dit").length, vae: Math.round(sum("vae")),
    te: Math.round(sum("te")), adapter: Math.round(sum("adapter")),
    ditSteps: runs.filter((r) => r.kind === "dit").map((r) => Math.round(r.ms)), loraStat,
    wall: [wall0, Date.now()], // VRAM 로그(nvidia-smi) 대조용
  };
  Object.defineProperty(r, "png", { value: img.png });
  return r;
}

// ── 자동완성 프로브: 질의별 드롭다운 상위 태그 ──
export const AC_QUERIES = ["1gi", "long_h", "silver hair", "school u", "머리카락", "장발", "교복", "긴 머리"];
export async function probeAutocomplete(queries = AC_QUERIES) {
  const ta = $("prompt"), pop = $("acPopPrompt"), saved = ta.value, out = {};
  // 창에 포커스가 없으면 focus()가 focus 이벤트를 안 쏜다 → 데이터 로드 트리거를 직접 발생
  ta.focus();
  ta.dispatchEvent(new FocusEvent("focus"));
  await waitFor(() => /태그 자동완성 (준비됨|로드 실패)/.test($("log").innerText), 120000, "자동완성 데이터");
  for (const q of queries) {
    ta.value = q;
    ta.setSelectionRange(q.length, q.length);
    ta.dispatchEvent(new Event("input", { bubbles: true }));
    await sleep(400); // debounce 150ms + 렌더
    out[q] = [...pop.querySelectorAll(".ac-tag")].slice(0, 10).map((e) => e.textContent).join(" | ");
  }
  ta.value = saved;
  ta.blur();
  return out;
}

// ── 백그라운드 실행 ──
export function start(cases = CASES, { autocomplete = true } = {}) {
  const g = (window.__golden = { status: "running", results: [], error: null, ac: null });
  (async () => {
    try {
      await installTiming();
      if (!sess_ready()) throw new Error("모델 미로드 — 먼저 '모델 로드'");
      for (const c of cases) { g.status = `running ${c.name}`; g.results.push(await runCase(c)); }
      if (autocomplete) { g.status = "running autocomplete"; g.ac = await probeAutocomplete(); }
      g.status = "done";
    } catch (e) {
      g.error = String(e && e.stack || e);
      g.status = "error";
    }
  })();
  return g;
}
const sess_ready = () => !$("btnGen").disabled;

// ── 성능 기준선: 케이스마다 워밍업 warm회(셰이더 특화) + reps회 측정, 단계별 중앙값 ──
// 판정 기준은 패널 표시 상태 + 웜 3회 중앙값. 모델은 미리 로드돼 있어야 한다.
//   h.bench([{ name: "t1024" }, { name: "t1280", w: 1280, h: 1280 }]) → window.__benchRes
const median = (a) => { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
export function bench(cases, { reps = 3, warm = 1 } = {}) {
  const b = (window.__benchRes = { status: "running", rows: [], raw: [], error: null });
  (async () => {
    try {
      await installTiming();
      for (const c of cases) {
        const runs = [];
        for (let i = 0; i < warm + reps; i++) {
          b.status = `running ${c.name} ${i + 1}/${warm + reps}`;
          const r = await runCase(c);
          b.raw.push(Object.assign(r, { warm: i < warm })); // 복사하면 png(비열거)가 빠진다 → compareToReference({results: b.raw})
          if (i >= warm) runs.push(r);
        }
        const med = (k) => median(runs.map((r) => r[k]));
        const steps = runs.flatMap((r) => r.ditSteps.slice(1)); // 첫 스텝 제외 (스텝 간 편차 확인용)
        b.rows.push({
          name: c.name, wh: runs[0].wh, sha: [...new Set(runs.map((r) => r.sha))].join(","), // 결정적이면 1개
          ms: med("ms"), te: med("te"), adapter: med("adapter"), dit: med("dit"), ditCalls: runs[0].ditCalls,
          ditStep: Math.round(median(steps)), vae: med("vae"), msRange: [Math.min(...runs.map((r) => r.ms)), Math.max(...runs.map((r) => r.ms))],
        });
      }
      b.status = "done";
    } catch (e) {
      b.error = String(e && e.stack || e);
      b.status = "error";
    }
  })();
  return b;
}

// 새로 연 페이지에서: 계측 설치 → 모델 로드 → 전체 케이스 실행 (백그라운드)
export function full(cases = CASES) {
  const g = (window.__golden = { status: "loading", results: [], error: null, ac: null });
  (async () => {
    try {
      await installTiming();
      await waitFor(() => !$("btnLoad").disabled && /WebGPU OK/.test($("gpuInfo").innerText), 30000, "GPU 확인");
      $("btnLoad").click();
      await waitFor(() => /로드 완료|로드 실패/.test($("log").innerText), 600000, "모델 로드");
      if (/로드 실패/.test($("log").innerText)) throw new Error("모델 로드 실패");
      const sub = start(cases);
      await waitFor(() => !/running/.test(sub.status), 1800000, "케이스 실행");
    } catch (e) {
      g.error = String(e && e.stack || e);
      g.status = "error";
    }
  })();
  return g;
}

// 골든(tools/golden.json)은 환경별 기준이다 — 빌드·ORT·GPU/드라이버가 다르면 해시도 다르므로 레포에 넣지 않고,
// 각 환경에서 처음 한 번 만든다: h.full() 완료 후 await h.saveGolden({ note: "무엇을 기준으로 삼았는지" })
async function loadGolden() {
  const res = await fetch("/__tools/golden.json", { cache: "no-store" });
  if (!res.ok)
    throw new Error("tools/golden.json 없음 — 이 환경의 기준을 먼저 만드세요: h.full() 완료 후 await h.saveGolden()");
  return res.json();
}
export async function saveGolden(cur = window.__golden, { note = "", force = false } = {}) {
  if (cur.status !== "done") throw new Error(`실행이 끝나지 않음 (${cur.status})`);
  const exists = (await fetch("/__tools/golden.json", { method: "HEAD", cache: "no-store" })).ok;
  if (exists && !force) throw new Error("tools/golden.json이 이미 있음 — 기준을 바꾸려면 force: true");
  const golden = {
    date: new Date().toISOString().slice(0, 10),
    env: navigator.userAgent,
    note,
    results: cur.results.map(({ name, wh, sha, ms, dit, ditCalls, vae }) => ({ name, wh, sha, ms, dit, ditCalls, vae })),
    ac: cur.ac,
  };
  const res = await fetch("/__tools/golden.json", { method: "PUT", body: JSON.stringify(golden, null, 1) });
  if (!res.ok) throw new Error(`저장 실패 ${res.status}`);
  return golden;
}

// 결과 비교: 골든과 현재 결과의 sha/ac 일치 여부
export async function compareToGolden(cur = window.__golden) {
  return compare(await loadGolden(), cur);
}
export function compare(golden, cur) {
  const rows = golden.results.map((g) => {
    const c = cur.results.find((r) => r.name === g.name);
    return { name: g.name, golden: g.sha, current: c?.sha ?? "-", same: c?.sha === g.sha, ms: c?.ms };
  });
  const acSame = golden.ac && cur.ac ? Object.keys(golden.ac).filter((q) => golden.ac[q] !== cur.ac[q]) : null;
  return { allSame: rows.every((r) => r.same) && (!acSame || acSame.length === 0), rows, acDiff: acSame };
}

// ── 기준 이미지 / PSNR — 수치가 바뀌는 변경(ORT 버전·양자화·커널 경로)의 판정용 ──
// 기준 PNG는 tools/ref/<케이스>.png (devserver PUT, git 미포함). 골든 해시와 같은 이미지만 기준으로 받는다.
//   h.full() 완료 후 → await h.saveReference()      (현재 결과를 기준으로 저장)
//   수치 변경 후 h.full() → await h.compareToReference()
export async function saveReference(cur = window.__golden, { force = false } = {}) {
  const golden = await loadGolden();
  const out = [];
  for (const r of cur.results) {
    if (!r.png) throw new Error(`${r.name}: 이미지 없음 (이번 페이지에서 실행한 결과만 저장 가능)`);
    const g = golden.results.find((x) => x.name === r.name);
    if (!force && g?.sha !== r.sha) throw new Error(`${r.name}: 골든 해시 불일치 (${r.sha} ≠ ${g?.sha}) — force로만 저장`);
    const res = await fetch(`/__tools/ref/${r.name}.png`, { method: "PUT", body: r.png });
    if (!res.ok) throw new Error(`${r.name}: 저장 실패 ${res.status}`);
    out.push({ name: r.name, sha: r.sha, bytes: r.png.size });
  }
  return out;
}

// RGB 채널만 비교 (알파 제외). a, b: RGBA Uint8ClampedArray, 같은 크기.
export function imageMetrics(a, b) {
  if (a.length !== b.length) return { psnr: NaN, maxErr: NaN, meanAbs: NaN, diffPx: NaN, note: "크기 다름" };
  let se = 0, ae = 0, maxErr = 0, diffPx = 0;
  const n = a.length / 4;
  for (let p = 0; p < a.length; p += 4) {
    let any = 0;
    for (let k = 0; k < 3; k++) {
      const d = Math.abs(a[p + k] - b[p + k]);
      se += d * d;
      ae += d;
      if (d > maxErr) maxErr = d;
      any |= d;
    }
    if (any) diffPx++;
  }
  const mse = se / (n * 3);
  return {
    psnr: mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse),
    maxErr, meanAbs: ae / (n * 3), diffPx: diffPx / n,
  };
}

export async function compareToReference(cur = window.__golden) {
  const rows = [];
  for (const r of cur.results) {
    const res = await fetch(`/__tools/ref/${r.name}.png`, { cache: "no-store" });
    if (!res.ok) { rows.push({ name: r.name, note: "기준 없음" }); continue; }
    const ref = await decodePixels(await res.blob());
    const now = await decodePixels(r.png);
    const m = ref.width === now.width && ref.height === now.height
      ? imageMetrics(ref.data, now.data) : { note: `크기 다름 ${ref.width}x${ref.height}` };
    rows.push({
      name: r.name, refSha: await pixelSha(ref.data), sha: r.sha,
      psnr: m.psnr === Infinity ? "∞" : +m.psnr?.toFixed(2), maxErr: m.maxErr,
      meanAbs: m.meanAbs === undefined ? undefined : +m.meanAbs.toFixed(4),
      diffPx: m.diffPx === undefined ? undefined : +(m.diffPx * 100).toFixed(3) + "%", note: m.note,
    });
  }
  return rows;
}
