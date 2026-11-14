// 플래그를 켠 별도 Chrome(임시 프로필)을 띄우고 CDP로 식 하나를 평가해 결과를 출력한다 (Node 22+).
// WebNN(navigator.ml)은 Chrome 157 기준 플래그 뒤라, 평소 프로필을 건드리지 않고 재려고 만든 것.
//   node tools/webnn/webnn-bench-run.mjs <url> <expr-file> [--features A,B] [--keep] [--timeout ms] [--port 9333] [--chrome <path>]
// 실행마다 새 프로필 폴더를 만들고 끝나면 프로세스 트리째 종료·삭제한다
// (같은 프로필을 곧바로 다시 띄우면 Chrome이 바로 종료돼 측정이 멈춘다 — 2026-11-13 확인).
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const url = args[0];
const expr = fs.readFileSync(args[1], "utf8");
const features = opt("--features", "WebMachineLearningNeuralNetwork");
const timeout = +opt("--timeout", "600000");
const PORT = +opt("--port", "9333");
const CHROME = opt("--chrome", "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe");
const profile = path.join(import.meta.dirname, "chrome-profile-" + Date.now());

const chrome = spawn(CHROME, [
  `--user-data-dir=${profile}`, `--remote-debugging-port=${PORT}`,
  "--no-first-run", "--no-default-browser-check", "--disable-background-timer-throttling",
  "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows",
  `--enable-features=${features}`, ...(opt("--extra", "") ? opt("--extra").split(" ") : []), url,
], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --mon: 측정 중 GPU 전체 사용량(nvidia-smi 200ms)과 이 Chrome(임시 프로필) 프로세스들의 private 메모리 합(1초)을 기록한다.
// 결과에 marks([{t, label}], Date.now() 기준)가 있으면 구간별 최댓값을 함께 출력한다.
const mon = { gpu: [], ram: [] };
const monProcs = [];
if (args.includes("--mon")) {
  const smi = spawn("nvidia-smi", ["--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "200"]);
  smi.stdout.on("data", (b) => { for (const l of String(b).split(/\r?\n/)) if (l.trim()) mon.gpu.push([Date.now(), +l.trim()]); });
  const ps = spawn("powershell", ["-NoProfile", "-Command",
    "while($true){ $p=Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | Where-Object { $_.CommandLine -like '*chrome-profile-*' }; $t=($p | Measure-Object PrivatePageCount -Sum).Sum; $g=($p | Where-Object { $_.CommandLine -like '*--type=gpu-process*' } | Measure-Object PrivatePageCount -Sum).Sum; $r=($p | Where-Object { $_.CommandLine -like '*--type=renderer*' } | Measure-Object PrivatePageCount -Sum).Sum; $tw=($p | Measure-Object WorkingSetSize -Sum).Sum; $gw=($p | Where-Object { $_.CommandLine -like '*--type=gpu-process*' } | Measure-Object WorkingSetSize -Sum).Sum; [Console]::Out.WriteLine(\"$t,$g,$r,$tw,$gw\"); [Console]::Out.Flush(); Start-Sleep -Milliseconds 1000 }"]);
  // 한 줄 = 전체, GPU 프로세스, 렌더러(페이지) 프로세스의 private 메모리(커밋, 바이트) + 전체·GPU 프로세스의 working set(물리 RAM)
  ps.stdout.on("data", (b) => { for (const l of String(b).split(/\r?\n/)) if (l.trim()) mon.ram.push([Date.now(), ...l.trim().split(",").map((x) => +x || 0)]); });
  monProcs.push(smi, ps);
}
function monSummary(marks) {
  const rows = [];
  for (let i = 0; i < marks.length; i++) {
    const t0 = marks[i].t, t1 = i + 1 < marks.length ? marks[i + 1].t : Infinity;
    const g = mon.gpu.filter(([t]) => t >= t0 && t < t1).map(([, v]) => v), rr = mon.ram.filter(([t]) => t >= t0 && t < t1), r = rr.map(([, v]) => v);
    const mx = (k) => (rr.length ? +(Math.max(...rr.map((x) => x[k])) / 2 ** 30).toFixed(2) : null);
    rows.push({ label: marks[i].label, sec: +(((t1 === Infinity ? Date.now() : t1) - t0) / 1000).toFixed(1),
      gpu_max_gb: g.length ? +(Math.max(...g) / 1024).toFixed(2) : null, gpu_end_gb: g.length ? +(g[g.length - 1] / 1024).toFixed(2) : null,
      ram_max_gb: r.length ? +(Math.max(...r) / 2 ** 30).toFixed(2) : null, ram_gpu_proc_gb: mx(2), ram_renderer_gb: mx(3), ws_gb: mx(4), ws_gpu_proc_gb: mx(5) });
  }
  return rows;
}
async function target() {
  for (let i = 0; i < 60; i++) {
    try {
      if (chrome.exitCode !== null) throw new Error("Chrome이 바로 종료됨 (exit " + chrome.exitCode + ")");
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`, { signal: AbortSignal.timeout(1500) })).json();
      const t = list.find((x) => x.type === "page" && x.url.startsWith(url.split("?")[0]));
      if (t) return t;
    } catch (e) { if (/바로 종료/.test(e.message)) throw e; }
    await sleep(500);
  }
  throw new Error("CDP 대상 페이지를 찾지 못함");
}

try {
  const t = await target();
  const ws = new WebSocket(t.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0; const pending = new Map();
  ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); } };
  const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  await sleep(1500);
  const idle = mon.gpu.length ? mon.gpu[mon.gpu.length - 1][1] : null; // 페이지가 아무것도 올리기 전
  const r = await Promise.race([
    send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }),
    sleep(timeout).then(() => ({ error: "timeout" })),
  ]);
  if (r.error) { // 시간 초과: 페이지가 남긴 진행 상태(window.__diag)를 읽어 본다
    const d = await Promise.race([send("Runtime.evaluate", { expression: "JSON.stringify(window.__diag ?? null)", returnByValue: true }), sleep(5000).then(() => null)]);
    console.log(JSON.stringify(r.error), d ? d.result?.result?.value : "(페이지 응답 없음 — 메인 스레드가 막혀 있음)");
  }
  else if (r.result.exceptionDetails) console.log("EXC", JSON.stringify(r.result.exceptionDetails.exception?.description ?? r.result.exceptionDetails));
  else {
    const v = r.result.result.value;
    const save = opt("--save-out", null); // 결과의 out_b64(fp32 원시)를 파일로
    if (v && v.out_b64) { if (save) fs.writeFileSync(save, Buffer.from(v.out_b64, "base64")); delete v.out_b64; }
    if (v && v.outs) { // 여러 출력: <save>_<이름>.f32
      if (save) for (const [k, b] of Object.entries(v.outs)) fs.writeFileSync(`${save}_${k}.f32`, Buffer.from(b, "base64"));
      delete v.outs;
    }
    if (v && v.marks && monProcs.length) { v.gpu_idle_gb = idle != null ? +(idle / 1024).toFixed(2) : null; v.mon = monSummary(v.marks); delete v.marks; }
    console.log(JSON.stringify(v, null, 1));
  }
  ws.close();
} finally {
  for (const p of monProcs) try { p.kill(); } catch {}
  if (!args.includes("--keep")) {
    const { execSync } = await import("node:child_process");
    try { execSync(`taskkill /PID ${chrome.pid} /T /F`, { stdio: "ignore" }); } catch {}
    await sleep(2500);
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch {}
  }
  process.exit(0);
}
