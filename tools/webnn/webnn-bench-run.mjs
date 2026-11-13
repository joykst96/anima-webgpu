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
    console.log(JSON.stringify(v, null, 1));
  }
  ws.close();
} finally {
  if (!args.includes("--keep")) {
    const { execSync } = await import("node:child_process");
    try { execSync(`taskkill /PID ${chrome.pid} /T /F`, { stdio: "ignore" }); } catch {}
    await sleep(2500);
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch {}
  }
  process.exit(0);
}
