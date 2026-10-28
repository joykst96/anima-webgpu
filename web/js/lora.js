// LoRA 상태 + 변환 워커 클라이언트

// ───────── LoRA (런타임 슬롯 주입) ─────────
const lora = { items: [], lastReport: "", dirty: false }; // items: [{file, fileName, src, weight, id}]
const LORA_DIRTY_MSG = "변경됨 — [LoRA 반영]을 눌러 적용";
// 워커 소스: web/workers/lora-worker.js

// ───────── LoRA 변환 Worker (svd-js, 메인 비차단) ─────────
let _loraWorker = null;
function getLoraWorker() {
  if (_loraWorker) return _loraWorker;
  _loraWorker = new Worker(new URL("../workers/lora-worker.js", import.meta.url), {
    type: "module",
  });
  return _loraWorker;
}
function buildLoraBinAsync(manifest, active, onProg) {
  return new Promise((resolve, reject) => {
    const wk = getLoraWorker();
    const handle = (ev) => {
      const m = ev.data;
      if (m.type === "progress") onProg && onProg(m.value);
      else if (m.type === "done") {
        wk.removeEventListener("message", handle);
        resolve({ bytes: m.bytes, report: m.report });
      } else if (m.type === "error") {
        wk.removeEventListener("message", handle);
        reject(new Error(m.message));
      }
    };
    wk.addEventListener("message", handle);
    const loras = active.map((i) => ({
      fileName: i.fileName,
      weight: i.weight,
      buffer: i.src.buffer,
    }));
    wk.postMessage({ manifest, loras });
  });
}

export { buildLoraBinAsync, lora, LORA_DIRTY_MSG };
