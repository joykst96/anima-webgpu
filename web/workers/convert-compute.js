// 변환 계산 Worker (convert-worker.js의 풀): 텐서 하나를 체크포인트 File에서 직접 읽어 병합·양자화하고 바이트를 돌려준다.
import { computeJob, sourceReader } from "../js/convert-core.js";

let opts = null;

self.onmessage = async ({ data }) => {
  if (data.type === "init") {
    opts = {
      read: sourceReader(data.ckpt),
      loraRead: data.lora ? sourceReader(data.lora) : null,
      variants: data.variants,
      strength: data.strength,
      block: data.block,
    };
    return;
  }
  try {
    const r = await computeJob(data.job, opts);
    const bufs = new Set();
    for (const ws of Object.values(r.writes)) for (const [, , b] of ws) bufs.add(b.buffer);
    for (const w of Object.values(r.inline)) bufs.add(w.buffer);
    self.postMessage({ id: data.id, r }, [...bufs]);
  } catch (e) {
    self.postMessage({ id: data.id, error: e.message });
  }
};
