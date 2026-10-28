// 체크포인트 변환 Worker: safetensors → 지정한 폴더(FileSystemDirectoryHandle)에 변형별 하위 폴더로 페이지용 DiT 그래프·샤드·manifest.
// 폴더는 사용자가 고른 실제 폴더(showDirectoryPicker) 또는 OPFS(다운로드용 임시) — 둘 다 createWritable로 같은 경로를 탄다.
// 계산은 convert-compute.js 풀에 텐서 단위로 나누고, 이 Worker는 파일 쓰기와 템플릿 패치를 맡는다.
// 메시지: {type:"start", dir, folder, ckpt, lora, strength, variants, mapUrl, templateDir}
//   ckpt/lora: File 또는 {url, size, name} (convert-core sourceReader), folder: 하위 폴더 이름 앞부분
//   → {type:"progress", frac, stage} … {type:"done", meta, folders} | {type:"error", message}
import { convertDit, patchGraph, shardManifest, sourceReader } from "../js/convert-core.js";

const post = (m) => self.postMessage(m);

async function sha1Hex(bytes) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-1", bytes));
  return [...d].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

function makePool(n, init) {
  const workers = Array.from({ length: n }, () => {
    const w = new Worker(new URL("./convert-compute.js", import.meta.url), { type: "module" });
    w.postMessage({ type: "init", ...init });
    return w;
  });
  const idle = [...workers],
    waiting = [],
    pending = new Map();
  let seq = 0;
  for (const w of workers)
    w.onmessage = ({ data }) => {
      const p = pending.get(data.id);
      pending.delete(data.id);
      idle.push(w);
      if (waiting.length) waiting.shift()();
      data.error ? p.reject(new Error(data.error)) : p.resolve(data.r);
    };
  return {
    async run(job) {
      while (!idle.length) await new Promise((r) => waiting.push(r));
      const w = idle.pop(),
        id = seq++;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        w.postMessage({ id, job });
      });
    },
    close: () => workers.forEach((w) => w.terminate()),
  };
}

// 위치 지정 쓰기. 같은 스트림에 write를 겹쳐 부르면 잠금 오류라 파일마다 순서대로 잇는다.
// createWritable은 임시 파일에 쓰고 close 때 바꿔치기한다(명세) → 중간에 실패하면 abort로 버린다.
async function openOut(dir, name, size) {
  const w = await (await dir.getFileHandle(name, { create: true })).createWritable();
  await w.truncate(size); // 정렬 패딩 = 0
  let tail = Promise.resolve();
  return {
    write: (off, bytes) => (tail = tail.then(() => w.write({ type: "write", position: off, data: bytes }))),
    close: async () => (await tail, await w.close()),
    abort: () => w.abort().catch(() => {}),
  };
}

async function writeFile(dir, name, bytes) {
  const w = await (await dir.getFileHandle(name, { create: true })).createWritable();
  await w.write(bytes);
  await w.close();
}

async function start(msg) {
  const t0 = performance.now();
  const { dir, folder, ckpt, lora, strength, variants, mapUrl, templateDir } = msg;
  post({ type: "progress", frac: 0, stage: "매핑표·템플릿 그래프 받는 중" });
  const map = await (await fetch(mapUrl)).json();

  // 템플릿: 그래프(인라인 패치 대상)와 LoRA 슬롯 manifest. 매핑표가 만든 그래프와 같은 파일인지 sha1로 확인한다.
  const templates = {},
    loraManis = {};
  for (const [gk, g] of Object.entries(map.graphs)) {
    if (!variants.includes(gk.startsWith("turbo") ? "turbo" : "base")) continue;
    const b = await fetchBytes(templateDir + g.file);
    if ((await sha1Hex(b)) !== g.sha1) throw new Error(`템플릿 그래프가 매핑표와 다름: ${g.file}`);
    templates[gk] = b;
  }
  for (const v of variants) loraManis[v] = await fetchBytes(`${templateDir}${map.stems[v]}_q8a4fnLs.onnx.lora_manifest.json`);

  // 변형마다 하위 폴더 하나 = 생성 페이지 '사용자 설정'에서 한 번에 고를 파일 묶음. 이미 있으면 덮어쓰지 않는다.
  const folders = Object.fromEntries(variants.map((v) => [v, `${folder}_${v}`]));
  for (const name of Object.values(folders)) {
    let exists = true;
    try {
      await dir.getDirectoryHandle(name);
    } catch {
      exists = false;
    }
    if (exists) throw new Error(`'${name}' 폴더가 이미 있음 — 다른 이름으로 변환하거나 지운 뒤 다시`);
  }
  const subs = {};
  for (const v of variants) subs[v] = await dir.getDirectoryHandle(folders[v], { create: true });

  const files = {},
    outs = {};
  let pool = null;
  try {
    for (const v of variants) {
      files[v] = [];
      for (const s of map.shards) files[v].push(await openOut(subs[v], s.name.replace("{stem}", map.stems[v]), s.size));
      outs[v] = { write: (i, off, bytes) => files[v][i].write(off, bytes) };
    }

    const n = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
    pool = makePool(n, { ckpt, lora, variants, strength, block: map.block });
    post({ type: "progress", frac: 0, stage: `변환 중 (Worker ${n}개)` });
    let last = 0;
    const res = await convertDit({
      map,
      read: sourceReader(ckpt),
      lora: lora ? { read: sourceReader(lora), strength } : null,
      outs,
      compute: (job) => pool.run(job),
      concurrency: n,
      onProgress: (f) => {
        const now = performance.now();
        if (now - last > 250 || f === 1) {
          last = now;
          post({ type: "progress", frac: f, stage: "변환 중" });
        }
      },
    });
    post({ type: "progress", frac: 1, stage: "파일 마무리 중" });
    for (const fs of Object.values(files)) for (const f of fs) await f.close();
    for (const k of Object.keys(files)) delete files[k];

    const enc = new TextEncoder();
    for (const [gk, tpl] of Object.entries(templates)) {
      const v = gk.startsWith("turbo") ? "turbo" : "base";
      await writeFile(subs[v], map.graphs[gk].file, patchGraph(tpl, res.patches[gk]));
    }
    const meta = {
      format: map.format,
      mapVersion: map.version,
      source: { name: ckpt.name, size: ckpt.size, lastModified: ckpt.lastModified ?? null, url: ckpt.url ?? null },
      lora: lora ? { name: lora.name, size: lora.size, strength, modules: res.loraModules } : null,
      created: new Date().toISOString(),
      seconds: +((performance.now() - t0) / 1000).toFixed(1),
    };
    for (const v of variants) {
      const stem = map.stems[v];
      const mani = enc.encode(shardManifest(map.shards.map((s) => s.name.replace("{stem}", stem))));
      for (const slot of ["_q8a4fns", "_q8a4fnLs"]) await writeFile(subs[v], `${stem}${slot}.onnx.manifest.json`, mani);
      await writeFile(subs[v], `${stem}_q8a4fnLs.onnx.lora_manifest.json`, loraManis[v]);
      await writeFile(subs[v], "convert-meta.json", enc.encode(JSON.stringify({ ...meta, variant: v }, null, 1)));
    }
    post({ type: "done", meta, folders });
  } catch (e) {
    for (const fs of Object.values(files)) for (const f of fs) await f.abort();
    for (const name of Object.values(folders))
      await dir.removeEntry(name, { recursive: true }).catch(() => {}); // 이 변환이 만든 폴더만 — 반쯤 쓴 결과는 남기지 않는다
    throw e;
  } finally {
    pool?.close();
  }
}

self.onmessage = ({ data }) => {
  if (data.type === "start") start(data).catch((e) => post({ type: "error", message: e.message }));
};
