// 모델 다운로드·OPFS 캐시·세션 생성·단일 상주 로드
import { $, log } from "./dom.js";
import { MODELS, PATHS } from "./config.js";
import { loadTokenizer } from "./tokenizer.js";
import { HAS_WEBNN, disposeRun, disposeTensor, loadOrtWebnn, makeF16Tensor, ort } from "./ort.js";
import { CTX_DIM, CTX_LEN, LAT_C } from "./constants.js";
import { buildLoraBinAsync, lora } from "./lora.js";
import { BUSY_MSG, queueBusy } from "./queue-state.js";
import { PRECISION_LABEL, settings } from "./settings.js";

// 로드 진행 중 여부 — 큐가 끝날 때 btnLoad를 다시 켜도 되는지 판단에 쓴다 (queue.js)
// warmup: 로드 직후 백그라운드 셰이더 워밍업의 Promise. 세션을 쓰거나 내리기 전에 기다린다.
const loadState = { loading: false, warmup: null };

// 사용자설정: 직접 선택한 DiT 파일들 (파일명 → File). loadSession이 URL 대신 사용.
const customFiles = new Map();
// ───────── 모델 로드 ─────────
let sess = {},
  tok = {};

const SHARD_PARALLEL = 3; // 샤드 동시 다운로드 수

// 이번 로드의 캐시 통계 (로드 끝에 요약 로그 — 캐시가 비워졌는지 사용자가 알 수 있게)
const cacheStats = { hit: 0, miss: 0, downloaded: 0 };

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  const total = +res.headers.get("Content-Length") || 0;
  const reader = res.body.getReader();
  // 길이를 알면 버퍼 하나를 미리 잡고 그 자리에 쓴다 — 청크를 모았다가 합치면 샤드(~400MB)마다 메모리가 2배로 든다.
  // 압축 전송 등으로 길이가 어긋나면 청크 모으기로 전환.
  let buf = total ? new Uint8Array(total) : null,
    chunks = buf ? null : [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buf && got + value.length <= buf.length) buf.set(value, got);
    else {
      if (buf) {
        chunks = [buf.subarray(0, got)];
        buf = null;
      }
      chunks.push(value);
    }
    got += value.length;
    if (total) onProgress(Math.min(1, got / total));
  }
  cacheStats.downloaded += got;
  if (buf) return got === buf.length ? buf : buf.slice(0, got);
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

// ───────── OPFS 캐시 ─────────
// 가중치를 디스크(Origin Private File System)에 저장 → 재방문 시 다운로드 0.
// 검증: 서버 Content-Length와 캐시 파일 크기 대조 (불일치 시 재다운로드).
let opfsDir = null;
async function getOpfs() {
  if (opfsDir !== null) return opfsDir;
  try {
    const root = await navigator.storage.getDirectory();
    opfsDir = await root.getDirectoryHandle("anima-cache", {
      create: true,
    });
  } catch {
    opfsDir = false;
  } // 미지원/거부 → 캐시 없이 동작
  return opfsDir;
}
const cacheName = (url) => url.replace(/[^a-zA-Z0-9._-]/g, "_");

// quiet: 캐시 히트 로그를 남기지 않는다(저VRAM 모드가 장마다 다시 읽을 때)
async function cachedFetch(url, onProgress, quiet = false) {
  const dir = await getOpfs();
  if (!dir) return fetchWithProgress(url, onProgress);

  const name = cacheName(url);
  let expected = -1;
  try {
    const head = await fetch(url, { method: "HEAD" });
    if (head.ok) expected = +head.headers.get("Content-Length") || -1;
  } catch {
    /* 오프라인 등 — 캐시가 있으면 그걸 신뢰 */
  }

  try {
    const fh = await dir.getFileHandle(name);
    const file = await fh.getFile();
    if (file.size > 0 && (expected < 0 || file.size === expected)) {
      const buf = new Uint8Array(await file.arrayBuffer());
      onProgress(1);
      cacheStats.hit++;
      if (!quiet) log(`  (캐시 히트: ${(file.size / 2 ** 20).toFixed(0)}MB)`, "ok");
      return buf;
    }
  } catch {
    /* 캐시 없음 */
  }

  cacheStats.miss++;
  const buf = await fetchWithProgress(url, onProgress);
  // createWritable()은 임시 파일에 쓰고 close() 때 한 번에 바꿔치기한다(OPFS 명세) → 중간에 끊겨도
  // 반쯤 쓴 캐시 파일은 남지 않는다. 실패하면 abort로 임시 파일을 버린다.
  let w = null;
  try {
    const fh = await dir.getFileHandle(name, { create: true });
    w = await fh.createWritable();
    await w.write(buf);
    await w.close();
  } catch (e) {
    try {
      await w?.abort();
    } catch {
      /* 이미 닫힘 */
    }
    log(`  (캐시 저장 실패 — 무시하고 진행: ${e.message})`, "dim");
  }
  return buf;
}

// 캐시를 브라우저의 '영구 보관'으로 요청한다. 기본(best-effort) 저장소는 브라우저가 공간이 부족하면
// 통째로 지울 수 있다(실측: 세션 중 3.5GB가 사라져 재다운로드 45s). Chrome은 묻지 않고 사이트 사용 이력 등으로 판단.
let persistChecked = false;
async function ensurePersistentCache() {
  if (persistChecked || !navigator.storage?.persist) return;
  persistChecked = true;
  try {
    let ok = await navigator.storage.persisted();
    if (!ok) ok = await navigator.storage.persist();
    const est = await navigator.storage.estimate();
    log(
      `캐시 저장소: ${ok ? "영구 보관" : "임시 보관 (브라우저가 공간이 부족하면 지울 수 있음)"} · ` +
        `사용 ${(est.usage / 2 ** 30).toFixed(1)}GB / 한도 ${(est.quota / 2 ** 30).toFixed(1)}GB`,
      "dim",
    );
  } catch {
    /* 미지원 — 조용히 진행 */
  }
}

// 지금 배포(config.js)가 참조하지 않는 옛 캐시 파일을 지운다 (파일명 교체 배포 정책 → 옛 가중치가 쌓임).
// 캐시 이름은 URL에서 만들어지므로, 알려진 그래프 경로의 '.onnx 앞부분'으로 시작하면 유지
// (샤드 .N.bin, .data, _f16/_head/_tail 변형, manifest 모두 같은 앞부분).
let pruned = false;
async function pruneOldCache() {
  const dir = await getOpfs();
  if (!dir || pruned) return;
  pruned = true;
  const stems = [...Object.values(PATHS), ...Object.values(MODELS).flatMap((m) => [m.dit, m.ditLora])]
    .filter(Boolean)
    .map((p) => cacheName(p.replace(/\.onnx$/, "")));
  let n = 0,
    bytes = 0;
  for await (const [name, handle] of dir.entries()) {
    if (stems.some((s) => name.startsWith(s))) continue;
    try {
      if (handle.kind === "file") bytes += (await handle.getFile()).size;
      await dir.removeEntry(name, { recursive: true });
      n++;
    } catch {
      /* 다른 탭이 쓰는 중 등 — 다음 기회에 */
    }
  }
  if (n) log(`옛 캐시 정리: ${n}개 파일 (${(bytes / 2 ** 20).toFixed(0)}MB)`, "dim");
}

async function clearCache() {
  const dir = await getOpfs();
  if (!dir) {
    log("OPFS 미지원 환경", "dim");
    return;
  }
  let n = 0;
  for await (const key of dir.keys()) {
    await dir.removeEntry(key);
    n++;
  }
  log(`캐시 ${n}개 파일 삭제됨`, "ok");
}


// 선택된 모델의 DiT를 내렸다 다시 올려 현재 LoRA 목록을 반영한다.
// 반환: "ok" | "busy"(생성 중) | "not-loaded"(선택 모델이 상주하지 않음 — 다음 로드 때 반영) | "failed"
async function reloadDit() {
  // 실행 중인 job이 쓰는 세션을 release()하면 run()이 깨지고 남은 큐도 sess.dit=null로 전부 실패한다
  if (queueBusy()) {
    log(BUSY_MSG, "err");
    return "busy";
  }
  await loadState.warmup; // 워밍업이 쓰는 세션을 release하지 않도록
  const key = $("modelSel").value;
  if (!ditSessions[key]) return "not-loaded"; // 미로드 상태면 다음 로드 때 자동 반영
  await releaseDit(key);
  await loadAll(); // 실패는 loadAll이 로그로 남기고 삼킨다 → 세션 유무로 판정
  return ditSessions[key] ? "ok" : "failed";
}

// DiT를 내릴 때는 공용 세션(TE·어댑터·VAE)도 함께 내린다(다음 loadAll이 OPFS 캐시에서 다시 올림).
// ORT WebGPU EP는 실행 중간 버퍼를 GPU 컨텍스트의 버킷 캐시에 남겨 두는데(생성 1회 후 ~3.6GB), 이 캐시는 세션 하나를
// release()해도 그대로이고 컨텍스트를 쓰는 세션이 전부 해제돼야 사라진다(ORT 1.26 WebGpuContextFactory::ReleaseContext).
// DiT만 갈아 끼우면 상주 VRAM이 8.6 → 9.7GB로 늘고 생성 정점이 16GB 상한에 닿아 VAE가 느려진다.
// 캐시 모드(storageBufferCacheMode)는 ORT JS가 EP를 붙인 뒤에 설정을 넣어 바꿀 수 없다.
async function releaseDit(key) {
  if (sess.dit === ditSessions[key]) sess.dit = null;
  try {
    await ditSessions[key].release();
  } catch (e) {
    console.warn("release 실패:", e);
  }
  delete ditSessions[key];
  for (const k of Object.keys(sess)) {
    if (k === "dit" || !sess[k] || typeof sess[k].release !== "function") continue; // DiT는 ditSessions가 관리
    try {
      await sess[k].release();
    } catch (e) {
      console.warn("release 실패:", e);
    }
    sess[k] = null;
  }
  sharedLoaded = false;
  await new Promise((r) => setTimeout(r, 300)); // GPU 버퍼 반납 정리 시간
}
// 파일명에서 디렉터리 제거 (manifest는 베이스명으로 샤드를 참조)
function baseName(p) {
  return p.split("/").pop();
}

async function readFileBytes(file) {
  return new Uint8Array(await file.arrayBuffer());
}

// 사용자설정 파일 선택을 바꾼다. 상주 중인 사용자 모델은 다음 로드 때 다시 올린다(customVersion).
let customVersion = 0;
function setCustomFiles(fileList) {
  customFiles.clear();
  for (const f of fileList) customFiles.set(f.name, f);
  customVersion++;
  return customStem();
}

// 변환 페이지(web/convert/) 결과처럼 '<stem>_q8a4fns.onnx' 묶음이면 그 stem (예: animav10_dit_turbo32)
function customStem() {
  const stems = [...new Set([...customFiles.keys()].map((n) => /^(.*)_q8a4fns\.onnx$/.exec(n)?.[1]).filter(Boolean))];
  if (stems.length > 1) throw new Error(`모델 하나의 파일만 선택하세요 (${stems.join(", ")})`);
  return stems[0] ?? null;
}

// 사용자설정에서 올릴 그래프. 묶음이 갖춰져 있으면 서버 모델과 같은 규칙(LoRA 장착 → 슬롯 그래프 _q8a4fnLs,
// '빠르게' → _f16, 그 외 → _x)으로 고르고, 아니면 선택한 .onnx 그대로. 샤드·manifest는 원본 그래프(base) 이름 기준.
function customSource(slot) {
  const stem = customStem();
  if (!stem) {
    const onnx = [...customFiles.keys()].find((n) => n.endsWith(".onnx"));
    if (!onnx) throw new Error(".onnx 파일이 선택되지 않았습니다");
    return { graph: onnx, base: onnx, precision: "standard" };
  }
  const base = `${stem}${slot && customFiles.has(`${stem}_q8a4fnLs.onnx`) ? "_q8a4fnLs" : "_q8a4fns"}.onnx`;
  const fast = settings.precision === "fast",
    g = base.replace(/\.onnx$/, fast ? "_f16.onnx" : "_x.onnx");
  if (customFiles.has(g)) return { graph: g, base, precision: fast ? "fast" : "standard" };
  return { graph: base, base, precision: "standard" };
}

// ── WebNN DiT (고급 옵션 'WebNN', 실험) ──
// <dit>_wnn1024.onnx = 정적 1024² fp16 그래프(export/webnn/build_webnn.bat: export_dit.py --size 1024 → webnn_fix.py → shard_onnx_data.py).
// 가중치가 fp16이라 샤드도 따로다(<그래프>.manifest.json). 이름이 <dit> 앞부분으로 시작해 pruneOldCache가 지우지 않는다.
// LoRA를 붙였으면 호출부가 슬롯 그래프 주소(<ditLora> = …_q8a4fnLs.onnx)를 주므로 여기서도 …_q8a4fnLs_wnn1024.onnx(슬롯 그래프, 샤드 공유)를 쓴다.
// 슬롯 레이아웃(lora_manifest)은 표준 슬롯 그래프와 같아 변환 워커·lora_scale 입력을 그대로 쓴다(export/add_lora_slots.py).
// 쓸 수 없으면(브라우저 미지원·그래프 없음·생성 실패·시간 초과) null → 호출부가 표준 경로로 로드한다.
// ORT Web 1.26은 브라우저가 그래프를 거부하면 create가 끝나지 않으므로 시간 제한을 둔다(ort.js loadOrtWebnn 주석).
const WEBNN_CREATE_TIMEOUT_MS = 180000;

// LoRA 슬롯 그래프의 외부 데이터(lora.bin): 장착된 LoRA를 manifest 레이아웃으로 변환한다(Worker). 없으면 0으로 채움(다운로드 0).
// 표준(WebGPU)·WebNN 슬롯 그래프가 같은 레이아웃을 쓴다.
async function buildLoraExternal(lman) {
  const total = lman.entries.reduce((m, e) => Math.max(m, e.B.offset + lman.rank * e.N * 4), 0);
  let bytes;
  const active = lora.items.filter((i) => i.src);
  if (active.length) {
    log(`LoRA 병합 (Worker, ${active.length}개)...`, "dim");
    const c = await buildLoraBinAsync(lman, active, (f) => {
      $("loadBar").style.width = f * 100 + "%";
      $("loadStat").textContent = `LoRA 병합 ${(f * 100) | 0}%`;
    });
    bytes = c.bytes;
    lora.lastReport =
      `${active.length}개 LoRA — ${c.report.applied}/${c.report.total} 슬롯 적용` +
      (c.report.notes ? ` (${c.report.notes})` : "");
    log(lora.lastReport, c.report.applied ? "ok" : "err");
  } else {
    bytes = new Uint8Array(total); // 제로 = LoRA 없음 (다운로드 0)
    return { path: lman.bin, data: bytes, get: async () => new Uint8Array(total) };
  }
  return { path: lman.bin, data: bytes };
}

// WebNN 로더가 파일을 얻는 곳. 서버 모델은 주소(URL)로 받아 OPFS에 캐시하고, 사용자 설정 모델은 고른 폴더의 파일을 이름으로 바로 읽는다
// (변환 페이지가 'WebNN용 파일도 만들기'로 같은 폴더에 써 둔 것 — 파일 이름 규칙은 서버 모델과 같다).
const serverIO = {
  where: "서버에",
  has: async (u) => {
    const r = await fetch(u, { method: "HEAD" }).catch(() => null);
    return !!(r && r.ok);
  },
  json: async (u) => {
    const r = await fetch(u).catch(() => null);
    return r && r.ok ? await r.json() : null;
  },
  bytes: (u, onProgress = () => {}) => cachedFetch(u, onProgress),
};
const customIO = {
  where: "선택한 폴더에",
  has: async (n) => customFiles.has(baseName(n)),
  json: async (n) => (customFiles.has(baseName(n)) ? JSON.parse(new TextDecoder().decode(await readFileBytes(customFiles.get(baseName(n))))) : null),
  bytes: async (n, onProgress = () => {}) => {
    const b = await readFileBytes(customFiles.get(baseName(n)));
    onProgress(1);
    return b;
  },
};

// Spectrum 머리 세션. DiT·VAE와 또 다른 ORT 인스턴스("head")에 올린다. releaseDit가 sess의 다른 세션과 함께 내린다.
//   sess.ditHead  : 특징(fp32, CPU에서 예측해 올림), timesteps → v_pred
//   sess.ditHeadK : feat0..K-1(fp16, 본체가 낸 GPU 텐서 그대로), coef[K], timesteps → v_pred — 가중합까지 GPU에서(<머리>k.onnx).
//                   본체와 같은 MLContext(ctx)여야 텐서를 넘길 수 있다. 없거나 못 올리면 ditHead(CPU 경로)만 쓴다.
// 돌려주는 값: false(Spectrum 불가) | { gpu: 가중합 머리를 쓸 수 있는지 }
async function loadSpectrumHead(headUrl, ctx, io = serverIO) {
  const create = async (ortW, graph, data, ep) => {
    let timer;
    return await Promise.race([
      ortW.InferenceSession.create(graph, { executionProviders: [ep], externalData: [{ path: baseName(headUrl) + ".data", data }] }),
      new Promise((_, rej) => {
        timer = setTimeout(() => rej(new Error("세션 생성 시간 초과")), WEBNN_VAE_CREATE_TIMEOUT_MS);
      }),
    ]).finally(() => clearTimeout(timer));
  };
  const ep = { name: "webnn", deviceType: "gpu", powerPreference: "high-performance" };
  try {
    const [graph, data] = await Promise.all([io.bytes(headUrl), io.bytes(headUrl + ".data")]);
    const ortW = await loadOrtWebnn("head");
    sess.ditHead = await create(ortW, graph, data, ep);
    sess.ditHeadK = null;
    const kUrl = headUrl.replace(/\.onnx$/, "k.onnx");
    if (ctx && (await io.has(kUrl))) {
      try {
        const hk = await create(ortW, await io.bytes(kUrl), data, { ...ep, context: ctx });
        hk.slots = hk.inputNames.filter((n) => /^feat\d+$/.test(n)).length;
        hk.wrap = (t) => ortW.Tensor.fromMLTensor(t.mlTensor, { dataType: t.type, dims: t.dims }); // 본체 인스턴스의 GPU 텐서를 이 인스턴스의 입력으로
        sess.ditHeadK = hk;
      } catch (e) {
        console.error(e);
        log(`Spectrum 가중합 머리를 올리지 못함 — ${e.message}. 특징을 CPU에서 합산`, "dim");
      }
    }
    return { gpu: !!sess.ditHeadK };
  } catch (e) {
    console.error(e);
    log(`Spectrum 머리 그래프를 올리지 못함 — ${e.message}. Spectrum 없이 진행`, "err");
    sess.ditHead = sess.ditHeadK = null;
    return false;
  }
}

async function loadWebnnDit(name, url, onProgress, io = serverIO) {
  const fallback = (why) => {
    log(`WebNN 사용 불가 — ${why}. ${PRECISION_LABEL.standard}으로 로드`, "err");
    return null;
  };
  if (!HAS_WEBNN) return fallback("이 브라우저에 WebNN(navigator.ml)이 없음 (Chrome은 chrome://flags의 WebNN 플래그 필요)");
  const isSlot = url.includes("_q8a4fnLs"); // LoRA 장착 → 슬롯 그래프
  // <그래프>_wnn1024n.onnx = 같은 그래프 + NegPip 입력(negpip_mask, export/add_negpip.py --graph-only — 샤드·머리 공유, manifest는 자기 이름으로).
  // 서버가 .onnx를 immutable로 캐시시키므로 새 이름이다. 없으면(이전 빌드·이전 변환 묶음) NegPip 없는 그래프로 — 마스크가 1이면 두 그래프의 결과는 같다.
  // int8 가중치(settings.webnnWeights): <그래프>_wnn1024q{,n}.onnx — 표준(WebGPU) 그래프의 샤드(<dit>.onnx.N.bin)와 작은 extra 파일을 가리키는
  // 같은 계산 그래프(export/webnn/webnn_stdq8.py). 표준으로 한 번이라도 받았으면 OPFS 캐시의 샤드를 그대로 써서 다운로드가 없다.
  // 없으면(int8 그래프를 아직 안 만든 모델·사용자 설정 모델) fp16 그래프로.
  const baseG = url.replace(/\.onnx$/, "_wnn1024.onnx"),
    negG = url.replace(/\.onnx$/, "_wnn1024n.onnx");
  const wantInt8 = settings.webnnWeights === "int8";
  const qG = url.replace(/\.onnx$/, "_wnn1024q.onnx"),
    qnG = url.replace(/\.onnx$/, "_wnn1024qn.onnx");
  let gUrl = null;
  if (wantInt8) {
    gUrl = (await io.has(qnG)) ? qnG : (await io.has(qG)) ? qG : null;
    if (!gUrl) log(`${name}: 이 모델의 WebNN int8 그래프가 ${io.where} 없음 — fp16 가중치로`, "dim");
  }
  const weights = gUrl ? "int8" : "fp16";
  if (!gUrl) gUrl = (await io.has(negG)) ? negG : baseG;
  if (gUrl === baseG && !(await io.has(gUrl)))
    return fallback(isSlot ? `이 모델의 WebNN LoRA 슬롯 그래프가 ${io.where} 없음` : `이 모델의 WebNN 그래프가 ${io.where} 없음`);
  const mani = await io.json(gUrl + ".manifest.json");
  if (!mani) return fallback("WebNN 그래프의 샤드 목록(manifest)이 없음");
  const lman = isSlot ? await io.json(gUrl + ".lora_manifest.json") : null;
  if (isSlot && !lman) return fallback("WebNN LoRA 슬롯 그래프의 슬롯 목록(lora_manifest)이 없음");
  // Spectrum(spectrum.js): 본체 = 같은 그래프 + 특징 출력(<그래프>_sp.onnx, 샤드·manifest 공유), 머리 = 특징·timesteps → v_pred
  // (<비슬롯 그래프>_head.onnx + .data, LoRA와 무관). 둘 다 있으면 본체를 _sp로 올린다 — 특징 출력을 안 읽으면 결과·시간은 같다.
  const spUrl = gUrl.replace(/\.onnx$/, "_sp.onnx"),
    headUrl = baseG.replace("_q8a4fnLs", "_q8a4fns").replace(/\.onnx$/, "_head.onnx");
  const useSp = (await io.has(spUrl)) && (await io.has(headUrl)) && (await io.has(headUrl + ".data"));
  try {
    log(`${name} WebNN 그래프 다운로드...`, "dim");
    const graph = await io.bytes(useSp ? spUrl : gUrl, (f) => onProgress(f * 0.05));
    const dir = gUrl.slice(0, gUrl.lastIndexOf("/") + 1);
    const n = mani.shards.length,
      ext = new Array(n),
      frac = new Array(n).fill(0);
    let next = 0;
    const worker = async () => {
      for (let i; (i = next++) < n; ) {
        log(`${name} 샤드 ${i + 1}/${n} (${mani.shards[i]})...`, "dim");
        ext[i] = {
          path: mani.shards[i],
          data: await io.bytes(dir + mani.shards[i], (f) => {
            frac[i] = f;
            onProgress(0.05 + (frac.reduce((a, b) => a + b, 0) / n) * 0.85);
          }),
        };
      }
    };
    await Promise.all(Array.from({ length: Math.min(SHARD_PARALLEL, n) }, worker));
    if (lman) ext.push(await buildLoraExternal(lman));
    const ortW = await loadOrtWebnn();
    log(`${name} 컴파일 (WebNN)...`, "dim");
    // Spectrum용 본체는 MLContext를 직접 만들어 넘기고 특징(fp16)을 GPU 텐서로 내게 한다 — 같은 컨텍스트의 가중합 머리에 그대로 넣기 위해.
    // (안 읽는 출력은 비용이 없다. 컨텍스트를 못 만들면 특징을 CPU로 읽는 경로만 쓴다.)
    const ep = { name: "webnn", deviceType: "gpu", powerPreference: "high-performance" };
    const ctx = useSp ? await navigator.ml.createContext({ deviceType: ep.deviceType, powerPreference: ep.powerPreference }).catch(() => null) : null;
    let timer;
    const s = await Promise.race([
      ortW.InferenceSession.create(graph, {
        executionProviders: [ctx ? { ...ep, context: ctx } : ep],
        externalData: ext,
        ...(ctx ? { preferredOutputLocation: { spectrum_feat16: "ml-tensor" } } : {}),
      }),
      new Promise((_, rej) => {
        timer = setTimeout(() => rej(new Error("세션 생성 시간 초과 (브라우저가 그래프를 거부했을 수 있음)")), WEBNN_CREATE_TIMEOUT_MS);
      }),
    ]).finally(() => clearTimeout(timer));
    const shape = s.inputMetadata?.find((m) => m.name === "latent")?.shape;
    s.staticLat = { h: Number(shape?.[3] ?? 128), w: Number(shape?.[4] ?? 128) }; // 정적 그래프 — 이 latent 크기만 받는다
    s.loraManifest = lman;
    s.precision = "webnn";
    s.weights = weights; // "fp16" | "int8" — 실제로 올린 가중치 (메타데이터·고급 옵션 표시)
    s.spectrum = useSp ? await loadSpectrumHead(headUrl, ctx, io) : false; // 머리를 못 올리면 Spectrum만 못 쓴다
    onProgress(1);
    log(`${name} OK (WebNN · ${weights} · ${s.staticLat.w * 8}×${s.staticLat.h * 8} 전용${s.spectrum ? " · Spectrum 가능" : ""})`, "ok");
    return s;
  } catch (e) {
    console.error(e);
    return fallback(e.message);
  }
}

// WebGPU 세션을 만들고, 같은 입력으로 다시 만들 재료(recipe)를 붙여 둔다(저VRAM 모드 — 아래 lowVramBeforeVae).
// ext: [{ path, data, get }] — get()은 같은 바이트를 다시 읽는다(OPFS 캐시·고른 파일). get이 없는 것(병합한 LoRA)은
// 저VRAM 모드가 켜져 있을 때만 바이트를 붙잡아 둔다. 못 붙잡으면 재료가 불완전해 저VRAM 모드를 건너뛴다.
async function createSession(graph, opts, ext = []) {
  const s = await ort.InferenceSession.create(graph, ext.length ? { ...opts, externalData: ext.map(({ path, data }) => ({ path, data })) } : opts);
  s.recipe = { graph, opts, ext: ext.map(({ path, data, get }) => ({ path, get: get ?? (settings.lowVram ? () => data : null) })) };
  return s;
}
const recipeComplete = (s) => !!(s && s.recipe && s.recipe.ext.every((e) => e.get));
async function recreateSession(r) {
  const ext = [];
  for (const e of r.ext) ext.push({ path: e.path, data: await e.get(), get: e.get });
  const s = await ort.InferenceSession.create(r.graph, ext.length ? { ...r.opts, externalData: ext.map(({ path, data }) => ({ path, data })) } : r.opts);
  s.recipe = r;
  return s;
}

async function loadSession(name, url, onProgress) {
  const opts = { executionProviders: ["webgpu"] };
  const ext = []; // 외부 가중치 [{ path, data, get }] — createSession이 opts.externalData로 넘긴다

  // 0) 사용자설정: 메모리 파일맵에서 직접 조립 (네트워크/OPFS 미사용). url = { custom: true, slot: LoRA 장착 여부 }
  if (url?.custom && name.startsWith("dit:")) {
    // WebNN(실험): 변환 묶음에 WebNN용 파일이 있으면 그 세션을 쓴다. 없거나 안 되면 아래 표준 경로.
    if (settings.precision === "webnn") {
      const stem = customStem();
      const s = stem ? await loadWebnnDit(name, `${stem}${url.slot ? "_q8a4fnLs" : "_q8a4fns"}.onnx`, onProgress, customIO) : null;
      if (s) {
        s.customVersion = customVersion;
        return s;
      }
    }
    const files = customFiles;
    const src = customSource(url.slot);
    const onnxFile = src.graph;
    log(`${name} (사용자 파일: ${onnxFile})...`, "dim");
    const graph = await readFileBytes(files.get(onnxFile));
    const maniName = src.base + ".manifest.json";
    if (files.has(maniName)) {
      const mani = JSON.parse(
        new TextDecoder().decode(
          await readFileBytes(files.get(maniName)),
        ),
      );
      for (const sName of mani.shards) {
        const fh =
          files.get(sName) || files.get(baseName(sName));
        if (!fh)
          throw new Error(
            `샤드 누락: ${sName} (선택 목록에 포함되었는지 확인)`,
          );
        ext.push({
          path: baseName(sName),
          data: await readFileBytes(fh),
          get: () => readFileBytes(fh),
        });
      }
    } else if (files.has(onnxFile + ".data")) {
      const fh = files.get(onnxFile + ".data");
      ext.push({ path: onnxFile + ".data", data: await readFileBytes(fh), get: () => readFileBytes(fh) });
    }
    // 사용자 모델의 LoRA manifest (선택)
    const lmanName = src.base + ".lora_manifest.json";
    let lman = files.has(lmanName)
      ? JSON.parse(
          new TextDecoder().decode(
            await readFileBytes(files.get(lmanName)),
          ),
        )
      : null;
    if (lman) {
      const total = lman.entries.reduce(
        (m, e) => Math.max(m, e.B.offset + lman.rank * e.N * 4),
        0,
      );
      const active = lora.items.filter((i) => i.src);
      let bytes;
      if (active.length) {
        log(`LoRA 병합 (Worker, ${active.length}개)...`, "dim");
        const c = await buildLoraBinAsync(lman, active, (f) => {
          $("loadBar").style.width = f * 100 + "%";
          $("loadStat").textContent = `LoRA 병합 ${(f * 100) | 0}%`;
        });
        bytes = c.bytes;
        lora.lastReport = `${active.length}개 LoRA — ${c.report.applied}/${c.report.total} 슬롯`;
        ext.push({ path: lman.bin, data: bytes });
      } else ext.push({ path: lman.bin, data: new Uint8Array(total), get: async () => new Uint8Array(total) });
    }
    log(`${name} 컴파일...`, "dim");
    const s = await createSession(graph, opts, ext);
    s.loraManifest = lman;
    s.precision = src.precision;
    s.customVersion = customVersion;
    if (settings.precision !== src.precision)
      log(`선택한 파일에 '${PRECISION_LABEL[settings.precision]}' 그래프가 없음 — ${PRECISION_LABEL[src.precision]}으로 로드`, "dim");
    onProgress(1);
    log(`${name} OK`, "ok");
    return s;
  }

  // WebNN(실험): 되면 그 세션을 쓰고, 안 되면 아래 표준 경로를 그대로 탄다
  if (name.startsWith("dit:") && settings.precision === "webnn") {
    const s = await loadWebnnDit(name, url, onProgress);
    if (s) return s;
  }

  log(`${name} 그래프 다운로드...`, "dim");
  // 같은 가중치(샤드·LoRA manifest 공유)를 쓰는 후처리 그래프 (export/make_fast_graphs.py):
  //   <dit>_f16.onnx : '빠르게' 모드 (블록 내부 fp16 + 패딩 압축 입력)
  //   <dit>_x.onnx   : 표준 + cross-attention 패딩 압축 입력 (압축을 끄면 원본과 같은 결과)
  // 서버가 .onnx를 1년 immutable로 캐시시키므로 바뀐 그래프는 반드시 새 이름이다. 없으면 다음 후보 → 원본.
  let graphUrl = url,
    precision = "standard";
  if (name.startsWith("dit:")) {
    const head = async (u) => {
      const r = await fetch(u, { method: "HEAD" }).catch(() => null);
      return !!(r && r.ok);
    };
    const fastUrl = url.replace(/\.onnx$/, "_f16.onnx"),
      xUrl = url.replace(/\.onnx$/, "_x.onnx");
    if (settings.precision === "fast" && (await head(fastUrl))) {
      graphUrl = fastUrl;
      precision = "fast";
    } else {
      if (settings.precision === "fast")
        log(`'${PRECISION_LABEL.fast}' 그래프가 서버에 없음 — ${PRECISION_LABEL.standard}으로 로드`, "err");
      if (await head(xUrl)) graphUrl = xUrl;
    }
  }
  const graph = await cachedFetch(graphUrl, (f) => onProgress(f * 0.05));

  // 샤드 manifest는 DiT와 te(text_encoder) 산출물에 존재.
  // adapter/vae는 단일 .data라 manifest 요청을 보내지 않아 404를 피한다.
  const isSharded = name.startsWith("dit") || name === "te";
  // LoRA 슬롯 manifest는 슬롯 모델(_q8a4fnLs)에만 존재. 비슬롯엔 요청 안 함.
  const isSlot = url.includes("_q8a4fnLs");

  // 1) 샤드 manifest 우선 (shard_onnx_data.py 산출물 — ArrayBuffer 2GB 상한 우회)
  const maniRes = isSharded
    ? await fetch(url + ".manifest.json").catch(() => null)
    : null;
  if (maniRes && maniRes.ok) {
    const mani = await maniRes.json();
    const dir = url.slice(0, url.lastIndexOf("/") + 1);
    // 샤드는 SHARD_PARALLEL개씩 동시에 받는다(첫 방문 다운로드 단축). 진행률은 샤드별 진척의 평균.
    const n = mani.shards.length,
      frac = new Array(n).fill(0);
    const report = () => onProgress(0.05 + (frac.reduce((a, b) => a + b, 0) / n) * 0.85);
    let next = 0;
    const worker = async () => {
      for (let i; (i = next++) < n; ) {
        const sName = mani.shards[i];
        log(`${name} 샤드 ${i + 1}/${n} (${sName})...`, "dim");
        const data = await cachedFetch(dir + sName, (f) => {
          frac[i] = f;
          report();
        });
        frac[i] = 1;
        report();
        ext[i] = { path: sName, data, get: () => cachedFetch(dir + sName, () => {}, true) };
      }
    };
    await Promise.all(Array.from({ length: Math.min(SHARD_PARALLEL, n) }, worker));
  } else {
    // 2) 단일 .data 폴백 (dynamo exporter 기본 산출물 — te/adapter/vae)
    const dataUrl = url + ".data";
    const probe = await fetch(dataUrl, { method: "HEAD" }).catch(
      () => null,
    );
    if (probe && probe.ok) {
      log(
        `${name} 가중치 다운로드 (${dataUrl.split("/").pop()})...`,
        "dim",
      );
      const data = await cachedFetch(dataUrl, (f) =>
        onProgress(0.05 + f * 0.85),
      );
      ext.push({ path: dataUrl.split("/").pop(), data, get: () => cachedFetch(dataUrl, () => {}, true) });
    }
  }
  // LoRA 슬롯 manifest는 슬롯 모델에만 있으므로 그때만 요청 (비슬롯 404 회피)
  let lman = null;
  const lres = isSlot
    ? await fetch(url + ".lora_manifest.json").catch(() => null)
    : null;
  if (lres && lres.ok) {
    lman = await lres.json();
    ext.push(await buildLoraExternal(lman));
  }
  log(`${name} 컴파일 (셰이더 빌드)...`, "dim");
  const s = await createSession(graph, opts, ext);
  s.loraManifest = lman;
  s.precision = precision;
  onProgress(1);
  log(`${name} OK${name.startsWith("dit:") ? ` (${PRECISION_LABEL[precision]})` : ""}`, "ok");
  return s;
}

// ── 셰이더 워밍업 ──
// 세션을 새로 만든 직후의 첫 생성은 셰이더 파이프라인 준비로 느리다(1024² 17.1s vs 이후 12.4s).
// 이 비용은 해상도와 무관하게 세션당 한 번이라(워밍업 안 한 832×1216도 패널티 없음 확인),
// 작은 해상도(256²)로 TE → 어댑터 → DiT 1회 → VAE 1회를 미리 돌려 둔다 (첫 생성 17.1 → 13.3s, 워밍업 5.7s).
// 로드 완료 후 백그라운드로 돌고, 생성·재로드는 끝날 때까지 기다린다. 입력 값은 결과와 무관(커널 준비용).
const WARMUP_LAT = 32; // latent 32×32 = 256px
async function warmup(label) {
  const t0 = performance.now();
  const stat = $("loadStat");
  stat.textContent = `${label} 준비됨 (셰이더 준비 중...)`;
  const tensors = [],
    outs = [];
  const t = (type, data, dims) => {
    const x = new ort.Tensor(type, data, dims);
    tensors.push(x);
    return x;
  };
  try {
    const te = await sess.te.run({
      input_ids: t("int64", BigInt64Array.of(1n, 2n, 3n), [1, 3]),
      attention_mask: t("int64", BigInt64Array.of(1n, 1n, 1n), [1, 3]),
    });
    outs.push(te);
    outs.push(
      await sess.adapter.run({
        qwen_hidden_states: te[Object.keys(te)[0]],
        t5_input_ids: t("int64", BigInt64Array.of(1n), [1, 1]),
      }),
    );
    const lat = makeF16Tensor(new Float32Array(LAT_C * WARMUP_LAT * WARMUP_LAT), [1, LAT_C, 1, WARMUP_LAT, WARMUP_LAT]);
    tensors.push(lat);
    const names = sess.dit.inputNames;
    // 정적 그래프(WebNN)는 자기 해상도만 받는다 — DiT 워밍업만 그 크기로 (VAE는 아래에서 종전대로 256²)
    const sl = sess.dit.staticLat;
    const ditLat = sl ? makeF16Tensor(new Float32Array(LAT_C * sl.h * sl.w), [1, LAT_C, 1, sl.h, sl.w]) : lat;
    if (sl) tensors.push(ditLat);
    const feed = {
      latent: ditLat,
      timesteps: t("float32", Float32Array.of(1), [1]),
      context: makeF16Tensor(new Float32Array(CTX_LEN * CTX_DIM), [1, CTX_LEN, CTX_DIM]),
    };
    tensors.push(feed.context);
    if (names.includes("negpip_mask")) {
      feed.negpip_mask = makeF16Tensor(new Float32Array(CTX_LEN).fill(1), [1, CTX_LEN, 1]);
      tensors.push(feed.negpip_mask);
    }
    if (names.includes("lora_scale")) feed.lora_scale = t("float32", Float32Array.of(1), [1]);
    if (names.includes("xattn_index")) {
      feed.xattn_index = t("int64", BigInt64Array.from({ length: CTX_LEN }, (_, i) => BigInt(i)), [CTX_LEN]);
      feed.xattn_bias_row = t("float32", new Float32Array(CTX_LEN), [1, 1, 1, CTX_LEN]);
    }
    outs.push(await sess.dit.run(feed));
    if (sess.vaeWebnn) {
      const wl = makeF16Tensor(new Float32Array(LAT_C * sess.vaeWebnn.lat.h * sess.vaeWebnn.lat.w), [1, LAT_C, 1, sess.vaeWebnn.lat.h, sess.vaeWebnn.lat.w]);
      tensors.push(wl);
      outs.push(await sess.vaeWebnn.run({ latent: wl }));
    }
    if (sess.vaeHead) {
      const ho = await sess.vaeHead.run({ latent: lat });
      outs.push(ho);
      const midName = sess.vaeTail.inputNames.find((n) => n !== "latent");
      outs.push(await sess.vaeTail.run({ [midName]: ho[Object.keys(ho)[0]], latent: lat }));
    } else outs.push(await sess.vae.run({ latent: lat }));
    log(`셰이더 워밍업 끝 (${((performance.now() - t0) / 1000).toFixed(1)}s)`, "dim");
  } catch (e) {
    // 워밍업 실패는 생성에 영향 없음 (첫 생성이 느릴 뿐)
    log(`셰이더 워밍업 건너뜀: ${e.message}`, "dim");
  }
  outs.forEach(disposeRun);
  tensors.forEach(disposeTensor);
  if (stat.textContent.endsWith("(셰이더 준비 중...)")) stat.textContent = `${label} 준비됨`;
}

// ── WebNN VAE (WebNN DiT와 함께 쓴다, 실험) ──
// <vae>_wnn16.onnx = 같은 동적 그래프를 fp16 계산으로 바꾼 것(export/webnn/vae_fp16.py all latent — build_webnn.bat). 가중치는 원본 .data(fp32)를 그대로 쓰고
// ORT가 세션 생성 때 fp16으로 접는다 → 추가 다운로드는 그래프 0.8MB뿐. WebNN은 정적 shape만 받아 freeDimensionOverrides로 크기를 고정한다.
// 1024² 실측(2026-11-13, RTX 5070 Ti, Chrome 157 플래그): WebGPU fp32 742ms·일시 VRAM +7~8GB → WebNN fp16 111ms. fp32 대비 PSNR 67.7dB.
// 실패하면 null → vae.js가 기존 WebGPU 경로로 디코드한다. DiT와 다른 ORT 인스턴스("vae")에 올린다(ort.js loadOrtWebnn 주석).
const WEBNN_VAE_CREATE_TIMEOUT_MS = 60000;
async function loadWebnnVae(url, lat) {
  const gUrl = url.replace(/\.onnx$/, "_wnn16.onnx");
  try {
    const r = await fetch(gUrl, { method: "HEAD" }).catch(() => null);
    if (!(r && r.ok)) {
      log("vae WebNN 그래프가 서버에 없음 — VAE는 기존 경로(WebGPU)", "dim");
      return null;
    }
    const [graph, data] = await Promise.all([cachedFetch(gUrl, () => {}), cachedFetch(url + ".data", () => {})]);
    const ortW = await loadOrtWebnn("vae");
    let timer;
    const s = await Promise.race([
      ortW.InferenceSession.create(graph, {
        executionProviders: [{ name: "webnn", deviceType: "gpu", powerPreference: "high-performance" }],
        externalData: [{ path: baseName(url) + ".data", data }],
        freeDimensionOverrides: { lat_h: lat.h, lat_w: lat.w },
      }),
      new Promise((_, rej) => {
        timer = setTimeout(() => rej(new Error("세션 생성 시간 초과")), WEBNN_VAE_CREATE_TIMEOUT_MS);
      }),
    ]).finally(() => clearTimeout(timer));
    s.lat = { ...lat };
    log(`vae OK (WebNN · fp16 · ${lat.w * 8}×${lat.h * 8} 전용)`, "ok");
    return s;
  } catch (e) {
    console.error(e);
    log(`vae WebNN 사용 불가 — ${e.message}. VAE는 기존 경로(WebGPU)`, "err");
    return null;
  }
}

// VAE 분할 그래프(<vae>_head.onnx / <vae>_tail.onnx, export/split_vae.py)가 서버에 있으면 둘을 올린다.
// 가중치는 원본 .data를 함께 쓴다. 없으면 null → 기존 단일 VAE (web/js/vae.js가 경로를 고른다).
async function loadVaeSplit(url, onProgress) {
  const stem = url.replace(/\.onnx$/, "");
  const urls = [`${stem}_head.onnx`, `${stem}_tail.onnx`];
  const probes = await Promise.all(urls.map((u) => fetch(u, { method: "HEAD" }).catch(() => null)));
  if (!probes.every((r) => r && r.ok)) return null;
  log("vae 분할 그래프(head/tail) 사용 — 고해상도는 띠 단위 디코드", "dim");
  const data = await cachedFetch(url + ".data", (f) => onProgress(f * 0.9));
  const opts = { executionProviders: ["webgpu"] };
  const ext = [{ path: baseName(url) + ".data", data, get: () => cachedFetch(url + ".data", () => {}, true) }];
  const out = [];
  for (const u of urls) out.push(await createSession(await cachedFetch(u, () => {}), opts, ext));
  const split = { head: out[0], tail: out[1] };
  // head 어텐션 청크 분할 조각(_head_a/_attn/_head_b)이 있으면 1536² 이상에서 쓴다 (vae.js runHead)
  const aUrls = ["_head_a", "_attn", "_head_b"].map((s) => `${stem}${s}.onnx`);
  const aProbes = await Promise.all(aUrls.map((u) => fetch(u, { method: "HEAD" }).catch(() => null)));
  if (aProbes.every((r) => r && r.ok)) {
    const [ga, gattn, gb] = await Promise.all(aUrls.map((u) => cachedFetch(u, () => {})));
    split.headA = await createSession(ga, { ...opts, preferredOutputLocation: "gpu-buffer" }, ext);
    split.attn = await createSession(gattn, opts);
    split.headB = await createSession(gb, opts, ext);
  }
  onProgress(1);
  log("vae OK", "ok");
  return split;
}

let sharedLoaded = false;
const ditSessions = {};
// 상주 DiT를 고급 옵션의 지금 설정으로 다시 올려야 하는가: 정밀도 모드, 또는 WebNN이면 가중치(fp16/int8)가 만들 때와 다르면
// (requested·requestedWeights = 요청했던 값. int8 그래프가 없어 fp16으로 폴백한 경우에도 반복 재로드하지 않도록 실제 값이 아니라 요청값과 비교)
function ditNeedsReload(dit) {
  return dit.requested !== settings.precision || (settings.precision === "webnn" && dit.requestedWeights !== settings.webnnWeights);
}
async function loadAll() {
  if (queueBusy()) {
    log(BUSY_MSG, "err");
    return;
  }
  await loadState.warmup; // 워밍업이 쓰는 세션을 release하지 않도록
  loadState.loading = true;
  Object.assign(cacheStats, { hit: 0, miss: 0, downloaded: 0 });
  await ensurePersistentCache();
  $("btnLoad").disabled = true;
  const t0 = performance.now();
  try {
    if (!tok.qwen) {
      log("토크나이저 로드 중...", "dim");
      tok.qwen = await loadTokenizer("qwen3");
      tok.t5 = await loadTokenizer("t5");
      log("토크나이저 OK", "ok");
    }

    const key = $("modelSel").value;
    if (
      key === "custom" &&
      ![...customFiles.keys()].some(
        (n) => n.endsWith(".onnx") && !n.endsWith(".data"),
      )
    ) {
      log("사용자 설정: 모델 폴더를 고르고 목록에서 모델을 선택하세요", "err");
      loadState.loading = false;
      $("btnLoad").disabled = false;
      return;
    }

    // 단일 상주 정책: 다른 DiT가 VRAM에 있으면 새 모델 로드 전에 내린다.
    // (가중치는 OPFS에 남아 있어 재전환 시 다운로드 없이 컴파일만 다시 함)
    for (const k of Object.keys(ditSessions)) {
      if (k !== key) {
        log(`${MODELS[k].label} VRAM 해제...`, "dim");
        await releaseDit(k);
      }
    }
    // 고급 옵션의 정밀도 모드가 상주 DiT를 만들 때와 다르면 다시 올린다
    // (requested = 요청한 모드. 서버에 '빠르게' 그래프가 없어 표준으로 폴백한 경우에도 반복 재로드하지 않도록)
    if (ditSessions[key] && ditNeedsReload(ditSessions[key])) {
      log(`정밀도 변경 (${PRECISION_LABEL[settings.precision]}${settings.precision === "webnn" ? " " + settings.webnnWeights : ""}) — DiT 다시 로드`, "dim");
      await releaseDit(key);
    }
    // 사용자설정 파일 선택이 바뀌었으면 다시 올린다
    if (key === "custom" && ditSessions.custom && ditSessions.custom.customVersion !== customVersion) {
      log("사용자 설정 파일 변경 — DiT 다시 로드", "dim");
      await releaseDit(key);
    }

    const items = [];
    if (!sharedLoaded)
      items.push(
        ["te", PATHS.te],
        ["adapter", PATHS.adapter],
        ["vae", PATHS.vae],
      );
    const ditReloaded = !ditSessions[key]; // 이번 호출에서 DiT를 새로 올리는지 (LoRA 목록이 실제 반영되는 시점)
    if (ditReloaded) {
      // LoRA 장착 시에만 슬롯 스켈레톤 사용 — 샤드는 동일 파일이라 추가 다운로드 없음.
      // 비장착 기본 상태는 슬롯 디스패치 오버헤드가 없는 빠른 그래프로.
      const hasLora = lora.items.some((i) => i.src);
      const ditUrl =
        key === "custom"
          ? { custom: true, slot: hasLora }
          : hasLora && MODELS[key].ditLora
            ? MODELS[key].ditLora
            : MODELS[key].dit;
      items.push([`dit:${key}`, ditUrl]);
    }

    for (let i = 0; i < items.length; i++) {
      const [name, url] = items[i];
      const prog = (f) => {
        $("loadBar").style.width = ((i + f) / items.length) * 100 + "%";
        $("loadStat").textContent = `${name} ${(f * 100) | 0}%`;
      };
      if (name === "vae") {
        const split = await loadVaeSplit(url, prog);
        if (split) {
          sess.vaeHead = split.head;
          sess.vaeTail = split.tail;
          sess.vaeHeadA = split.headA ?? null;
          sess.vaeAttn = split.attn ?? null;
          sess.vaeHeadB = split.headB ?? null;
          continue;
        }
      }
      const s = await loadSession(name, url, prog);
      if (name.startsWith("dit:")) {
        s.requested = settings.precision;
        s.requestedWeights = settings.webnnWeights;
        ditSessions[key] = s;
      } else sess[name] = s;
    }
    sharedLoaded = true;
    sess.dit = ditSessions[key];
    // WebNN DiT면 VAE도 WebNN으로(같은 크기 전용). releaseDit가 sess의 다른 세션과 함께 내리므로 DiT를 새로 올릴 때마다 다시 만든다.
    if (sess.dit.staticLat && !sess.vaeWebnn) sess.vaeWebnn = await loadWebnnVae(PATHS.vae, sess.dit.staticLat);
    // LoRA 상태는 DiT를 새로 올렸을 때만 갱신한다. 이미 상주한 모델이면 LoRA 목록이 반영되지 않았으므로
    // 변경 표시(dirty)를 지우면 안 된다. LoRA를 전부 지운 경우도 여기서 '해제·컴파일...' 문구를 정리한다.
    if (ditReloaded) {
      lora.dirty = false;
      const ab = $("btnLoraApply");
      if (ab) {
        ab.disabled = true;
        ab.textContent = "LoRA 반영";
      }
      $("loraStat").textContent = !lora.items.some((i) => i.src)
        ? ""
        : sess.dit.loraManifest
          ? lora.lastReport
          : "이 모델엔 LoRA 슬롯이 없음 (슬롯 빌드 필요)";
    }
    if (cacheStats.hit + cacheStats.miss)
      log(
        `캐시 ${cacheStats.hit}/${cacheStats.hit + cacheStats.miss}개 사용` +
          (cacheStats.miss ? ` · 다운로드 ${(cacheStats.downloaded / 2 ** 20).toFixed(0)}MB` : ""),
        cacheStats.miss ? "dim" : "ok",
      );
    log(
      `로드 완료 (${((performance.now() - t0) / 1000).toFixed(1)}s)`,
      "ok",
    );
    $("btnGen").disabled = false;
    $("btnGenTop").disabled = false;
    $("loadStat").textContent = `${MODELS[key].label} 준비됨`;
    loadState.warmup = warmup(MODELS[key].label);
    pruneOldCache(); // 백그라운드 — 로드가 끝난 뒤라 쓰는 중인 캐시 파일이 없다
  } catch (e) {
    log("로드 실패: " + e.message, "err");
    console.error(e);
  }
  loadState.loading = false;
  $("btnLoad").disabled = false;
}

// ── 저VRAM 모드 (고급 옵션 settings.lowVram, 표준·빠르게 경로) ──
// ORT WebGPU는 실행 중간 버퍼를 GPU 컨텍스트 캐시에 남기고, 그 캐시는 컨텍스트의 세션이 전부 풀려야 사라진다(releaseDit 주석).
// VRAM 8GB급에서는 VAE 디코드 때 생긴 버퍼가 캐시에 남아, 다음 장부터 DiT 가중치가 시스템 메모리로 밀려나 2배 가까이 느려진다.
// 띠를 더 잘게 나누면 VAE는 빨라져도 캐시에 남는 크기 종류가 늘어 DiT가 더 느려진다. 그래서 VAE 직전에 세션을 전부 내려
// 캐시를 비우고 VAE만 다시 만들어 디코드한 뒤, 장이 끝나면(queue.js pumpQueue) 나머지를 재료(recipe)로 다시 만든다.
// 8GB 흉내 실측(2026-11-14, Turbo 1024², 2장째부터): 35~38s → 약 19.4s(그림 14.8s + 다시 만들기 4.6s), 그림 비트 동일(LoRA 포함).
// 16GB에서는 다시 만드는 시간(3.7~4.3s)과 그 뒤 첫 실행의 셰이더 준비(약 1s)만큼 느려진다(12.6 → 약 18s). 다시 읽은 가중치 사본 때문에 물리 RAM이 약 3GB 더 잡힌다(GC 전까지).
// WebNN DiT에는 쓰지 않는다(별도 장치라 이 캐시와 무관하고, 8GB에서는 WebNN 자체가 맞지 않음).
let lowVramParked = null; // 내린 상태: { key, dit: { recipe, meta }, items: [[sess 키, recipe]] }
function lowVramReady() {
  if (!settings.lowVram || lowVramParked || !sess.dit || sess.dit.staticLat) return false;
  const warn = (msg) => {
    if (lowVramReady.warned === msg) return false;
    lowVramReady.warned = msg;
    log(`저VRAM 모드 — ${msg}`, "dim");
    return false;
  };
  // OPFS 캐시가 없으면(사생활 모드 등) 다시 올릴 때마다 가중치를 네트워크로 다시 받게 된다
  if (opfsDir === false) return warn("이 브라우저는 모델 캐시(OPFS)를 쓸 수 없어 적용하지 않음 (장마다 다시 내려받게 됨)");
  const all = [sess.dit, ...Object.values(sess).filter((s) => s && typeof s.release === "function")];
  if (all.every(recipeComplete)) return true;
  return warn("지금 올라간 모델에는 적용할 수 없음(LoRA 가중치를 보관하지 않음). 모델을 다시 로드하면 적용됩니다");
}
// VAE 디코드 직전: 세션을 전부 내리고 VAE만 다시 만든다.
async function lowVramBeforeVae(key) {
  const d = ditSessions[key];
  const meta = {};
  for (const k of ["loraManifest", "precision", "requested", "requestedWeights", "customVersion"]) meta[k] = d[k];
  const parked = { key, dit: { recipe: d.recipe, meta }, items: [] };
  lowVramParked = parked;
  sess.dit = null;
  delete ditSessions[key];
  await d.release();
  for (const k of Object.keys(sess)) {
    if (!sess[k] || typeof sess[k].release !== "function") continue;
    parked.items.push([k, sess[k].recipe]);
    await sess[k].release();
    sess[k] = null;
  }
  await new Promise((r) => setTimeout(r, 300)); // GPU 버퍼 반납 정리 시간 (releaseDit와 같게)
  for (const [k, r] of parked.items) if (k.startsWith("vae")) sess[k] = await recreateSession(r);
}
// 장이 끝난 뒤(실패해도): 내린 세션을 다시 만든다. 실패하면 모델을 내린 상태로 두고 다시 로드를 안내한다.
async function lowVramRestore() {
  const p = lowVramParked;
  if (!p) return;
  const t0 = performance.now();
  try {
    for (const [k, r] of p.items) if (!sess[k]) sess[k] = await recreateSession(r);
    const d = Object.assign(await recreateSession(p.dit.recipe), p.dit.meta);
    ditSessions[p.key] = d;
    sess.dit = d;
    log(`저VRAM 모드 — 모델 다시 올림 (${((performance.now() - t0) / 1000).toFixed(1)}s)`, "dim");
  } catch (e) {
    console.error(e);
    log(`저VRAM 모드 — 모델을 다시 올리지 못함: ${e.message}. 모델 로드를 다시 눌러 주세요`, "err");
    for (const k of Object.keys(sess)) {
      try {
        if (sess[k] && typeof sess[k].release === "function") await sess[k].release();
      } catch {}
      sess[k] = null;
    }
    sharedLoaded = false;
  }
  lowVramParked = null;
  // 내린 동안 다른 동작(LoRA 반영 시도 등)이 생성 버튼 상태를 바꿨을 수 있다 — 선택한 모델 기준으로 다시 맞춘다(model-ui onModelChange와 같은 기준)
  const ready = !!ditSessions[$("modelSel").value];
  $("btnGen").disabled = !ready;
  $("btnGenTop").disabled = !ready;
}

export { clearCache, customFiles, ditNeedsReload, ditSessions, loadAll, loadState, lowVramBeforeVae, lowVramReady, lowVramRestore, reloadDit, sess, setCustomFiles, tok };
