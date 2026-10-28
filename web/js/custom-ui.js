// '사용자 설정 (직접 모델 지정)': 폴더를 고르면 그 안(하위 폴더 포함)을 훑어 고를 수 있는 DiT 모델 목록을 보여 준다.
// 변환 페이지 결과 폴더(<이름>_base/·_turbo/)든, 여러 모델이 섞인 배포 폴더(out/dit 같은)든 된다.
// 폴더 열기 API(showDirectoryPicker)가 있으면 고른 폴더를 IndexedDB에 기억해 다음 방문에 다시 연다. 없으면 <input webkitdirectory>.
import { $, log } from "./dom.js";
import { VARIANT_DEFAULTS } from "./config.js";
import { setCustomFiles } from "./loader.js";

const HAS_PICKER = typeof window.showDirectoryPicker === "function";
const MAX_DEPTH = 3; // 고른 폴더 아래 몇 단계까지 훑을지 (드라이브 루트를 골라도 끝나게)
const LAST_KEY = "anima.customModel"; // 마지막으로 고른 모델 id (localStorage)

let models = [];
let current = null;

// ── 고른 폴더 기억 (IndexedDB). 실패해도 기능엔 지장 없음 — 다음 방문에 다시 고르면 된다.
function idb(mode, op) {
  return new Promise((resolve) => {
    try {
      const rq = indexedDB.open("anima-custom-dir", 1);
      rq.onupgradeneeded = () => rq.result.createObjectStore("kv");
      rq.onerror = () => resolve(null);
      rq.onsuccess = () => {
        const db = rq.result;
        const tx = db.transaction("kv", mode);
        const r = op(tx.objectStore("kv"));
        tx.oncomplete = () => (db.close(), resolve(r.result));
        tx.onerror = () => (db.close(), resolve(null));
      };
    } catch {
      resolve(null);
    }
  });
}
const saveDir = (h) => idb("readwrite", (s) => s.put(h, "dir"));
const loadDir = () => idb("readonly", (s) => s.get("dir"));

// ── 폴더 훑기 → [{ dir: "고른 폴더 기준 상대 경로", name, get: () => Promise<File> }]
async function walkHandle(dir, path = "", depth = 0, out = []) {
  for await (const [name, h] of dir.entries()) {
    if (h.kind === "file") out.push({ dir: path, name, get: () => h.getFile() });
    else if (depth < MAX_DEPTH) await walkHandle(h, path ? `${path}/${name}` : name, depth + 1, out);
  }
  return out;
}
// <input webkitdirectory>: webkitRelativePath의 첫 칸은 고른 폴더 이름 → 빼서 walkHandle과 같은 상대 경로로
function walkInput(files) {
  return [...files].map((f) => {
    const p = f.webkitRelativePath.split("/").slice(1, -1);
    return { dir: p.join("/"), name: f.name, get: async () => f };
  });
}

const isTurbo = (stem) => /_turbo\d*$/.test(stem);

// 같은 폴더의 '<stem>_q8a4fns.onnx' = 변환·빌드 묶음 하나 (그 폴더에서 '<stem>_'로 시작하는 파일 전부).
// 묶음에 안 속한 .onnx는 단독 그래프 하나 (같은 이름으로 시작하는 .data·manifest·샤드와 함께).
function findModels(entries, rootName) {
  const byDir = new Map();
  for (const e of entries) {
    if (!byDir.has(e.dir)) byDir.set(e.dir, []);
    byDir.get(e.dir).push(e);
  }
  const found = [];
  for (const [dir, list] of byDir) {
    const stems = list.map((e) => /^(.*)_q8a4fns\.onnx$/.exec(e.name)?.[1]).filter(Boolean);
    const taken = new Set();
    for (const stem of stems) {
      const files = list.filter((e) => e.name.startsWith(stem + "_"));
      files.forEach((e) => taken.add(e));
      // 변환 결과처럼 폴더당 묶음 하나면 폴더 이름(사용자가 붙인 이름)이 더 알아보기 쉽다
      const name = stems.length === 1 ? dir.split("/").pop() || rootName : stem;
      found.push({ id: `${dir}/${stem}`, dir, stem, files, label: `${name} (${isTurbo(stem) ? "Turbo" : "베이스"})` });
    }
    for (const e of list) {
      if (!e.name.endsWith(".onnx") || taken.has(e)) continue;
      const files = list.filter((x) => !taken.has(x) && x.name.startsWith(e.name));
      found.push({ id: `${dir}/${e.name}`, dir, stem: null, files, label: `${dir ? dir + "/" : ""}${e.name} (단일 그래프)` });
    }
  }
  return found.sort((a, b) => a.label.localeCompare(b.label));
}

// 묶음이면 그 변형(베이스/Turbo)의 생성 기본값(스텝·CFG·샘플러, 모델 목록 yml의 defaults). 사용자 설정이 선택돼 있을 때만.
function applyDefaults() {
  if ($("modelSel").value !== "custom" || !current?.stem) return;
  const m = VARIANT_DEFAULTS[isTurbo(current.stem) ? "turbo" : "base"];
  if (!m) return;
  for (const k of ["steps", "cfg", "sampler", "scheduler"]) $(k).value = m[k];
  $("cfg").dispatchEvent(new Event("input")); // NegPip 자동 체크 규칙 (model-ui.js)
}

async function selectModel(id) {
  const m = models.find((x) => x.id === id);
  if (!m) return;
  let files;
  try {
    files = await Promise.all(m.files.map((e) => e.get()));
  } catch (e) {
    log(`사용자 모델 파일을 읽지 못함 (${m.label}): ${e.message} — 폴더를 다시 고르세요`, "err");
    return;
  }
  setCustomFiles(files);
  current = m;
  try {
    localStorage.setItem(LAST_KEY, id);
  } catch {}
  log(`사용자 모델: ${m.label} — 파일 ${files.length}개${m.dir ? ` (${m.dir}/)` : ""}`, "ok");
  applyDefaults();
}

function showModels(rootName, entries) {
  models = findModels(entries, rootName);
  const sel = $("customModelSel");
  sel.textContent = "";
  for (const m of models) sel.appendChild(new Option(m.label, m.id));
  sel.style.display = models.length ? "" : "none";
  $("customDirName").textContent = `${rootName}/ — 모델 ${models.length}개`;
  if (!models.length) {
    current = null;
    setCustomFiles([]);
    log(`'${rootName}' 폴더에서 모델을 찾지 못했습니다 — 변환 결과 폴더(또는 그 상위 폴더)를 고르세요`, "err");
    return;
  }
  let last = null;
  try {
    last = localStorage.getItem(LAST_KEY);
  } catch {}
  sel.value = models.some((m) => m.id === last) ? last : models[0].id;
  return selectModel(sel.value);
}

async function openHandle(h) {
  $("customDirName").textContent = `${h.name}/ 훑는 중...`;
  $("btnCustomReopen").style.display = "none";
  try {
    await showModels(h.name, await walkHandle(h));
  } catch (e) {
    $("customDirName").textContent = "";
    log(`폴더를 읽지 못함: ${e.message}`, "err");
  }
}

async function pickFolder() {
  if (!HAS_PICKER) return $("customDirInput").click();
  let h;
  try {
    h = await window.showDirectoryPicker({ id: "anima-custom-models", mode: "read" });
  } catch (e) {
    if (e.name !== "AbortError") log(`폴더 열기 실패: ${e.message}`, "err");
    return;
  }
  await saveDir(h);
  await openHandle(h);
}

export async function initCustomUI() {
  $("btnCustomDir").onclick = pickFolder;
  $("customDirInput").onchange = (e) => {
    const files = e.target.files;
    if (files.length) showModels(files[0].webkitRelativePath.split("/")[0], walkInput(files));
  };
  $("customModelSel").onchange = (e) => selectModel(e.target.value);
  // '사용자 설정'으로 바꿀 때 onModelChange가 custom 항목 기본값을 넣으므로, 그 뒤에 고른 묶음의 기본값으로 덮는다
  $("modelSel").addEventListener("change", applyDefaults);

  // 지난번에 고른 폴더: 권한이 살아 있으면 바로 훑고, 아니면 버튼 한 번(브라우저 권한 확인)으로 다시 연다
  const h = HAS_PICKER ? await loadDir() : null;
  if (!h?.queryPermission) return;
  if ((await h.queryPermission({ mode: "read" })) === "granted") return openHandle(h);
  const btn = $("btnCustomReopen");
  btn.textContent = `다시 열기: ${h.name}/`;
  btn.style.display = "";
  btn.onclick = async () => {
    if ((await h.requestPermission({ mode: "read" })) === "granted") openHandle(h);
  };
}

export { findModels, openHandle, walkInput }; // 확인용
