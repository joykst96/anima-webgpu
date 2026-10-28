// 결과 갤러리·라이트박스·ZIP 저장·폴더 자동저장
import { $, log } from "./dom.js";
import { genQueue, jobState } from "./queue-state.js";
import { acEscape } from "./autocomplete.js";

// ───────── 생성 결과 갤러리 (세션 메모리 — 새로고침 시 소멸) ─────────
// 각 항목: { id, url(dataURL), w, h, meta }. localStorage 미사용(아티팩트/프라이버시).
const gallery = [];
let galleryId = 0;

// ───────── 폴더 자동저장 모드 (File System Access API · Chrome 계열) ─────────
// 폴더를 걸면 이후 생성 결과가 그 폴더에 {prefix}_{00001}.png 로 자동 저장된다.
// 원칙 유지: 파일은 사용자 로컬 디스크에 직접 쓰이며 서버를 경유하지 않는다.
// 표시 범위는 두 모드 모두 '이 세션에 생성한 것'만. 폴더 기존 내용은 스캔하지 않되,
// 카운터 base 산정을 위해 '파일 이름'만 훑는다(내용은 읽지 않음).
const folderMode = {
  active: false,
  dir: null, // FileSystemDirectoryHandle
  prefix: "anima", // 새니타이즈된 접두사
  counter: 1, // 다음 저장 번호
};
const HAS_FS_ACCESS = typeof window.showDirectoryPicker === "function";

// 접두사 새니타이즈: 영문/숫자/_/- 만. 그 외 제거. 비면 'anima'.
function sanitizePrefix(s) {
  const clean = (s || "").replace(/[^A-Za-z0-9_-]/g, "");
  return clean || "anima";
}
const pad5 = (n) => String(n).padStart(5, "0");
const fileNameFor = (prefix, n) => `${prefix}_${pad5(n)}.png`;

// 폴더 이름만 훑어 {prefix}_{숫자}.png 최댓값+1 을 base로. (내용 미열람)
// 낙관적: 중간 빈 번호가 있어도 메우지 않고 항상 최댓값 위로 쌓는다.
async function scanCounterBase(dir, prefix) {
  const re = new RegExp(
    "^" + prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "_(\\d+)\\.png$",
    "i",
  );
  let max = 0;
  try {
    for await (const [name, handle] of dir.entries()) {
      if (handle.kind !== "file") continue;
      const m = name.match(re);
      if (m) {
        const v = parseInt(m[1], 10);
        if (v > max) max = v;
      }
    }
  } catch (e) {
    console.warn("폴더 스캔 실패:", e);
  }
  return max + 1;
}

function updateSaveModeUI() {
  const ind = $("saveModeInd");
  const wrap = $("savePrefixWrap");
  const txt = ind.querySelector(".sm-txt");
  if (folderMode.active) {
    ind.classList.add("folder");
    txt.innerHTML = `폴더 자동저장 · <span class="sm-name">${acEscape(folderMode.dir.name)}</span>`;
    wrap.classList.add("on");
    $("btnPickDir").disabled = true;
    $("savePrefix").value = folderMode.prefix;
    $("savePrefixEg").textContent = `_${pad5(folderMode.counter)}.png`;
  } else {
    ind.classList.remove("folder");
    txt.textContent = HAS_FS_ACCESS
      ? "세션 저장 (수동)"
      : "세션 저장 (이 브라우저는 폴더 저장 미지원)";
    wrap.classList.remove("on");
  }
}

// 폴더 선택은 '첫 생성 전'에만 허용. 큐·실행·갤러리가 모두 비어있어야 함.
function canPickFolder() {
  return (
    !folderMode.active &&
    gallery.length === 0 &&
    genQueue.length === 0 &&
    !jobState.cur
  );
}
function refreshPickDirBtn() {
  const btn = $("btnPickDir");
  if (!btn) return;
  if (folderMode.active) {
    btn.disabled = true;
    return;
  }
  btn.disabled = !HAS_FS_ACCESS || !canPickFolder();
}

async function pickFolder() {
  if (!HAS_FS_ACCESS) {
    log(
      "이 브라우저는 폴더 저장을 지원하지 않습니다 (Chrome/Edge 계열 필요)",
      "err",
    );
    return;
  }
  if (!canPickFolder()) {
    log("폴더는 첫 생성 전에만 지정할 수 있습니다", "err");
    return;
  }
  let dir;
  try {
    dir = await window.showDirectoryPicker({ mode: "readwrite" });
  } catch (e) {
    // 사용자 취소 등 — 조용히 일반 모드 유지
    return;
  }
  // 쓰기 권한 확인/요청
  try {
    if (dir.queryPermission) {
      let p = await dir.queryPermission({ mode: "readwrite" });
      if (p !== "granted" && dir.requestPermission)
        p = await dir.requestPermission({ mode: "readwrite" });
      if (p !== "granted") {
        log("폴더 쓰기 권한이 거부되었습니다", "err");
        return;
      }
    }
  } catch (e) {
    log("폴더 권한 확인 실패: " + e.message, "err");
    return;
  }
  folderMode.dir = dir;
  folderMode.active = true;
  folderMode.prefix = sanitizePrefix($("savePrefix").value);
  folderMode.counter = await scanCounterBase(dir, folderMode.prefix);
  updateSaveModeUI();
  refreshPickDirBtn();
  // 폴더 모드에선 '현재 이미지 저장'이 무의미 → 자동저장 안내로 대체
  const bs = $("btnSave");
  bs.disabled = true;
  bs.textContent = "자동으로 저장됩니다";
  log(
    `폴더 자동저장 ON — '${dir.name}' · ${fileNameFor(folderMode.prefix, folderMode.counter)}부터`,
    "ok",
  );
}

// canvas → PNG Blob
function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("toBlob 실패"))),
      "image/png",
    );
  });
}

// 폴더에 blob 저장. 성공 시 {fileName} 반환.
async function saveBlobToFolder(blob) {
  const name = fileNameFor(folderMode.prefix, folderMode.counter);
  const fh = await folderMode.dir.getFileHandle(name, { create: true });
  const w = await fh.createWritable();
  await w.write(blob);
  await w.close();
  folderMode.counter++;
  $("savePrefixEg").textContent = `_${pad5(folderMode.counter)}.png`;
  return name;
}

function renderThumb(item) {
  const btn = document.createElement("button");
  btn.className = "thumb";
  btn.title = item.meta;
  btn.dataset.gid = item.id;
  const img = document.createElement("img");
  img.src = item.url;
  img.alt = item.meta;
  btn.appendChild(img);
  // 우상단 빨간 × 삭제 (썸네일 클릭과 분리)
  const del = document.createElement("span");
  del.className = "del";
  del.textContent = "×";
  del.title = "이 결과 삭제";
  del.onclick = (e) => {
    e.stopPropagation();
    removeFromGallery(item);
  };
  btn.appendChild(del);
  btn.onclick = () => showFromGallery(item);
  return btn;
}

// 썸네일 클릭 → 메인 캔버스에 다시 크게 표시. 이후 'PNG 저장'은 이 이미지를 저장.
function showFromGallery(item) {
  const cv = $("canvas");
  const draw = () => {
    cv.width = item.w;
    cv.height = item.h;
    cv.getContext("2d").drawImage(im2, 0, 0);
    if (!folderMode.active) $("btnSave").disabled = false;
    $("genStat").textContent = item.meta;
    markActiveThumb(item.id);
  };
  const im2 = new Image();
  im2.onload = draw;
  im2.src = item.url;
}

// item.id 기준으로 active 표시 (삭제로 인덱스 어긋나도 안전)
function markActiveThumb(id) {
  [...$("gallery").children].forEach((el) =>
    el.classList.toggle("active", el.dataset.gid === String(id)),
  );
}

// ───────── 라이트박스 모달 (크게 보기 + 방향키 순회) ─────────
let lbIndex = -1; // gallery 배열 인덱스

function openLightbox(item) {
  const idx = gallery.indexOf(item);
  if (idx < 0) return;
  lbIndex = idx;
  renderLightbox();
  $("lightbox").classList.add("open");
}

function renderLightbox() {
  const item = gallery[lbIndex];
  if (!item) {
    closeLightbox();
    return;
  }
  $("lbImg").src = item.url;
  $("lbMeta").textContent =
    `${item.meta}   ·   ${lbIndex + 1} / ${gallery.length}`;
  // 스택은 최신이 앞(idx 0). ‹ 이전 = 더 최신, › 다음 = 더 오래된 것.
  $("lbPrev").disabled = lbIndex <= 0;
  $("lbNext").disabled = lbIndex >= gallery.length - 1;
}

function lbStep(d) {
  const n = lbIndex + d;
  if (n < 0 || n >= gallery.length) return;
  lbIndex = n;
  renderLightbox();
}

function closeLightbox() {
  $("lightbox").classList.remove("open");
  lbIndex = -1;
}

export function initLightbox() {
  $("lbClose").onclick = closeLightbox;
  $("lbPrev").onclick = () => lbStep(-1);
  $("lbNext").onclick = () => lbStep(1);
  // '화면에 맞춰 늘리기' 토글 — 저해상도 이미지를 화면 크기로 업스케일(비율 유지).
  $("lbFitChk").onchange = (e) => {
    $("lightbox").classList.toggle("fit", e.target.checked);
  };
  $("lbFit").onclick = (e) => e.stopPropagation(); // 라벨 클릭이 배경닫기로 안 가게
  // 배경(이미지·버튼 외) 클릭 시 닫기
  $("lightbox").onclick = (e) => {
    if (e.target.id === "lightbox") closeLightbox();
  };
  // 메인 캔버스 클릭 → 현재 보이는 이미지를 모달로. 갤러리에 있으면 그 위치부터 순회.
  $("canvas").onclick = () => {
    if (!gallery.length) return;
    // 현재 active 썸네일(=캔버스에 그려진 것)을 찾고, 없으면 최신(0).
    const activeEl = [...$("gallery").children].find((c) =>
      c.classList.contains("active"),
    );
    const item = activeEl
      ? gallery.find((g) => String(g.id) === activeEl.dataset.gid)
      : gallery[0];
    if (item) openLightbox(item);
  };
  // 키보드: 모달 열렸을 때 ←/→ 순회, Esc 닫기
  window.addEventListener("keydown", (e) => {
    if (!$("lightbox").classList.contains("open")) return;
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      lbStep(-1);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      lbStep(1);
    } else if (e.key === "Escape") {
      e.preventDefault();
      closeLightbox();
    }
  });
}

// 개별 삭제: 배열 + DOM 모두 item 기준 제거.
// 폴더 모드 항목이면 디스크 파일도 삭제하고 objectURL을 해제한다(확인 없음).
function removeFromGallery(item) {
  const idx = gallery.indexOf(item);
  if (idx >= 0) gallery.splice(idx, 1);
  const el = [...$("gallery").children].find(
    (c) => c.dataset.gid === String(item.id),
  );
  if (el) el.remove();
  // 폴더 모드: 디스크 파일 삭제 + objectURL revoke
  if (item.fileName && folderMode.dir) {
    folderMode.dir
      .removeEntry(item.fileName)
      .catch((e) => log(`파일 삭제 실패: ${item.fileName} (${e.message})`, "err"));
  }
  if (item.objectUrl) {
    try {
      URL.revokeObjectURL(item.objectUrl);
    } catch (_) {}
  }
  updateGalleryBar();
}

// 갤러리에 항목 추가하는 공통부 (스택: 최신이 앞).
function pushGalleryItem(item) {
  gallery.unshift(item);
  const g = $("gallery");
  g.insertBefore(renderThumb(item), g.firstChild);
  markActiveThumb(item.id);
  updateGalleryBar();
}

// 일반 모드: 캔버스 내용을 dataURL로 세션 메모리에 보관.
function pushToGallery(canvas, meta) {
  pushGalleryItem({
    id: ++galleryId,
    url: canvas.toDataURL("image/png"),
    w: canvas.width,
    h: canvas.height,
    meta,
  });
}

// 폴더 모드: 디스크에 쓴 blob의 objectURL을 썸네일 소스로 사용(dataURL 미보관).
// fileName을 보관해 개별 삭제 시 디스크에서 제거한다.
function pushToGalleryFromBlob(blob, w, h, meta, fileName) {
  const objectUrl = URL.createObjectURL(blob);
  pushGalleryItem({
    id: ++galleryId,
    url: objectUrl,
    objectUrl,
    fileName,
    w,
    h,
    meta,
  });
}

// 갤러리 1장 이상이면 '전체 저장' 버튼 활성 + 개수 표시.
function updateGalleryBar() {
  const btn = $("btnSaveAll");
  if (!btn) return;
  btn.disabled = gallery.length < 1;
  btn.textContent =
    gallery.length > 0 ? `전체 저장 (${gallery.length})` : "전체 저장";
}

// 전체 결과를 ZIP으로 묶어 다운로드. JSZip은 다운로드 시점에만 CDN에서 동적 로드.
let _jszipLoading = null;
function loadJSZip() {
  if (window.JSZip) return Promise.resolve(window.JSZip);
  if (_jszipLoading) return _jszipLoading;
  _jszipLoading = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js";
    s.onload = () =>
      window.JSZip
        ? resolve(window.JSZip)
        : reject(new Error("JSZip 전역 없음"));
    s.onerror = () => reject(new Error("JSZip 스크립트 로드 실패"));
    document.head.appendChild(s);
  });
  return _jszipLoading;
}

// 갤러리 항목들 → ZIP Blob. 항목 url은 일반 모드면 data: URL, 폴더 모드면 blob: objectURL이라
// 문자열 분해(split(",")) 대신 fetch로 바이트를 읽는다 — 두 형식 모두 원본 PNG 바이트 그대로.
async function buildGalleryZip(items) {
  const JSZip = await loadJSZip();
  const zip = new JSZip();
  // 최신이 앞이므로 역순으로 번호 매겨 시간순 정렬 (001=가장 오래된 것)
  const ordered = [...items].reverse();
  for (let i = 0; i < ordered.length; i++) {
    const it = ordered[i];
    const bytes = new Uint8Array(await (await fetch(it.url)).arrayBuffer());
    const name = `anima_${String(i + 1).padStart(3, "0")}_${it.w}x${it.h}.png`;
    zip.file(name, bytes);
  }
  return zip.generateAsync({ type: "blob" });
}

async function downloadAllZip(btn) {
  if (gallery.length < 1) return;
  const orig = btn.textContent;
  btn.disabled = true;
  btn.textContent = "ZIP 준비 중...";
  try {
    const blob = await buildGalleryZip(gallery);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `anima_gallery_${gallery.length}_${Date.now()}.zip`;
    a.click();
    URL.revokeObjectURL(a.href);
    btn.textContent = orig;
  } catch (e) {
    log("ZIP 저장 실패: " + e.message, "err");
    btn.textContent = orig;
  } finally {
    btn.disabled = false;
  }
}
export function initFolderMode() {
  // ── 폴더 자동저장 모드 바인딩 ──
  $("btnPickDir").onclick = pickFolder;
  // 접두사 수정: 새니타이즈 후 재스캔하여 카운터 base 갱신 (폴더 모드에서만 유효)
  let _prefixScanSeq = 0;
  $("savePrefix").addEventListener("change", async () => {
    const clean = sanitizePrefix($("savePrefix").value);
    $("savePrefix").value = clean;
    if (!folderMode.active) return;
    folderMode.prefix = clean;
    const my = ++_prefixScanSeq;
    folderMode.counter = await scanCounterBase(folderMode.dir, clean);
    if (my !== _prefixScanSeq) return; // 더 최신 스캔이 있으면 폐기
    $("savePrefixEg").textContent = `_${pad5(folderMode.counter)}.png`;
    log(
      `접두사 '${clean}' — ${fileNameFor(clean, folderMode.counter)}부터 저장`,
      "dim",
    );
  });
  // 초기 인디케이터/버튼 상태
  updateSaveModeUI();
  refreshPickDirBtn();
}

export { buildGalleryZip, canvasToBlob, downloadAllZip, folderMode, pushToGallery, pushToGalleryFromBlob, refreshPickDirBtn, saveBlobToFolder };
