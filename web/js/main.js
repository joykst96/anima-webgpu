// 진입점: 모듈 로드 후 원래 순서대로 초기화
// transformers.js(tokenizer.js)를 ORT보다 먼저 평가 — 원본의 정적 import → ORT 동적 import 순서 유지
import "./tokenizer.js";
import { $ } from "./dom.js";
import { HAS_JSPI } from "./ort.js";
import { clearCache, loadAll } from "./loader.js";
import { applyModelSettings, initModelUI, onModelChange } from "./model-ui.js";
import { initCustomUI } from "./custom-ui.js";
import { downloadAllZip, initFolderMode, initLightbox } from "./gallery.js";
import { cancelCurrent, runBatch } from "./queue.js";
import { initPresets } from "./presets.js";
import { initLoraUI } from "./lora-ui.js";
import { initAutocomplete } from "./autocomplete.js";
import { renderSiteDefaults } from "./site.js";
import { initAdvanced } from "./advanced.js";

let gpuPreference = "high-performance"; // 고정 (GPU 선호도 UI 제거 — 대부분 환경서 무시됨)

// 배포 구성 → 드롭다운/프롬프트/출처 링크. modelSel.value를 읽는 어떤 코드보다 먼저.
renderSiteDefaults();

initLightbox();

// ───────── 초기화 ─────────
(async () => {
  const fail = (msg) => {
    $("gpuInfo").innerHTML =
      `<span style="color:var(--err)">${msg}</span>`;
    $("btnLoad").disabled = true;
  };
  if (!window.isSecureContext)
    return fail(
      "보안 컨텍스트가 아닙니다 (http + IP 접속). WebGPU는 HTTPS 또는 localhost에서만 동작합니다 — https 주소로 접속하세요.",
    );
  if (!navigator.gpu)
    return fail(
      "이 브라우저는 WebGPU를 지원하지 않습니다. 데스크톱 Chrome/Edge 최신 버전이 필요합니다.",
    );
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: gpuPreference,
  });
  if (!adapter)
    return fail(
      "GPU 어댑터를 찾을 수 없습니다. GPU 드라이버를 확인하세요.",
    );
  const f16 = adapter.features.has("shader-f16");
  if (!f16)
    return fail(
      "이 GPU/드라이버는 shader-f16을 지원하지 않아 모델 실행이 불가능합니다.",
    );
  const bufGB = adapter.limits.maxBufferSize / 2 ** 30;
  const warn =
    bufGB < 2
      ? ' · <span style="color:var(--err)">maxBufferSize가 작아 DiT 로드가 실패할 수 있음</span>'
      : "";
  const engine = HAS_JSPI
    ? '<span style="color:var(--ok)">native EP (jspi)</span>'
    : '<span style="color:var(--warn)">JSEP ⚠ — DP4A·FlashAttention 가속 불가 (Chrome/Edge 권장)</span>';
  $("gpuInfo").innerHTML =
    `WebGPU OK · shader-f16 <span style="color:var(--ok)">지원</span>` +
    ` · maxBufferSize ${bufGB.toFixed(1)}GB` +
    ` · 엔진 ${engine}${warn}` +
    `<br><span style="color:var(--dim)">권장: VRAM 8GB+ · 최초 로드 시 ~2.5GB 다운로드 (이후 캐시)</span>`;
})();

$("btnLoad").onclick = loadAll;
$("btnClearCache").onclick = clearCache;
$("modelSel").onchange = onModelChange;
applyModelSettings();
initPresets();

$("btnGen").onclick = runBatch;
$("btnCancel").onclick = cancelCurrent;
$("btnGenTop").onclick = runBatch;
$("btnCancelTop").onclick = cancelCurrent;
$("btnSaveAll").onclick = () => downloadAllZip($("btnSaveAll"));

initFolderMode();

// 아코디언: 헤더 클릭으로 패널 접기/펴기
document.querySelectorAll(".accordion .acc-head").forEach((head) => {
  head.onclick = () =>
    head.closest(".accordion").classList.toggle("collapsed");
});
$("btnSave").onclick = () => {
  const a = document.createElement("a");
  a.download = `anima_${Date.now()}.png`;
  a.href = $("canvas").toDataURL("image/png");
  a.click();
};

initCustomUI();
initLoraUI();
initModelUI();
initAutocomplete();
initAdvanced();

