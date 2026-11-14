// 고급 옵션 모달 (모델 행의 톱니바퀴). 연산 정밀도(표준/빠르게/WebNN) + 속도 옵션(패딩 압축, EasyCache, Spectrum) + 메모리(저VRAM 모드).
// 설정은 settings.js에 저장되고, 실제 반영은 모델 로드 시점(loader.js loadAll — 상주 DiT의 요청 모드와 다르면 다시 올림).
import { $ } from "./dom.js";
import { ditNeedsReload, ditSessions, loadAll, loadState } from "./loader.js";
import { HAS_WEBNN } from "./ort.js";
import { queueBusy } from "./queue-state.js";
import { PRECISION_LABEL, saveSettings, settings } from "./settings.js";

function refresh() {
  const dlg = $("advDialog");
  for (const r of dlg.querySelectorAll('input[name="precision"]')) {
    r.checked = r.value === settings.precision;
    r.disabled = loadState.loading || (r.value === "webnn" && !HAS_WEBNN); // 로드 도중 바꾸면 요청 모드와 실제 그래프가 어긋난다
  }
  // WebNN 가중치(fp16/int8)는 WebNN을 골랐을 때만 의미가 있다
  for (const r of dlg.querySelectorAll('input[name="webnnWeights"]')) {
    r.checked = r.value === settings.webnnWeights;
    r.disabled = loadState.loading || !HAS_WEBNN || settings.precision !== "webnn";
  }
  if (!HAS_WEBNN)
    $("advWebnnNote").textContent =
      "이 브라우저에는 WebNN(navigator.ml)이 없습니다. Chrome은 chrome://flags에서 WebNN 플래그를 켜야 합니다.";
  const dit = ditSessions[$("modelSel").value];
  const btn = $("btnAdvApply");
  let status;
  if (!dit) {
    status = "선택한 모델이 아직 로드되지 않았습니다 — 로드할 때 적용됩니다.";
    btn.hidden = true;
  } else {
    status = `로드된 모델: ${PRECISION_LABEL[dit.precision]}${dit.weights ? " " + dit.weights : ""}`;
    if (dit.precision !== dit.requested)
      status += dit.requested === "webnn" ? " (WebNN을 쓸 수 없어 표준으로 로드됨 — 이유는 로그)" : ` (${PRECISION_LABEL[dit.requested]} 그래프가 서버에 없음)`;
    else if (dit.weights && dit.weights !== dit.requestedWeights) status += " (int8 그래프가 없어 fp16으로 로드됨)";
    const pending = ditNeedsReload(dit);
    if (pending)
      status += ` → 다음 로드부터 ${PRECISION_LABEL[settings.precision]}${settings.precision === "webnn" ? " " + settings.webnnWeights : ""}`;
    btn.hidden = !pending;
    btn.disabled = queueBusy() || loadState.loading;
  }
  // 패딩 압축은 그래프가 xattn_index 입력을 가질 때만 동작 (export/xattn_compact.py로 만든 그래프)
  $("advXattn").checked = settings.xattnCompact;
  if (dit && !dit.inputNames.includes("xattn_index"))
    status += " · 패딩 압축: 이 모델 그래프는 미지원 (서버 그래프 갱신 필요)";
  $("advEasyCache").checked = settings.easyCache;
  // Spectrum은 본체(+특징 출력)·머리 그래프를 올린 WebNN 세션에서만 동작 (export/webnn/spectrum_split.py)
  $("advSpectrum").checked = settings.spectrum;
  if (settings.spectrum && dit && !dit.spectrum)
    status += " · Spectrum: 지금 로드된 모델에서는 쓸 수 없음 (WebNN + Spectrum 그래프 필요)";
  // 저VRAM 모드는 WebGPU 세션(표준·빠르게)에만 적용된다 (loader.js lowVramBeforeVae)
  $("advLowVram").checked = settings.lowVram;
  if (settings.lowVram && dit && dit.staticLat) status += " · 저VRAM 모드: WebNN으로 로드한 모델에는 적용되지 않음";
  $("advStatus").textContent = status;
}

function initAdvanced() {
  const dlg = $("advDialog");
  $("btnAdvanced").onclick = () => {
    refresh();
    dlg.showModal();
  };
  dlg.addEventListener("change", (e) => {
    if (e.target.name === "precision") settings.precision = e.target.value;
    else if (e.target.name === "webnnWeights") settings.webnnWeights = e.target.value;
    else if (e.target.id === "advXattn") settings.xattnCompact = e.target.checked;
    else if (e.target.id === "advEasyCache") settings.easyCache = e.target.checked;
    else if (e.target.id === "advSpectrum") settings.spectrum = e.target.checked;
    else if (e.target.id === "advLowVram") settings.lowVram = e.target.checked;
    else return;
    saveSettings();
    refresh();
  });
  $("btnAdvApply").onclick = () => {
    dlg.close();
    loadAll(); // 상주 DiT만 새 모드로 다시 올린다 (진행·오류는 로그)
  };
  // 바깥(배경) 클릭으로 닫기 — 대화상자 안쪽 여백 클릭은 무시
  dlg.addEventListener("click", (e) => {
    const r = dlg.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) dlg.close();
  });
}

export { initAdvanced };
