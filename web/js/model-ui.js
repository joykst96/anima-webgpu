// 모델 선택에 따른 설정·NegPip/CFG UI 상태
import { $ } from "./dom.js";
import { MODELS } from "./config.js";
import { ditSessions, sess } from "./loader.js";
import { lora, LORA_DIRTY_MSG } from "./lora.js";

function applyModelSettings() {
  const m = MODELS[$("modelSel").value];
  $("steps").value = m.steps;
  $("cfg").value = m.cfg;
  $("sampler").value = m.sampler;
  $("scheduler").value = m.scheduler;
}

let _prevCfg = 1; // CFG 변경 추적 (초기값은 아래 초기화에서 실제값으로 동기화)
function updateModelUI() {
  const key = $("modelSel").value;
  $("customPanel").style.display = key === "custom" ? "" : "none";
  const cfg = +$("cfg").value; // 사용자가 직접 바꾼 실제 CFG 기준
  const chk = $("negpipChk"),
    nw = $("negpipW"),
    neg = $("negative");
  // CFG>1: 일반 부정 프롬프트가 동작 → NegPip 무의미(해제+비활성). 부정칸은 활성 유지.
  // CFG=1: NegPip 미체크면 부정 프롬프트가 무시됨 → 부정칸 비활성.
  if (cfg > 1) {
    chk.checked = false;
    chk.disabled = true;
    nw.disabled = true;
    neg.disabled = false; // 터보여도 CFG>1이면 부정칸 활성
    neg.placeholder = "";
    chk.parentElement.title =
      "CFG>1에서는 일반 부정 프롬프트가 동작하므로 NegPip 불필요";
  } else {
    chk.disabled = false;
    nw.disabled = !chk.checked;
    neg.disabled = !chk.checked; // CFG=1: NegPip 없으면 부정 프롬프트 무시되므로 비활성
    neg.placeholder = chk.checked
      ? ""
      : "CFG=1에서는 NegPip 체크 시에만 부정 프롬프트가 적용됩니다";
    chk.parentElement.title = "";
  }
}

function onModelChange() {
  applyModelSettings();
  _prevCfg = +$("cfg").value; // 모델 전환으로 바뀐 CFG 동기화
  // 모델 전환 시 CFG=1이면 NegPip 자동 활성화 — 안 켜면 부정 프롬프트가 무시되므로.
  // (전환 순간에만. 이후 사용자가 직접 끄는 건 updateModelUI가 존중.)
  if (+$("cfg").value <= 1) $("negpipChk").checked = true;
  updateModelUI();
  const key = $("modelSel").value;
  if (ditSessions[key]) {
    sess.dit = ditSessions[key];
    $("btnGen").disabled = false;
    $("btnGenTop").disabled = false;
    $("loadStat").textContent = `${MODELS[key].label} 준비됨`;
  } else {
    $("btnGen").disabled = true;
    $("btnGenTop").disabled = true;
    $("loadStat").textContent =
      `${MODELS[key].label} 가중치 필요 — 모델 로드 버튼 클릭`;
  }
  // 반영 안 된 LoRA 변경이 있으면 문구를 기본 '변경됨'으로 되돌린다
  // (미로드 모델에서 반영을 시도했을 때의 안내가 모델을 바꾼 뒤에도 남지 않도록)
  if (lora.dirty) $("loraStat").textContent = LORA_DIRTY_MSG;
}
export function initModelUI() {
  $("negpipChk").addEventListener("change", updateModelUI);
  // CFG 직접 변경: 1 이하로 내려오면 NegPip 자동 활성화(최초), 1 초과면 updateModelUI가 해제.
  _prevCfg = +$("cfg").value;
  $("cfg").addEventListener("input", () => {
    const cur = +$("cfg").value;
    if (cur <= 1 && _prevCfg > 1) $("negpipChk").checked = true; // 1초과→1: 자동 활성
    _prevCfg = cur;
    updateModelUI();
  });
  // 초기 로드: 기본 모델이 CFG=1이면 NegPip 자동 활성화
  if (+$("cfg").value <= 1) $("negpipChk").checked = true;
  updateModelUI();
}

export { applyModelSettings, onModelChange, updateModelUI };
