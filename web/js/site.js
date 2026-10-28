// 배포 구성(config.js)을 화면에 반영: 모델 드롭다운, 프롬프트 기본값, 출처 링크.
// 예전엔 gen_models.py가 이 내용을 index.html에 직접 써 넣었다 — 이제 HTML은 배포와 무관하다.
import { $ } from "./dom.js";
import { CONVERT_URL, MODELS, MODEL_SOURCES, PROMPT_DEFAULTS, SELECTED_MODEL } from "./config.js";

// 다른 모듈이 modelSel.value를 읽기 전에 호출해야 한다 (main.js 맨 앞).
export function renderSiteDefaults() {
  // 드롭다운: MODELS 순서대로, HTML에 남아 있는 '사용자 설정'(custom) 옵션 앞에 삽입
  const sel = $("modelSel");
  const customOpt = sel.querySelector('option[value="custom"]');
  for (const [key, m] of Object.entries(MODELS)) {
    if (key === "custom") continue;
    const opt = new Option(m.label, key, key === SELECTED_MODEL, key === SELECTED_MODEL);
    sel.insertBefore(opt, customOpt);
  }
  if (SELECTED_MODEL in MODELS) sel.value = SELECTED_MODEL;

  // 프롬프트 기본값 (textContent = 초기값; 사용자가 고친 value와 별개인 defaultValue)
  if (PROMPT_DEFAULTS.positive !== undefined) $("prompt").textContent = PROMPT_DEFAULTS.positive;
  if (PROMPT_DEFAULTS.negative !== undefined) $("negative").textContent = PROMPT_DEFAULTS.negative;

  // 파인튜닝 모델 출처 링크 (출처가 하나도 없는 배포면 제목째 숨김)
  $("modelSourcesBlock").hidden = !MODEL_SOURCES.length;
  const box = $("modelSources");
  for (const { label, url } of MODEL_SOURCES) {
    const a = document.createElement("a");
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = label;
    box.appendChild(a);
  }

  // 변환 페이지 링크 ('사용자 설정' 안내문). 별도 서브도메인이면 배포 목록의 convert_url, 없으면 HTML 기본값(convert/)
  if (CONVERT_URL) $("convertLink").href = CONVERT_URL;
}
