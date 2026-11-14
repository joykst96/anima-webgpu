// 고급 옵션 설정값 (톱니바퀴 모달). loader·advanced가 함께 읽으므로 의존성 없는 모듈로 둔다.
// 브라우저 저장소는 막혀 있을 수 있어(사생활 모드 등) 읽기·쓰기 실패는 기본값으로 진행한다.
const KEY = "anima.settings.v1";
// precision: "standard" | "fast" | "webnn" (다음 로드부터) · xattnCompact: cross-attention 패딩 압축 (다음 생성부터, 재로드 불필요)
// easyCache: 변화가 작은 스텝의 DiT 호출 건너뛰기 (다음 생성부터)
// spectrum: 건너뛰는 스텝의 특징을 예측해 머리만 돌리기 (다음 생성부터. WebNN으로 로드했고 Spectrum 그래프가 있을 때만 — spectrum.js)
// webnnWeights: "fp16" | "int8" — WebNN 가중치 (다음 로드부터). int8은 표준(WebGPU) 샤드를 그대로 쓰는 그래프(<dit>_wnn1024q*.onnx,
//   export/webnn/webnn_stdq8.py)라 추가 다운로드가 없다. 그 그래프가 없으면 fp16으로 로드한다(loader.js loadWebnnDit).
// lowVram: VAE 직전에 세션을 전부 내렸다가 장이 끝나면 다시 올린다(VRAM 8GB급용, 다음 생성부터. 표준·빠르게에서만 — loader.js lowVramBeforeVae).
//   LoRA를 쓴 모델은 켠 뒤 다시 로드해야 적용된다(병합한 LoRA 가중치를 켰을 때만 보관).
const DEFAULTS = { precision: "standard", xattnCompact: false, easyCache: false, spectrum: false, webnnWeights: "fp16", lowVram: false };

function load() {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) || "{}");
    return { ...DEFAULTS, ...saved };
  } catch {
    return { ...DEFAULTS };
  }
}

const settings = load();

function saveSettings() {
  try {
    localStorage.setItem(KEY, JSON.stringify(settings));
  } catch {
    /* 저장 불가 — 이번 세션에서만 유지 */
  }
}

// 정밀도 모드 표시 이름 (모달·로그·이미지 메타데이터 공용)
// webnn: WebNN 가속(실험) — 브라우저에 navigator.ml이 있고 모델에 WebNN 그래프가 있을 때만. 없으면 표준으로 로드된다(loader.js).
const PRECISION_LABEL = { standard: "표준", fast: "빠르게", webnn: "WebNN" };

export { PRECISION_LABEL, saveSettings, settings };
