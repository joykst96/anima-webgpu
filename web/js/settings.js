// 고급 옵션 설정값 (톱니바퀴 모달). loader·advanced가 함께 읽으므로 의존성 없는 모듈로 둔다.
// 브라우저 저장소는 막혀 있을 수 있어(사생활 모드 등) 읽기·쓰기 실패는 기본값으로 진행한다.
const KEY = "anima.settings.v1";
// precision: "standard" | "fast" (다음 로드부터) · xattnCompact: cross-attention 패딩 압축 (다음 생성부터, 재로드 불필요)
// easyCache: 변화가 작은 스텝의 DiT 호출 건너뛰기 (다음 생성부터)
const DEFAULTS = { precision: "standard", xattnCompact: false, easyCache: false };

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
const PRECISION_LABEL = { standard: "표준", fast: "빠르게" };

export { PRECISION_LABEL, saveSettings, settings };
