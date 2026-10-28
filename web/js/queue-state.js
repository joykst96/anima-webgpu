// 생성 큐 상태. 실행(queue.js) 외에 폴더 지정 잠금(gallery.js) 등이 읽으므로
// 별도 모듈로 둬서 queue ↔ gallery 순환 의존을 끊는다.
export const genQueue = []; // 대기 작업 (파라미터 스냅샷)
export const jobState = { cur: null }; // cur: 실행 중 작업

// 생성이 진행 중이거나 대기 중이면 true. 이때 DiT 세션을 내리는 동작(모델 로드·LoRA 반영)은 막는다.
export const queueBusy = () => genQueue.length > 0 || jobState.cur !== null;
export const BUSY_MSG = "생성 중에는 모델 로드/LoRA 반영을 할 수 없습니다 — 큐가 끝나거나 중단한 뒤 다시 시도하세요";
