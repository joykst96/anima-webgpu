// 멀티 LoRA 입력 UI + 반영
import { $, log } from "./dom.js";
import { lora, LORA_DIRTY_MSG } from "./lora.js";
import { ditSessions, reloadDit } from "./loader.js";
import { BUSY_MSG, queueBusy } from "./queue-state.js";

let loraSeq = 0;
// ── 멀티 LoRA UI ──
function markLoraDirty() {
  lora.dirty = true;
  const btn = $("btnLoraApply");
  btn.disabled = queueBusy(); // 생성 중엔 반영 불가 — 큐가 끝나면 queue.js가 다시 켠다
  btn.textContent = "LoRA 반영 ●";
  $("loraStat").textContent = LORA_DIRTY_MSG;
}

function createLoraRow(item) {
  const row = document.createElement("div");
  row.dataset.loraId = item.id;
  row.style.cssText =
    "display:flex;gap:6px;align-items:center;margin-bottom:5px";
  const file = document.createElement("input");
  file.type = "file";
  file.accept = ".safetensors";
  file.style.cssText = "flex:1;min-width:0;font-size:12px";
  file.onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try {
      log(
        `LoRA 읽는 중: ${f.name} (${(f.size / 2 ** 20).toFixed(0)}MB)`,
        "dim",
      );
      // 파싱은 Worker가 수행 — 메인은 ArrayBuffer만 보관 (메인 스레드 비차단)
      const buffer = await f.arrayBuffer();
      item.src = { buffer };
      item.fileName = f.name;
      markLoraDirty(); // 즉시 컴파일 안 함 — 대기
    } catch (err) {
      log("LoRA 로드 실패: " + err.message, "err");
      item.src = null;
    }
  };
  const wlabel = document.createElement("span");
  wlabel.style.cssText =
    "font-size:11px;color:var(--dim,#888);width:30px;text-align:right";
  wlabel.textContent = item.weight.toFixed(2);
  const w = document.createElement("input");
  w.type = "range";
  w.min = "0";
  w.max = "2";
  w.step = "0.05";
  w.value = item.weight;
  w.style.cssText = "width:90px";
  w.oninput = () => {
    item.weight = +w.value;
    wlabel.textContent = item.weight.toFixed(2);
  };
  w.onchange = () => {
    if (!item.src) return;
    // 단일 LoRA면 강도가 그래프 lora_scale로 가므로 재컴파일 불필요 — 다음 생성에 즉시 반영.
    // 멀티면 SVD 병합에 강도가 들어가므로 재반영 필요.
    if (lora.items.filter((i) => i.src).length >= 2) markLoraDirty();
  };
  const del = document.createElement("button");
  del.textContent = "×";
  del.style.cssText =
    "width:auto;padding:0 9px;font-size:16px;line-height:1;height:28px;" +
    "display:inline-flex;align-items:center;justify-content:center;flex:0 0 auto";
  del.onclick = () => {
    const had = !!item.src;
    lora.items = lora.items.filter((x) => x.id !== item.id);
    row.remove(); // 이 행만 제거 — 나머지 행은 그대로
    if (had) markLoraDirty();
  };
  row.append(file, w, wlabel, del);
  return row;
}

async function applyLora() {
  if (queueBusy()) {
    log(BUSY_MSG, "err");
    return;
  }
  const btn = $("btnLoraApply");
  btn.disabled = true;
  btn.textContent = "반영 중...";
  // 병합·컴파일 중 생성 차단 (오류 방지) — 하단·상단 생성 버튼 둘 다
  const genBtns = [$("btnGen"), $("btnGenTop")];
  for (const b of genBtns) b.disabled = true;
  $("loadBar").style.width = "0%";
  const n = lora.items.filter((i) => i.src).length;
  $("loraStat").textContent = n
    ? `${n}개 LoRA 병합·컴파일...`
    : "LoRA 해제·컴파일...";
  try {
    const r = await reloadDit(); // 한 번만 컴파일 (병합은 Worker)
    // 실제로 다시 올렸을 때만 '반영됨'. 건너뛴 경우 dirty를 남겨야 나중에 다시 반영할 수 있다
    // (예전엔 항상 dirty=false → 상주 모델엔 LoRA가 안 들어갔는데 반영 버튼까지 잠겼다).
    if (r === "ok") lora.dirty = false; // 상태 문구는 loadAll이 채운다
    else if (r === "not-loaded")
      $("loraStat").textContent = "선택한 모델이 아직 로드되지 않음 — [모델 로드] 때 함께 적용됩니다";
    else if (r === "failed") $("loraStat").textContent = "LoRA 반영 실패 (로그 참고)";
  } catch (e) {
    log("LoRA 반영 실패: " + e.message, "err");
    $("loraStat").textContent = "LoRA 반영 실패";
  } finally {
    btn.textContent = lora.dirty ? "LoRA 반영 ●" : "LoRA 반영";
    btn.disabled = !lora.dirty;
    // 생성 가능 = 드롭다운의 모델이 상주 중 (onModelChange와 같은 기준). 다른 모델이 상주해도 켜지 않는다.
    const ready = !!ditSessions[$("modelSel").value];
    for (const b of genBtns) b.disabled = !ready;
    $("loadBar").style.width = "0%";
  }
}

function addLoraRow() {
  const item = { id: ++loraSeq, src: null, fileName: "", weight: 1.0 };
  lora.items.push(item);
  $("loraList").appendChild(createLoraRow(item)); // 새 행만 추가
}
export function initLoraUI() {
  $("btnLoraAdd").addEventListener("click", addLoraRow);
  $("btnLoraApply").addEventListener("click", applyLora);
  addLoraRow(); // 기본 1줄
}

