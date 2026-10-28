// 버그 6 시나리오: 드롭다운이 '로드 안 된 모델'을 가리킬 때 LoRA 반영.
//   A) 모델 로드 → 드롭다운을 다른(미로드) 모델로 → LoRA 추가 → 반영 → 드롭다운을 원래 모델로 → 반영
//   B) (새 페이지에서) 모델 미로드 상태로 LoRA 추가 → 반영 → 모델 로드
// 사용: const s = await import("/__tools/scenario-lora-unloaded.mjs"); window.__s6 = s.runA();  // 또는 runB()
import { makeSyntheticLora } from "./harness.mjs";

const $ = (id) => document.getElementById(id);
const tick = () => new Promise((r) => { const c = new MessageChannel(); c.port1.onmessage = r; c.port2.postMessage(0); });
async function until(pred, ms) { const t0 = performance.now(); while (!pred()) { if (performance.now() - t0 > ms) return false; await tick(); } return true; }
const count = (re) => ($("log").innerText.match(re) || []).length;
const snap = () => ({
  stat: $("loraStat").textContent, applyBtn: $("btnLoraApply").textContent, applyDisabled: $("btnLoraApply").disabled,
  genDisabled: $("btnGen").disabled, genTopDisabled: $("btnGenTop").disabled, model: $("modelSel").value,
});
function select(key) { $("modelSel").value = key; $("modelSel").dispatchEvent(new Event("change")); }
async function addLora() {
  const f = await makeSyntheticLora({ seed: 1, rank: 4, dtype: "F16", name: "synthA.safetensors" });
  for (const r of [...$("loraList").children]) r.querySelector("button").click();
  $("btnLoraAdd").click();
  const i = $("loraList").lastElementChild.querySelector('input[type="file"]'), dt = new DataTransfer();
  dt.items.add(f); i.files = dt.files; i.dispatchEvent(new Event("change", { bubbles: true }));
  await until(() => /변경됨/.test($("loraStat").textContent), 5000);
}
async function apply() {
  const c0 = count(/컴파일 \(셰이더 빌드\)/g);
  $("btnLoraApply").click();
  await until(() => $("btnLoraApply").textContent !== "반영 중...", 60000);
  return count(/컴파일 \(셰이더 빌드\)/g) - c0; // 실제 DiT 재컴파일 횟수
}
async function load() {
  const n0 = count(/로드 완료|로드 실패/g);
  $("btnLoad").click();
  await until(() => count(/로드 완료|로드 실패/g) > n0, 60000);
}

function wrap(fn) {
  const st = { status: "running", steps: [] };
  fn(st).then(() => { st.status = "done"; }, (e) => { st.status = "error"; st.error = String(e.stack || e); });
  return st;
}

export const runA = () => wrap(async (st) => {
  await until(() => /WebGPU OK/.test($("gpuInfo").innerText), 10000);
  await load();
  const loaded = $("modelSel").value;
  // 로드 안 된 아무 서버 모델 (배포 목록에서 지금 것이 아닌 첫 항목)
  select([...$("modelSel").options].map((o) => o.value).find((v) => v !== loaded && v !== "custom"));
  await addLora();
  st.steps.push({ step: "미로드 모델 선택 + LoRA 추가", ...snap() });
  const recompiles1 = await apply();
  st.steps.push({ step: "반영 (미로드 모델 선택 중)", recompiles: recompiles1, ...snap() });
  select(loaded);
  st.steps.push({ step: "원래 모델로 되돌림", ...snap() });
  const recompiles2 = await apply().catch(() => -1);
  st.steps.push({ step: "반영 (로드된 모델)", recompiles: recompiles2, ...snap() });
});

export const runB = () => wrap(async (st) => {
  await until(() => /WebGPU OK/.test($("gpuInfo").innerText), 10000);
  await addLora();
  const recompiles = await apply();
  st.steps.push({ step: "모델 미로드 상태 반영", recompiles, ...snap() });
  await load();
  st.steps.push({ step: "모델 로드", ...snap() });
});
