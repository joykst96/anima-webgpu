// 버그 1 시나리오: 생성 2장을 큐에 넣고, 1장째 도중 LoRA 반영을 누른다.
//   수정 전: 실행 중 DiT 세션이 release() → 생성 실패, 남은 큐도 실패.
//   수정 후: 반영이 거부되고(로그) 2장 모두 완료, 큐가 끝나면 버튼이 다시 켜진다.
// 사용: (모델 로드 후) const s = await import("/__tools/scenario-busy.mjs"); window.__busy = s.run();
import { makeSyntheticLora } from "./harness.mjs";

const $ = (id) => document.getElementById(id);
const tick = () => new Promise((r) => { const c = new MessageChannel(); c.port1.onmessage = r; c.port2.postMessage(0); });
async function until(pred, ms) { const t0 = performance.now(); while (!pred()) { if (performance.now() - t0 > ms) return false; await tick(); } return true; }
const count = (re) => ($("log").innerText.match(re) || []).length;

export function run() {
  const st = { status: "running", obs: {} };
  (async () => {
    try {
      const done0 = count(/완료 —/g), fail0 = count(/생성 실패|건너뜀/g), busy0 = count(/생성 중에는 모델 로드/g);
      $("seed").value = 4242; $("batch").value = 2;
      $("btnGen").click();
      await until(() => /step 2\//.test($("genStat").textContent), 30000); // 1장째 진행 중
      st.obs.duringLocks = { load: $("btnLoad").disabled, loraApply: $("btnLoraApply").disabled };
      // 생성 중 LoRA 파일을 넣어 dirty로 만든다 → 반영 버튼이 켜지는지
      const file = await makeSyntheticLora({ seed: 1, rank: 4, dtype: "F16", name: "synthA.safetensors" });
      for (const row of [...$("loraList").children]) row.querySelector("button").click();
      $("btnLoraAdd").click();
      const input = $("loraList").lastElementChild.querySelector('input[type="file"]'), dt = new DataTransfer();
      dt.items.add(file); input.files = dt.files; input.dispatchEvent(new Event("change", { bubbles: true }));
      await until(() => /변경됨/.test($("loraStat").textContent), 5000);
      st.obs.applyEnabledWhileBusy = !$("btnLoraApply").disabled;
      // 버튼이 켜져 있다고 가정하고(수정 전 동작) 강제로 클릭
      $("btnLoraApply").disabled = false;
      $("btnLoraApply").click();
      await until(() => !$("queueList").children.length && !/반영 중/.test($("btnLoraApply").textContent), 120000);
      st.obs.completed = count(/완료 —/g) - done0;
      st.obs.failed = count(/생성 실패|건너뜀/g) - fail0;
      st.obs.busyRejected = count(/생성 중에는 모델 로드/g) - busy0;
      st.obs.afterLocks = { load: $("btnLoad").disabled, loraApply: $("btnLoraApply").disabled };
      st.obs.logTail = $("log").innerText.split("\n").filter(Boolean).slice(-8);
      st.status = "done";
    } catch (e) { st.status = "error"; st.error = String(e.stack || e); }
  })();
  return st;
}
