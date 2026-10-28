// 해상도 프리셋 칩 + SPEED 모드 토글
import { $ } from "./dom.js";
import { updateModelUI } from "./model-ui.js";

// ───────── 해상도 프리셋 (칩 그리드: 버킷 칩 → 비율 칩) ─────────
// 표 기준. 장변 1664까지(1280버킷 16:9). [가로, 세로, 라벨]. 자유 입력은 2048까지 허용(VRAM 의존).
const RES_BUCKETS = {
  512: [
    [512, 512, "1:1"],
    [384, 576, "2:3"],
    [576, 384, "3:2"],
    [384, 704, "9:16"],
    [704, 384, "16:9"],
  ],
  768: [
    [768, 768, "1:1"],
    [640, 960, "2:3"],
    [960, 640, "3:2"],
    [576, 1024, "9:16"],
    [1024, 576, "16:9"],
  ],
  1024: [
    [1024, 1024, "1:1"],
    [832, 1216, "2:3"],
    [1216, 832, "3:2"],
    [768, 1344, "9:16"],
    [1344, 768, "16:9"],
  ],
  1280: [
    [1280, 1280, "1:1"],
    [1024, 1536, "2:3"],
    [1536, 1024, "3:2"],
    [960, 1664, "9:16"],
    [1664, 960, "16:9"],
  ],
};
let curBucket = "1024";

// 비율 아이콘 placeholder — 비율 모양의 빈 박스 (나중에 실제 아이콘/이미지로 교체).
function ratioIcon(w, h) {
  const span = document.createElement("span");
  span.className = "ratio-ico";
  const base = 13;
  if (w >= h) {
    span.style.width = base + "px";
    span.style.height = Math.round((base * h) / w) + "px";
  } else {
    span.style.height = base + "px";
    span.style.width = Math.round((base * w) / h) + "px";
  }
  return span;
}

function renderRatioChips(bucket) {
  const box = $("resRatioChips");
  box.innerHTML = "";
  for (const [w, h, label] of RES_BUCKETS[bucket] || []) {
    const b = document.createElement("button");
    b.className = "chip";
    b.dataset.wh = `${w}x${h}`;
    b.appendChild(ratioIcon(w, h));
    const t = document.createElement("span");
    t.textContent = label;
    b.appendChild(t);
    b.title = `${w}×${h}`; // 실제 해상도는 툴팁으로
    b.onclick = () => {
      $("width").value = w;
      $("height").value = h;
      [...box.children].forEach((c) =>
        c.classList.toggle("active", c === b),
      );
    };
    box.appendChild(b);
  }
}

function renderBucketChips() {
  const box = $("resBucketChips");
  box.innerHTML = "";
  for (const bk of Object.keys(RES_BUCKETS)) {
    const b = document.createElement("button");
    b.className = "chip" + (bk === curBucket ? " active" : "");
    b.textContent = bk;
    b.onclick = () => {
      curBucket = bk;
      [...box.children].forEach((c) =>
        c.classList.toggle("active", c.textContent === bk),
      );
      renderRatioChips(bk);
      // 버킷 선택 시 정사각(첫 항목) 자동 적용
      const [w, h] = RES_BUCKETS[bk][0];
      $("width").value = w;
      $("height").value = h;
      markActiveRatio();
    };
    box.appendChild(b);
  }
}

// 현재 width/height에 맞는 비율 칩을 active 표시 (없으면 모두 해제)
function markActiveRatio() {
  const w = +$("width").value,
    h = +$("height").value;
  [...$("resRatioChips").children].forEach((c) => {
    const [cw, ch] = c.dataset.wh.split("x").map(Number);
    c.classList.toggle("active", cw === w && ch === h);
  });
}
// ── SPEED 모드 토글 ──────────────────────────────────────────────
// 켜면: 1024/1280 버킷만 허용(나머지 잠금), 1024·1:1 초기화,
//   가로/세로 수동입력·부정프롬프트·negpip 비활성(시각 dim), 개발패널 노출.
// 끄면: 전부 원복.
const SPEED_ALLOWED_BUCKETS = ["1024", "1280"];

function applySpeedMode(on) {
  const bucketBox = $("resBucketChips");
  $("speedDevPanel").style.display = on ? "" : "none";

  // 버킷칩 잠금/허용 표시
  [...bucketBox.children].forEach((c) => {
    const allowed = SPEED_ALLOWED_BUCKETS.includes(c.textContent);
    c.classList.toggle("chip-locked", on && !allowed);
    c.classList.toggle("speed-allowed", on && allowed);
  });

  // 수동 입력만 비활성/원복 (부정프롬프트·negpip은 SPEED와 무관)
  const lockTargets = [
    $("width").closest("div"),
    $("height").closest("div"),
  ];
  for (const el of lockTargets) {
    if (!el) continue;
    el.classList.toggle("speed-disabled", on);
  }
  $("width").disabled = on;
  $("height").disabled = on;

  if (on) {
    // 1024 / 1:1로 초기화
    if (curBucket !== "1024") {
      curBucket = "1024";
      [...bucketBox.children].forEach((c) =>
        c.classList.toggle("active", c.textContent === "1024"),
      );
      renderRatioChips("1024");
    }
    const [w, h] = RES_BUCKETS["1024"][0]; // 1:1
    $("width").value = w;
    $("height").value = h;
    markActiveRatio();
  } else {
    // 원복: CFG 기반 negative/negpip 상태를 다시 적용
    updateModelUI();
  }
}

export function initPresets() {
  $("width").addEventListener("input", markActiveRatio);
  $("height").addEventListener("input", markActiveRatio);

  renderBucketChips();
  renderRatioChips(curBucket);
  markActiveRatio();
  $("speedChk").addEventListener("change", (e) =>
    applySpeedMode(e.target.checked),
  );
}

