// 태그 자동완성 (Danbooru + 한글 alias, 전부 로컬)
import { $, log } from "./dom.js";

// ═════════ 태그 자동완성 (Danbooru + 한글 alias, 전부 로컬) ═════════
// 데이터: out/danbooru_ko.csv (205k행). 프롬프트는 서버로 안 나감 — 검색은
// 전부 브라우저 메모리에서 수행(이 페이지의 프롬프트 미전송 원칙 유지).
// 파싱/인덱스 빌드는 Worker에서 1회 → IndexedDB 직렬화 → 재방문 시 즉시 로드.
//   CSV 컬럼: [0]tag [1]무시(0) [2]post_count [3]"설명…키워드:<카테고리>,한글alias…"
//   영어: prefix 매칭. 한글: 2글자↑ bigram 역인덱스 → desc substring. pc 내림차순.
const AC_CSV_URL = "out/danbooru_ko.csv";
const AC_DB_NAME = "anima-tagac";
const AC_DB_VER = 3; // 스키마 변경 시 올려서 캐시 무효화 (v3: type2 제거, tagNorm 추가, 신규 CSV)
const AC_STORE = "index";

// ── Worker: CSV 파싱 + 인덱스 빌드 (메인 비차단) ──
// 워커 소스: web/workers/tag-worker.js

// ── IndexedDB: 빌드 결과 저장/로드 (재방문 시 파싱 스킵) ──
function acIdbOpen() {
  return new Promise((resolve, reject) => {
    const rq = indexedDB.open(AC_DB_NAME, AC_DB_VER);
    rq.onupgradeneeded = () => {
      const db = rq.result;
      // 버전 업 = 스키마 변경. 기존 store가 있으면 비우고 새로 만든다(옛 데이터 무효화).
      if (db.objectStoreNames.contains(AC_STORE))
        db.deleteObjectStore(AC_STORE);
      db.createObjectStore(AC_STORE);
    };
    rq.onsuccess = () => resolve(rq.result);
    rq.onerror = () => reject(rq.error);
  });
}
async function acIdbGet(key) {
  try {
    const db = await acIdbOpen();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(AC_STORE, "readonly");
      const rq = tx.objectStore(AC_STORE).get(key);
      rq.onsuccess = () => resolve(rq.result || null);
      rq.onerror = () => reject(rq.error);
    });
  } catch {
    return null;
  }
}
async function acIdbSet(key, val) {
  try {
    const db = await acIdbOpen();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(AC_STORE, "readwrite");
      tx.objectStore(AC_STORE).put(val, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch (e) {
    console.warn("자동완성 캐시 저장 실패:", e);
  }
}

// ── 자동완성 데이터 상태 ──
const ac = {
  ready: false,
  loading: false,
  tags: null, // [{tag, pc, desc, alias}]  (pc 내림차순)
  bigram: null, // Map("가슴" -> [rowIdx...])
};

async function acEnsureLoaded() {
  if (ac.ready || ac.loading) return;
  ac.loading = true;
  try {
    // 1) IndexedDB 캐시 우선
    const cached = await acIdbGet("data");
    if (cached && cached.tags && cached.bigramKeys) {
      ac.tags = cached.tags;
      ac.bigram = new Map();
      for (let i = 0; i < cached.bigramKeys.length; i++)
        ac.bigram.set(cached.bigramKeys[i], cached.bigramVals[i]);
      ac.ready = true;
      ac.loading = false;
      log(`태그 자동완성 준비됨 (캐시 ${ac.tags.length}개)`, "ok");
      return;
    }
    // 2) Worker로 CSV 파싱 + 인덱스 빌드
    log("태그 자동완성 데이터 빌드 중... (최초 1회)", "dim");
    const wk = new Worker(new URL("../workers/tag-worker.js", import.meta.url), {
      type: "module",
    });
    const result = await new Promise((resolve, reject) => {
      wk.onmessage = (ev) => {
        const m = ev.data;
        if (m.type === "progress")
          $("loadStat").textContent = `자동완성 ${m.stage}...`;
        else if (m.type === "done") resolve(m);
        else if (m.type === "error") reject(new Error(m.message));
      };
      wk.onerror = (e) => reject(new Error(e.message || "worker 오류"));
      // Worker는 blob: URL에서 실행돼 상대경로 base가 없으므로 절대 URL로 변환해 전달
      wk.postMessage({ url: new URL(AC_CSV_URL, location.href).href });
    });
    wk.terminate();
    ac.tags = result.tags;
    ac.bigram = new Map();
    for (let i = 0; i < result.bigramKeys.length; i++)
      ac.bigram.set(result.bigramKeys[i], result.bigramVals[i]);
    ac.ready = true;
    log(`태그 자동완성 준비됨 (${ac.tags.length}개)`, "ok");
    // 3) IndexedDB에 직렬화 저장 (다음 방문 즉시)
    acIdbSet("data", {
      tags: result.tags,
      bigramKeys: result.bigramKeys,
      bigramVals: result.bigramVals,
    });
  } catch (e) {
    log(`태그 자동완성 로드 실패: ${e.message}`, "err");
  } finally {
    ac.loading = false;
  }
}

// ── 검색 ──
const AC_LIMIT = 50; // 후보 최대(드롭다운은 스크롤). 너무 많으면 렌더 부담.
const HANGUL_RE = /[\uac00-\ud7a3]/; // 완성형 한글 음절
function acSearch(qRaw) {
  if (!ac.ready) return [];
  const q = qRaw.trim();
  if (!q) return [];
  const out = [];
  const isHangul = HANGUL_RE.test(q);
  if (isHangul) {
    if (q.length < 2) return []; // 한글 1글자 스킵(노이즈+느림)
    // 인덱스는 공백이 낀 bigram을 만들지 않으므로("긴 머리"의 "긴 ") 공백 없는 첫 bigram을 키로 쓴다.
    // q가 포함된 desc는 q의 모든 bigram을 포함하므로 어느 bigram으로 좁혀도 결과(순서 포함)는 같다.
    let key = null;
    for (let p = 0; p + 1 < q.length; p++)
      if (q[p] !== " " && q[p + 1] !== " ") { key = q.slice(p, p + 2); break; }
    if (!key) return [];
    const cand = ac.bigram.get(key);
    if (!cand) return [];
    for (let i = 0; i < cand.length; i++) {
      const r = ac.tags[cand[i]];
      if (r.desc.includes(q)) {
        out.push(r);
        if (out.length >= AC_LIMIT) break;
      }
    }
  } else {
    // 영어 prefix. 데이터 태그는 언더스코어(long_hair), 사용자는 _ 또는 공백
    // 둘 중 무엇으로 쳐도 매칭되게 양쪽 공백 정규화한 tagNorm으로 비교.
    const ql = q.toLowerCase().replace(/_/g, " ");
    for (let i = 0; i < ac.tags.length; i++) {
      if (ac.tags[i].tagNorm.startsWith(ql)) {
        out.push(ac.tags[i]);
        if (out.length >= AC_LIMIT) break;
      }
    }
  }
  return out;
}

// post_count 약식 표기: 6577634 → 6.6M, 1739697 → 1.7M, 12345 → 12.3k
function acFmtCount(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}
function acEscape(s) {
  return s.replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
  );
}
// 매칭 부분 강조(<mark>). q는 이미 검증된 검색어.
function acHighlight(text, q) {
  const esc = acEscape(text);
  if (!q) return esc;
  const idx = text.toLowerCase().indexOf(q.toLowerCase());
  if (idx < 0) return esc;
  // 원문 인덱스 기준으로 자르되 출력은 이스케이프
  return (
    acEscape(text.slice(0, idx)) +
    "<mark>" +
    acEscape(text.slice(idx, idx + q.length)) +
    "</mark>" +
    acEscape(text.slice(idx + q.length))
  );
}

// ── 토큰 추출: 콤마 구분 + 가중치 (tag:1.2) 안의 태그부만 ──
// 커서 위치 기준 현재 토큰의 [시작, 끝, 검색어] 반환.
function acCurrentToken(text, caret) {
  // 콤마로 토큰 경계. 커서가 속한 구간을 찾음.
  let start = text.lastIndexOf(",", caret - 1) + 1;
  let end = text.indexOf(",", caret);
  if (end < 0) end = text.length;
  let seg = text.slice(start, end);
  // 가중치/괄호 문법 제거하여 검색어만: 앞쪽 '(' 들, 뒤쪽 ':1.2)' 들
  // 토큰 내부에서 실제 단어 부분의 오프셋을 추적.
  let inner = seg;
  let lead = 0;
  // 선행 공백/여는 괄호 제거
  const mLead = inner.match(/^[\s(]*/);
  if (mLead) {
    lead = mLead[0].length;
    inner = inner.slice(lead);
  }
  // 후행 ':숫자' 가중치, 닫는 괄호, 공백 제거
  const mTail = inner.match(/(:[+-]?\d*\.?\d*)?[\s)]*$/);
  let tail = 0;
  if (mTail && mTail[0]) {
    tail = mTail[0].length;
    inner = inner.slice(0, inner.length - tail);
  }
  const wordStart = start + lead;
  const wordEnd = wordStart + inner.length;
  return {
    tokenStart: start,
    tokenEnd: end,
    wordStart,
    wordEnd,
    query: inner,
  };
}

// ── 드롭다운 UI (긍정/부정 각각 인스턴스) ──
function makeAutocomplete(taId, popId) {
  const ta = $(taId);
  const pop = $(popId);
  let items = []; // 현재 후보 [{tag,pc,desc,alias}]
  let active = -1; // 키보드 선택 인덱스
  let token = null; // 현재 토큰 위치
  let debounce = 0;

  function close() {
    pop.innerHTML = "";
    items = [];
    active = -1;
    token = null;
  }
  function render() {
    if (!items.length) {
      pop.innerHTML = "";
      return;
    }
    const q = token ? token.query : "";
    const html = items
      .map((r, i) => {
        const aliasHtml = r.alias
          ? `<span class="ac-alias">${acHighlight(r.alias, q)}</span>`
          : "";
        const catHtml = r.cat
          ? `<span class="ac-badge ac-badge-cat">${acEscape(r.cat)}</span>`
          : "";
        return (
          `<div class="ac-item${i === active ? " active" : ""}" data-i="${i}">` +
          `<div class="ac-row1">` +
          `<span class="ac-tag">${acHighlight(r.tag.replace(/_/g, " "), q)}</span>` +
          `<span class="ac-count">${acFmtCount(r.pc)}</span>` +
          `</div>` +
          `<div class="ac-row2">` +
          catHtml +
          aliasHtml +
          `</div>` +
          `</div>`
        );
      })
      .join("");
    pop.innerHTML = html;
    // 클릭 선택
    [...pop.children].forEach((el) => {
      el.onmousedown = (e) => {
        e.preventDefault(); // textarea 포커스 유지
        choose(+el.dataset.i);
      };
    });
    scrollActiveIntoView();
  }
  function scrollActiveIntoView() {
    if (active < 0) return;
    const el = pop.children[active];
    if (el) el.scrollIntoView({ block: "nearest" });
  }
  // 선택 → 영어 태그를 현재 토큰 단어부에 치환. 콤마/공백 정리.
  function choose(i) {
    const r = items[i];
    if (!r || !token) return close();
    const before = ta.value.slice(0, token.wordStart);
    const after = ta.value.slice(token.wordEnd);
    // 삽입 태그: 언더스코어→공백 (Anima/IllustriousXL 표기 규약)
    const tagOut = r.tag.replace(/_/g, " ");
    // 삽입 태그 뒤에 ", " 자동(이미 콤마가 뒤따르면 중복 안 함)
    const trimmedAfter = after.replace(/^\s*/, "");
    const needsComma = !trimmedAfter.startsWith(",");
    const insert = tagOut + (needsComma ? ", " : "");
    const newCaret = before.length + insert.length;
    ta.value = before + insert + after;
    ta.setSelectionRange(newCaret, newCaret);
    ta.focus();
    close();
  }
  function update() {
    if (!ac.ready) {
      acEnsureLoaded();
      return;
    }
    const caret = ta.selectionStart;
    token = acCurrentToken(ta.value, caret);
    const q = token.query;
    if (!q || q.length < 1) {
      close();
      return;
    }
    items = acSearch(q);
    active = items.length ? 0 : -1;
    render();
  }

  ta.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(update, 150); // 타이핑 멈춘 뒤 검색
  });
  ta.addEventListener("keydown", (e) => {
    if (!items.length) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      active = (active + 1) % items.length;
      render();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      active = (active - 1 + items.length) % items.length;
      render();
    } else if (e.key === "Enter" || e.key === "Tab") {
      if (active >= 0) {
        e.preventDefault();
        choose(active);
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      close();
    }
  });
  // 포커스 잃으면 닫기 (mousedown으로 선택은 이미 처리됨)
  ta.addEventListener("blur", () => setTimeout(close, 120));
  // 커서만 이동(클릭 등)해도 토큰 갱신
  ta.addEventListener("click", () => {
    clearTimeout(debounce);
    debounce = setTimeout(update, 150);
  });
}
export function initAutocomplete() {
  makeAutocomplete("prompt", "acPopPrompt");
  makeAutocomplete("negative", "acPopNegative");
  // 첫 포커스 시 데이터 로드 시작(페이지 로드를 막지 않음 — 지연 로드)
  $("prompt").addEventListener("focus", acEnsureLoaded, { once: true });
  $("negative").addEventListener("focus", acEnsureLoaded, { once: true });
}

export { acEscape };
