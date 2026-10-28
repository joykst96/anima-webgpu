// DOM 헬퍼와 화면 로그

const $ = (id) => document.getElementById(id);
// 줄을 덧붙인다. innerHTML += 는 매번 로그 전체를 다시 파싱하고 파일명 등을 HTML로 해석하므로 텍스트 노드로 추가.
const log = (msg, cls = "") => {
  const el = $("log");
  const line = document.createElement("span");
  if (cls) line.className = cls;
  line.textContent = msg;
  el.append(line, "\n");
  el.scrollTop = el.scrollHeight;
};

export { $, log };
