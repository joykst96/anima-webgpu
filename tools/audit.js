// 이벤트 바인딩 감사: 페이지 스크립트보다 먼저 실행돼 addEventListener / on* 핸들러 등록을
// 순서대로 기록한다. 리팩터링 전후 window.__audit 목록이 같으면 UI 배선이 보존된 것.
// devserver가 ?audit 쿼리일 때 <head> 맨 앞에 주입한다.
(() => {
  const log = (window.__audit = []);
  const label = (t) =>
    t === window ? "window" : t === document ? "document"
      : t instanceof Element ? t.id ? `#${t.id}` : `${t.tagName.toLowerCase()}${t.className ? "." + String(t.className).split(" ")[0] : ""}`
      : t?.constructor?.name ?? "?";
  const origAdd = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function (type, fn, opts) {
    log.push(`add ${label(this)} ${type}${opts && opts.once ? " once" : ""}`);
    return origAdd.call(this, type, fn, opts);
  };
  for (const proto of [HTMLElement.prototype, Window.prototype, Document.prototype, Worker.prototype]) {
    for (const prop of ["onclick", "onchange", "oninput", "onkeydown", "onmessage", "onload", "onerror", "onmousedown"]) {
      const d = Object.getOwnPropertyDescriptor(proto, prop);
      if (!d || !d.set) continue;
      Object.defineProperty(proto, prop, {
        ...d,
        set(v) { log.push(`set ${label(this)} ${prop}`); d.set.call(this, v); },
      });
    }
  }
})();
