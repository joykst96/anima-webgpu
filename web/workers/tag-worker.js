// 태그 자동완성 워커: danbooru_ko.csv 파싱 + 한글 bigram 역인덱스 빌드 (main.js에서 1회 생성)
// tag,0,pc,"설명…키워드:<카테고리>,alias…"  — desc는 따옴표 포함, 내부 콤마 있음
function parseLine(line) {
  const m = line.match(/^([^,]*),([^,]*),([^,]*),([\s\S]*)$/);
  if (!m) return null;
  let desc = m[4].trim();
  if (desc.startsWith('"') && desc.endsWith('"')) desc = desc.slice(1, -1);
  const pc = parseInt(m[3], 10);
  if (!m[1] || isNaN(pc)) return null;
  return { tag: m[1], pc, desc };
}
// 한글 alias 추출(표시용): "키워드:" 뒤 콤마 분리 중 <카테고리> 제외
function extractAliases(desc) {
  const i = desc.indexOf('키워드:');
  if (i < 0) return [];
  return desc.slice(i + 4).split(',').map(s => s.trim())
    .filter(s => s && s[0] !== '<');
}
// 카테고리 추출: 맨 앞 [카테고리] 또는 [카테고리 > 하위] (없을 수 있음)
function extractCategory(desc) {
  const mb = desc.match(/^\s*\[([^\]]*)\]/);
  return mb ? mb[1].trim() : null;
}
self.onmessage = async (ev) => {
  try {
    const { url } = ev.data;
    const res = await fetch(url);
    if (!res.ok) throw new Error('CSV HTTP ' + res.status);
    const text = await res.text();
    self.postMessage({ type: 'progress', stage: '파싱' });
    const lines = text.split('\n');
    const tags = [];      // { tag, pc, desc, alias, cat }
    for (let i = 0; i < lines.length; i++) {
      const ln = lines[i];
      if (!ln.trim()) continue;
      // 헤더 행 스킵: "name,category,post_count,description"
      if (i === 0 && /^name\s*,/.test(ln)) continue;
      const r = parseLine(ln);
      if (!r) continue;
      tags.push({ tag: r.tag, pc: r.pc, desc: r.desc,
                  alias: extractAliases(r.desc).join(', '),
                  cat: extractCategory(r.desc),
                  // 검색용 정규화: 언더스코어→공백, 소문자 (Anima/ILXL 표기 기준)
                  tagNorm: r.tag.toLowerCase().replace(/_/g, ' ') });
    }
    // post_count 내림차순 (검색은 이 순서대로 훑어 상위부터 채움)
    tags.sort((a, b) => b.pc - a.pc);
    self.postMessage({ type: 'progress', stage: '인덱스' });
    // 한글 bigram 역인덱스: desc의 연속 2글자(공백 미포함) → 정렬 후 행 인덱스
    // 검색 시 쿼리 앞 2글자로 후보를 좁힌 뒤 desc.includes로 확정.
    const bi = new Map();
    for (let i = 0; i < tags.length; i++) {
      const s = tags[i].desc;
      const seen = new Set();
      for (let p = 0; p < s.length - 1; p++) {
        const a = s[p], b = s[p + 1];
        if (a === ' ' || b === ' ') continue;
        const bg = a + b;
        if (seen.has(bg)) continue;
        seen.add(bg);
        let arr = bi.get(bg);
        if (!arr) { arr = []; bi.set(bg, arr); }
        arr.push(i);
      }
    }
    // 직렬화: bigram Map → 평면 배열(구조화복제로 IndexedDB 저장 가능)
    const bigramKeys = [];
    const bigramVals = [];
    for (const [k, v] of bi) { bigramKeys.push(k); bigramVals.push(v); }
    self.postMessage(
      { type: 'done', tags, bigramKeys, bigramVals },
      // tags/bigramVals는 복제됨(transfer 불가능한 일반객체) — 그대로 전송
    );
  } catch (e) {
    self.postMessage({ type: 'error', message: e.message });
  }
};
