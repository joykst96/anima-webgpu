// 프롬프트 가중치 파싱 + 텍스트 인코딩(TE → 어댑터)
import { log } from "./dom.js";
import { CTX_DIM, CTX_LEN } from "./constants.js";
import { HAS_F16, disposeRun, disposeTensor, f16ToF32, f32ToF16, makeF16Tensor, ort } from "./ort.js";
import { sess, tok } from "./loader.js";

// ───────── 추론 ─────────
// ───────── 프롬프트 가중치 ─────────
// ComfyUI comfy/sd1_clip.py escape_important / parse_parentheses / token_weights 포팅 (규칙 그대로):
//   (x) = 바깥 가중치 ×1.1, (x:w) = w로 **덮어씀**(바깥 가중치와 곱하지 않음), \( \) = 리터럴 괄호,
//   짝이 안 맞는 괄호는 글자로 남는다. w는 Python float()처럼 앞뒤 공백·지수 표기 허용(inf/nan은 글자로 취급).
const escapeImportant = (t) => t.replaceAll("\\)", "\u0000\u0001").replaceAll("\\(", "\u0000\u0002");
const unescapeImportant = (t) => t.replaceAll("\u0000\u0001", ")").replaceAll("\u0000\u0002", "(");

function parseParentheses(s) {
  const result = [];
  let cur = "",
    level = 0;
  for (const ch of s) {
    if (ch === "(") {
      if (level === 0) {
        if (cur) result.push(cur);
        cur = "(";
      } else cur += ch;
      level++;
    } else if (ch === ")") {
      level--;
      if (level === 0) {
        result.push(cur + ")");
        cur = "";
      } else cur += ch;
    } else cur += ch;
  }
  if (cur) result.push(cur);
  return result;
}

const FLOAT_RE = /^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\s*$/;
function tokenWeights(s, current) {
  const out = [];
  for (let x of parseParentheses(s)) {
    let w = current;
    if (x.length >= 2 && x[0] === "(" && x[x.length - 1] === ")") {
      x = x.slice(1, -1);
      const xx = x.lastIndexOf(":");
      w *= 1.1;
      if (xx > 0 && FLOAT_RE.test(x.slice(xx + 1))) {
        w = parseFloat(x.slice(xx + 1));
        x = x.slice(0, xx);
      }
      out.push(...tokenWeights(x, w));
    } else out.push([x, current]);
  }
  return out;
}

// 반환: [{text, weight}] — ComfyUI가 토큰화하는 구간 단위 그대로 (Qwen·T5 모두 구간마다 따로 토큰화)
function parsePromptWeights(text) {
  return tokenWeights(escapeImportant(text), 1.0).map(([t, w]) => ({ text: unescapeImportant(t), weight: w }));
}

// NegPip 자동 병합: 긍정 + 부정 프롬프트를 음수 가중치로. 부정 프롬프트의 구간마다 가중치에 w(<0)를 곱해 명시한다
// — ComfyUI 규칙에선 "(부정:w)"로 감싸면 안쪽 (tag:1.2)가 +1.2로 덮어써지므로 구간별로 풀어 쓴다.
// 구간 분할은 "(부정:w)"로 감쌌을 때와 같아 토큰화도 같다.
const escapeParens = (t) => t.replaceAll("(", "\\(").replaceAll(")", "\\)");
function mergeNegPip(pos, neg, w) {
  const groups = parsePromptWeights(neg)
    .filter((s) => s.text)
    .map((s) => `(${escapeParens(s.text)}:${s.weight * w})`);
  return pos + ", " + groups.join("");
}

async function encodePrompt(text, negpipOK) {
  const segs = parsePromptWeights(text);

  // Qwen: ComfyUI처럼 구간마다 따로 토큰화해 이어 붙인다(가중치는 Qwen엔 쓰지 않음 — text_encoders/anima.py).
  // 문법을 지운 전체 문자열을 한 번에 토큰화하면 구간 경계의 BPE가 달라져 context가 5~12% 달라진다.
  const parts = segs.filter((s) => s.text).map((s) => tok.qwen(s.text).input_ids.data);
  const nIds = parts.reduce((n, p) => n + p.length, 0);
  let ids, mask;
  if (nIds) {
    ids = new BigInt64Array(nIds);
    let o = 0;
    for (const p of parts) {
      ids.set(p, o);
      o += p.length;
    }
    mask = new BigInt64Array(nIds).fill(1n);
  } else {
    const q = tok.qwen(""); // 빈 프롬프트는 종전대로
    ids = q.input_ids.data;
    mask = q.attention_mask.data;
  }
  const teIdsT = new ort.Tensor("int64", ids, [1, ids.length]);
  const teMaskT = new ort.Tensor("int64", mask, [1, mask.length]);
  const teOut = await sess.te.run({
    input_ids: teIdsT,
    attention_mask: teMaskT,
  });
  const hidden = teOut[Object.keys(teOut)[0]]; // fp16 (1,S,1024)

  // T5: 세그먼트별 토크나이즈 → 토큰 단위 가중치 벡터 (EOS는 마지막에 1회)
  const eosIds = Array.from(tok.t5("").input_ids.data);
  const EOS_ID = eosIds.length ? eosIds[eosIds.length - 1] : null;
  const t5arr = [],
    tw = [];
  for (const sgm of segs) {
    if (!sgm.text) continue;
    const enc = Array.from(tok.t5(sgm.text).input_ids.data);
    if (EOS_ID !== null && enc.length && enc[enc.length - 1] === EOS_ID)
      enc.pop();
    for (const id of enc) {
      t5arr.push(id);
      tw.push(sgm.weight);
    }
  }
  for (const id of eosIds) {
    t5arr.push(id);
    tw.push(1);
  }
  if (t5arr.length > CTX_LEN) {
    t5arr.length = CTX_LEN;
    tw.length = CTX_LEN;
  }
  const t5ids = BigInt64Array.from(t5arr);

  const adT5T = new ort.Tensor("int64", t5ids, [1, t5ids.length]);
  const adOut = await sess.adapter.run({
    qwen_hidden_states: hidden,
    t5_input_ids: adT5T,
  });
  const ctx = adOut[Object.keys(adOut)[0]]; // fp16 (1,S,1024)

  // 가중치 적용 — ComfyUI Anima 동일 지점: out = out * t5xxl_weights (512 패딩 전)
  let cdata = ctx.data;
  let hasW = false,
    hasNeg = false;
  for (const w of tw) {
    if (w !== 1) hasW = true;
    if (w < 0) hasNeg = true;
  }
  if (hasNeg && !negpipOK)
    log("이 모델엔 NegPip 입력이 없음 — 음수 가중치를 0으로 처리", "dim");
  if (hasW) {
    const S = Math.min(tw.length, (cdata.length / CTX_DIM) | 0);
    const f = HAS_F16 ? cdata : f16ToF32(cdata); // 사실 3: u16 비트패턴 분기
    for (let t = 0; t < S; t++) {
      // NegPip: 크기 |w|는 context에, 부호는 negpip_mask로 분리 (참조 구현과 동일)
      let w = negpipOK ? Math.abs(tw[t]) : Math.max(0, tw[t]);
      if (w === 1) continue;
      const off = t * CTX_DIM;
      for (let d = 0; d < CTX_DIM; d++) f[off + d] *= w;
    }
    cdata = HAS_F16 ? f : f32ToF16(f);
  }

  const padded = HAS_F16
    ? new Float16Array(CTX_LEN * CTX_DIM)
    : new Uint16Array(CTX_LEN * CTX_DIM); // zero-pad → 512
  padded.set(cdata.subarray(0, Math.min(cdata.length, padded.length)));

  // cdata 복사 완료 → TE/adapter 중간 텐서(입력·출력) 전부 반환.
  // 반환되는 ctx는 아래에서 padded로 새로 만드는 별개 텐서이므로 여기서 원본을 놓아도 안전.
  disposeTensor(teIdsT);
  disposeTensor(teMaskT);
  disposeTensor(adT5T);
  disposeRun(teOut); // hidden 포함
  disposeRun(adOut); // 원본 ctx 포함

  // negpip 마스크: 토큰별 부호(±1), 패딩 구간은 1
  let npMask = null;
  if (negpipOK) {
    const mf = new Float32Array(CTX_LEN).fill(1);
    for (let t = 0; t < Math.min(tw.length, CTX_LEN); t++)
      if (tw[t] < 0) mf[t] = -1;
    npMask = makeF16Tensor(mf, [1, CTX_LEN, 1]);
  }
  return {
    ctx: new ort.Tensor("float16", padded, [1, CTX_LEN, CTX_DIM]),
    mask: npMask,
    len: Math.min(tw.length, CTX_LEN), // 실제 토큰 수 (나머지는 0 패딩) — cross-attention 패딩 압축용
  };
}

export { encodePrompt, mergeNegPip, parsePromptWeights };
