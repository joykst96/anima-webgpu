// 토크나이저 로드 (transformers.js, 로컬 우선 → HF 폴백)
import { log } from "./dom.js";

import {
  AutoTokenizer,
  env as hfEnv,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1"; // 고정 — 미지정 시 최신을 따라가 토큰화가 바뀔 수 있음 (2026-10-27 기준 최신 = 골든 생성 때 버전)
hfEnv.allowLocalModels = true;
hfEnv.localModelPath = "tokenizers/"; // prepare_tokenizers.py 산출물 위치

const TOK_FALLBACK = {
  qwen3: "Qwen/Qwen3-0.6B-Base",
  t5: "Xenova/flan-t5-small",
};
async function loadTokenizer(id) {
  try {
    return await AutoTokenizer.from_pretrained(id);
  } catch (e) {
    const fb = TOK_FALLBACK[id];
    if (!fb) throw e;
    log(`  (로컬 토크나이저 '${id}' 없음 → HF '${fb}' 폴백)`, "dim");
    return await AutoTokenizer.from_pretrained(fb);
  }
}

export { loadTokenizer };
