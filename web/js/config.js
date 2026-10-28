// 배포 구성: 경로·모델 목록·화면 기본값. 배포별로 다른 것은 이 파일뿐이다.
// 마커 사이는 export/gen_models.py가 model-list*.yml로 생성 — 직접 수정 금지.
//   python export/gen_models.py export/model-list.yml web/js/config.js

// PATH_DEFAULTS:START (gen_models.py 자동생성 — 직접 수정 금지)
const PATHS = {
  te: "out/text_encoder/qwen3_06b_s.onnx",
  adapter: "out/adapter/anima_llm_adapter.onnx",
  vae: "out/vae/qwen_image_vae_decoder_dyn32_2dc.onnx",
};
// PATH_DEFAULTS:END
const MODELS = {
  // MODELS:START (gen_models.py 자동생성 — 직접 수정 금지)
  anima_base: {
    label: "Anima (베이스 · 30스텝 · CFG 5)",
    dit: "out/dit/anima_dit_dyn32_q8a4fns.onnx",
    ditLora: "out/dit/anima_dit_dyn32_q8a4fnLs.onnx",
    steps: 30,
    cfg: 5.0,
    sampler: "er_sde",
    scheduler: "simple",
  },
  anima_turbo: {
    label: "Anima+Turbo v0.2 (8스텝 · CFG 1)",
    dit: "out/dit/anima_dit_turbo32_q8a4fns.onnx",
    ditLora: "out/dit/anima_dit_turbo32_q8a4fnLs.onnx",
    steps: 8,
    cfg: 1.0,
    sampler: "euler",
    scheduler: "simple",
  },
  // MODELS:END

  custom: {
    label: "사용자 설정 (직접 모델 지정)",
    dit: null,
    ditLora: null,
    steps: 8,
    cfg: 1.0,
    sampler: "euler",
    scheduler: "simple",
  },
};

// SITE_DEFAULTS:START (gen_models.py 자동생성 — 직접 수정 금지)
const SELECTED_MODEL = "anima_turbo";
const PROMPT_DEFAULTS = {
  positive: "masterpiece, best quality, score_7, safe",
  negative: "worst quality, low quality, score_1, score_2, score_3, artist name, nsfw",
};
const MODEL_SOURCES = [
];
const CONVERT_URL = "";
const VARIANT_DEFAULTS = {
  base: { steps: 30, cfg: 5.0, sampler: "er_sde", scheduler: "simple" },
  turbo: { steps: 8, cfg: 1.0, sampler: "euler", scheduler: "simple" },
};
// SITE_DEFAULTS:END

export { CONVERT_URL, MODELS, MODEL_SOURCES, PATHS, PROMPT_DEFAULTS, SELECTED_MODEL, VARIANT_DEFAULTS };
