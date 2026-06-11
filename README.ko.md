# Anima WebGPU — 브라우저 로컬 애니메 이미지 생성

[English](README.md) | 한국어

[Anima](https://huggingface.co/circlestone-labs/Anima) (CircleStone Labs / Comfy Org, 2B DiT)를 브라우저에서 통째로 실행합니다. 서버 추론 없음, 설치 없음 — 가중치를 최초 1회 다운로드하면 이후 모든 생성은 방문자 본인의 GPU에서 WebGPU로 돌아갑니다.

이 레포는 전체 파이프라인을 담고 있습니다: PyTorch/ComfyUI → ONNX 변환 스크립트, weight-only 양자화, 브라우저/CDN용 샤드 패키징, 단일 파일 웹 프론트엔드, 배포 설정.

**특징**

- 100% 클라이언트 사이드 추론 (ONNX Runtime Web, WebGPU EP). 프롬프트가 기기 밖으로 나가지 않음.
- 단일 그래프로 동적 해상도·종횡비 지원 (RoPE를 심볼릭 trace — 해상도별 빌드 불필요).
- ComfyUI 샘플링의 충실 포팅: `er_sde`(CONST/flow 경로)와 `simple` 스케줄러, 수치 단위로 일치 검증.
- int8 weight-only 양자화 (MatMulNBits): 연산은 fp16, 다운로드 ~2.3GB. `shader-f16`만 지원하면 어떤 WebGPU GPU에서도 동작 — INT8 텐서코어 같은 하드웨어 요구 없음.
- 가중치 샤딩 (파일당 ~480MB 이하): 브라우저 단일 ArrayBuffer 한계 우회 + 무료 CDN 캐시 한도 대응.
- OPFS 캐싱: 첫 방문만 다운로드, 재방문은 디스크에서 수 초 내 로드.
- 한 페이지에서 두 모델 변형 선택: 베이스(er_sde · 30스텝 · CFG 5), Turbo LoRA 머지본(euler · 8스텝 · CFG 1).

## 동작 구조

| 단계 | 모델 | 포맷 | 비고 |
|---|---|---|---|
| 텍스트 인코딩 | Qwen3-0.6B-Base | ONNX fp16 | last hidden states |
| 컨텍스트 어댑터 | Anima LLMAdapter | ONNX fp16 | **이중 토큰화**: Qwen3 토큰 → 인코더 hidden states, T5 토큰 → 어댑터 쿼리. 출력은 512×1024로 zero-pad |
| 디노이저 | Anima DiT 2B (Cosmos-Predict2) | ONNX, int8 가중치 / fp32 활성, fp16 입출력 | H/W 동적 축, RoPE는 그래프 내부 |
| 디코드 | Qwen-Image (Wan 2.1) VAE 디코더 | ONNX fp32 | Wan21 latent 역정규화를 그래프에 베이킹 |

샘플러(Euler / ER-SDE-Solver)와 시그마 스케줄은 순수 JavaScript로 돌고, 스텝마다 DiT 세션을 1회(CFG 1) 또는 2회(CFG > 1) 호출합니다.

## 요구사항

**변환 머신** — Anima가 정상 동작하는 ComfyUI 환경, Python 3.10+, 최신 PyTorch(신형 dynamo ONNX exporter 사용), `onnx`, `onnxruntime`, `transformers`, `sentencepiece`. GPU는 16GB면 여유 — DiT fp32 export가 VRAM ~10GB 정점. ComfyUI **portable** 사용자는 아래 명령의 `python`을 `python_embeded\python.exe`로 바꾸고 portable 루트에서 실행하세요.

**서빙** — 아무 정적 파일 서버. 프로덕션에선 HTTPS 필수 (WebGPU와 OPFS가 secure context 요구, `localhost`는 테스트용 예외).

**클라이언트** — WebGPU 지원 브라우저(데스크톱 Chrome/Edge), `shader-f16` 지원 GPU, VRAM 6GB+ 권장. 최초 방문 시 ~2.5GB 다운로드 (이후 OPFS 캐시).

## 1. 모델 변환

필요 파일: `anima-base-v1.0.safetensors`, 공식 **Anima Turbo LoRA**(고속 변형용, 선택), `qwen_image_vae.safetensors`. Qwen3-0.6B-Base는 HF에서 자동 다운로드됩니다.

```bash
# DiT — 베이스 (WebGPU에는 fp32 활성이 필수 — 아래 '함정' 참고)
python export/export_dit.py --comfyui /path/to/ComfyUI \
  --ckpt /path/to/anima-base-v1.0.safetensors \
  --out out/dit/anima_dit_dyn32.onnx --dynamic --fp32-act

# DiT — Turbo 변형 (export 시점에 LoRA 머지 — 별도 머지 단계 불필요)
python export/export_dit.py --comfyui /path/to/ComfyUI \
  --ckpt /path/to/anima-base-v1.0.safetensors \
  --lora /path/to/anima_turbo.safetensors \
  --out out/dit/anima_dit_turbo32.onnx --dynamic --fp32-act

# 어댑터 (같은 체크포인트에서 추출) / 텍스트 인코더 / VAE
python export/export_adapter.py --comfyui /path/to/ComfyUI \
  --ckpt /path/to/anima-base-v1.0.safetensors --out out/adapter/anima_llm_adapter.onnx
python export/export_text_encoder.py --out out/text_encoder/qwen3_06b.onnx
python export/export_vae_decoder.py --comfyui /path/to/ComfyUI \
  --vae /path/to/qwen_image_vae.safetensors \
  --out out/vae/qwen_image_vae_decoder_dyn32.onnx --dynamic --size 512
```

참고: VAE의 `--size 512`는 trace 크기일 뿐이고(그래프는 동적), fp32 VAE를 1024로 trace하면 16GB 카드에서 OOM이 날 수 있습니다. `export_dit.py --verify`를 주면 두 해상도에서 ONNX vs PyTorch 출력 비교까지 수행합니다.

## 2. DiT 양자화 & 샤딩

```bash
# int8 weight-only (권장 — 이 DiT에서 사실상 무손실)
python export/quantize_dit.py --src out/dit/anima_dit_dyn32.onnx \
  --out out/dit/anima_dit_dyn32_q8.onnx --bits 8 --no-exclude

# 가중치를 CDN/브라우저 친화적 샤드로 분할
python export/shard_onnx_data.py --src out/dit/anima_dit_dyn32_q8.onnx \
  --out out/dit/anima_dit_dyn32_q8s.onnx --shard-mb 400
```

Turbo export에도 같은 두 단계를 반복하세요. `.data`가 CDN 캐시 한도를 넘는 다른 모델도 샤딩하는 게 좋습니다 (fp16 텍스트 인코더가 ~1.2GB):

```bash
python export/shard_onnx_data.py --src out/text_encoder/qwen3_06b.onnx \
  --out out/text_encoder/qwen3_06b_s.onnx --shard-mb 400
```

이 선택의 근거: **naive int4 RTN은 이 DiT를 눈에 띄게 망가뜨립니다** (diffusion 트랜스포머는 LLM보다 가중치 양자화에 훨씬 민감 — SVDQuant가 존재하는 이유). int8 RTN은 사실상 무손실입니다. 다운로드 크기가 더 중요하면 `--algo hqq`로 고품질 int4(~1.2GB)를 시도할 수 있습니다. 양자화는 weight-only라 셰이더가 런타임에 fp16/fp32로 역양자화하므로 **하드웨어 INT8 요구가 전혀 없고**, batch-1 추론은 대역폭 바운드라 오히려 fp16보다 빠릅니다.

## 3. 로컬 검증 (권장)

```bash
pip install onnxruntime pillow   # 또는 onnxruntime-directml / onnxruntime-gpu
python export/run_pipeline.py \
  --dit out/dit/anima_dit_dyn32_q8s.onnx \
  --adapter out/adapter/anima_llm_adapter.onnx \
  --te out/text_encoder/qwen3_06b_s.onnx \
  --vae out/vae/qwen_image_vae_decoder_dyn32.onnx \
  --prompt "1girl, silver hair" --steps 30 --cfg 5.0 --size 768 --out check.png
```

브라우저와 정확히 같은 파이프라인(동일 스케줄/샘플러 수식)을 파이썬으로 돌립니다. `check.png`가 정상이면 이후 브라우저에서 생기는 문제는 프론트엔드 코드 문제로 격리됩니다 — 이 분리가 개발 과정에서 여러 번 결정적이었습니다.

## 4. 토크나이저 (자가호스팅)

```bash
python export/prepare_tokenizers.py --out tokenizers
```

Qwen3와 진짜 `t5-v1_1` 토크나이저를 transformers.js가 요구하는 fast 포맷(`tokenizer.json`)으로 변환해 저장합니다. 페이지는 `tokenizers/`를 먼저 찾고, 없으면 HF Hub로 폴백합니다.

## 5. 배포

웹 루트 구성:

```
index.html
tokenizers/{qwen3,t5}/
out/
├── dit/           *_q8s.onnx + .bin 샤드 + .manifest.json   (변형별)
├── adapter/       anima_llm_adapter.onnx (+ .data)
├── text_encoder/  qwen3_06b_s.onnx + 샤드 + manifest
└── vae/           qwen_image_vae_decoder_dyn32.onnx (+ .data)
```

파일명이 다르면 `web/index.html` 상단의 `PATHS` / `MODELS` 상수를 수정하세요. **양자화 전 원본 모델을 웹 루트에 두지 마세요.**

바로 쓸 수 있는 정적 서버가 `deploy/`에 있습니다 (`docker compose up -d`; nginx — 가중치는 `immutable` 캐시 헤더, 페이지는 `no-cache`). Cloudflare 뒤에 두는 경우 `.bin/.onnx/.data/.json`을 캐시 대상으로 지정하는 Cache Rule을 반드시 추가하세요 — 이 확장자들은 기본적으로 캐시되지 않아서, 규칙 없이는 모든 요청이 오리진까지 내려옵니다. 무료 플랜의 파일당 캐시 한도 때문에 샤드는 ~500MB 이하로 유지하세요. 가중치가 `immutable`로 서빙되므로 **모델 파일을 같은 이름으로 덮어쓰면 안 됩니다** — 새 파일명으로 올리고 `index.html`만 갱신하세요.

## 구현 노트 (다른 환경으로 포팅할 때)

ComfyUI 소스 대조로 검증한 Anima의 추론 계약:

- Flow 모델, `ModelSamplingDiscreteFlow`, **shift = 3.0**, 그리고 결정적으로 **multiplier = 1.0**: DiT의 timestep 입력은 **σ 그 자체(0~1)이지 σ×1000이 아님**. ×1000을 먹이면 순수 노이즈가 나옵니다.
- 시그마 스케줄(`simple`): t = j/1000에 대해 `σ(t)=3t/(1+2t)`로 1000개 테이블을 만들고, 위에서부터 균등 샘플링 후 0을 추가.
- Euler 스텝: `x ← x + (σ_next − σ)·v`; `denoised = x − σ·v` (CONST). CFG는 v 공간과 denoised 공간에서 등가.
- ER-SDE (CONST 경로): λ = σ/(1−σ), α = 1−σ, logit 발산 방지를 위해 σ₀를 σ(1−10⁻⁴)로 오프셋. 3단계 솔버 전체 구현은 `run_pipeline.py` / `index.html` 참고.
- 텍스트 경로는 **이중 토큰화**: 같은 프롬프트가 Qwen3 토크나이저(→ 인코더 hidden states)와 T5 토크나이저(→ 어댑터 쿼리 id, vocab 32128)를 모두 거침. 어댑터 출력은 512 토큰으로 zero-pad.
- 초기 latent: `randn × σ₀`, σ₀ = 1.0. latent 포맷은 Wan21 (16ch, /8 공간 압축), 픽셀 크기는 16의 배수여야 함.

## 밟은 지뢰들 (재구현 전 필독)

1. **WebGPU에서 fp16 활성이 NaN.** CPU/DML 실행 공급자는 fp16 연산을 조용히 fp32로 승격하므로 "로컬에선 되는" 모델이 진짜 fp16 셰이더에서는 NaN이 납니다. DiT는 `--fp32-act`(fp16 입출력, fp32 연산을 캐스트로 베이킹), VAE는 fp32로 export하세요 — Wan VAE는 fp16에서 오버플로해 고전적인 검은 이미지 버그를 재현합니다.
2. **ORT Web이 `Float16Array`를 반환.** 네이티브 `Float16Array`가 있는 브라우저에서는 fp16 텐서의 `.data`가 uint16 비트 패턴이 아니라 *실제 float*입니다. 비트 연산으로 디코드하면 전부 깨집니다(0 + NaN). 감지해서 분기하세요 (`index.html`의 `toF32`/`makeF16Tensor`).
3. **naive int4 RTN은 diffusion DiT를 망가뜨립니다** — 민감 레이어(adaLN/t_embedder/final_layer)를 제외해도 마찬가지. int8을 쓰거나, int4가 필요하면 HQQ를 쓰세요.
4. **dynamo ONNX exporter는 이름을 익명화합니다** (`node_linear`, `val_46`) — 이름 기반 레이어 매칭이 불가능. `quantize_dit.py`는 그래프 위상(`timesteps`에서 도달 가능하고 `latent`에서 도달 불가)과 가중치 shape으로 민감 레이어를 식별합니다.
5. **`aten._fused_rms_norm`에 ONNX lowering이 없습니다** (최신 PyTorch). export 전에 `F.rms_norm`을 수동 분해 구현으로 몽키패치합니다 (`common.py`).
6. **브라우저 단일 ArrayBuffer 한계(~2GB)** 때문에 큰 `.data`를 통째로 로드할 수 없습니다 — 그래서 샤딩이며, ORT Web의 `externalData`는 파일 목록을 네이티브로 지원합니다.
7. ComfyUI 커스텀 op(`comfy_kitchen.apply_rope_split_half`)는 trace 전에 순수 PyTorch 등가 구현으로 치환해야 합니다.

## 성능 (참고치)

RTX 5070 Ti, 1024×1024 기준: Turbo 변형(8스텝, CFG 1)은 수십 초, 베이스(30스텝, CFG 5 → DiT 60회)는 분 단위. WebGPU에는 아직 fused attention이 없어 네이티브 CUDA 대비 상당한 격차가 있고, ORT Web 커널이 발전하면서 좁혀질 영역입니다. 새 해상도의 첫 생성은 셰이더 특화 때문에 느리고 이후 캐시됩니다.

## 라이선스

- **이 레포의 코드**: MIT (`LICENSE` 참고).
- **Anima 모델 및 모든 변환/양자화/머지 파생물**: [CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md). 비상업 사용만 가능하며, 파생물 배포 시 지정 고지문 표시, "수정함" 명시, 공식 승인 오인 금지 의무가 따릅니다 — `web/index.html`에 필수 고지가 포함돼 있습니다. 생성 이미지(Outputs)는 라이선스에 따라 상업적 사용이 가능합니다. Cosmos 파생 모델로서 NVIDIA Open Model License도 적용됩니다 ("Built on NVIDIA Cosmos").
- **Qwen3-0.6B-Base**: Apache 2.0.

이 레포는 모델 가중치를 **포함하지 않습니다.** 공식 배포본에서 직접 변환하고, 결과물을 배포할 때 라이선스 의무를 유지하세요.

## 감사

CircleStone Labs & Comfy Org (Anima), NVIDIA (Cosmos-Predict2), Alibaba (Qwen3, Qwen-Image VAE), ComfyUI 프로젝트 (샘플러/스케줄러 레퍼런스 구현), ONNX Runtime Web과 transformers.js 팀.
