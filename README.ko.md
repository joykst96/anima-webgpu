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
- 한 페이지에서 두 모델 변형 선택: 베이스(er_sde · 30스텝 · CFG 5), Turbo LoRA 머지본(euler · 8스텝 · CFG 1). 추가로 **사용자 설정** 옵션으로 자신의 ONNX DiT를 디스크에서 직접 로드 (모바일 친화적, OPFS 불필요).
- **네이티브 WebGPU EP (JSPI 빌드)** + DP4A int8 matmul(`accuracy_level=4`) — JSEP 빌드 대비 샘플러 약 2배 빠름, 구형 브라우저는 자동 폴백.
- **FlashAttention 융합 (DiT)**: 분해된 SDPA를 `com.microsoft.MultiHeadAttention`으로 묶어 JSPI EP의 FA2 커널을 발동. attention score 행렬을 통째로 들지 않게 되어(O(S²)→O(S)) VRAM 상주 사용량이 절반으로 떨어지고, 1024 버킷 초과 해상도가 DiT 단계에서 통과합니다.
- **VAE chunked attention**: VAE 디코더의 mid-block self-attention을 쿼리 토큰축으로 8분할해 score 단일 버퍼를 1/8로 제한(수치 동치). 고해상도(예 1280²·1536²)에서 VAE가 2 GB 단일 버퍼 한계로 OOM 나던 문제를 해소 — **1536²까지 브라우저에서 생성**.
- **런타임 LoRA**: 아무 Anima `.safetensors` LoRA(kohya 또는 `diffusion_model.` 포맷)를 강도와 함께 장착. 여러 개는 ΔW + SVD로 정확히 병합(Web Worker에서 — UI 안 멈춤). 재export 불필요.
- **프롬프트 가중치** `(태그:1.2)`와 **NegPip**(CFG=1/터보에서 부정 프롬프트), 둘 다 ComfyUI 충실 구현.

## 동작 구조

| 단계 | 모델 | 포맷 | 비고 |
|---|---|---|---|
| 텍스트 인코딩 | Qwen3-0.6B-Base | ONNX fp16 | last hidden states |
| 컨텍스트 어댑터 | Anima LLMAdapter | ONNX fp16 | **이중 토큰화**: Qwen3 토큰 → 인코더 hidden states, T5 토큰 → 어댑터 쿼리. 출력은 512×1024로 zero-pad |
| 디노이저 | Anima DiT 2B (Cosmos-Predict2) | ONNX, int8 가중치(DP4A용 `accuracy_level=4`) / fp32 활성(권장 — '함정' 참고), fp16 입출력 | H/W 동적 축, RoPE 그래프 내부; self/cross attention은 `MultiHeadAttention`으로 융합(FA2); NegPip·LoRA 슬롯 입력 선택 |
| 디코드 | Qwen-Image (Wan 2.1) VAE 디코더 | ONNX fp32 | T=1에서 CausalConv3d를 2D로 폴딩(Conv2d 커널 경로); Wan21 역정규화 베이킹; mid-block attention은 쿼리축 8-chunk로 score 버퍼 제한 |

샘플러(Euler / ER-SDE-Solver)와 시그마 스케줄은 순수 JavaScript로 돌고, 스텝마다 DiT 세션을 1회(CFG 1) 또는 2회(CFG > 1) 호출합니다.

## 요구사항

**변환 머신** — Anima가 정상 동작하는 ComfyUI 환경, Python 3.10+, 최신 PyTorch(신형 dynamo ONNX exporter 사용), `onnx`, `onnxruntime`, `transformers`, `sentencepiece`. GPU는 16GB면 여유 — DiT fp32 export가 VRAM ~10GB 정점. ComfyUI **portable** 사용자는 아래 명령의 `python`을 `python_embeded\python.exe`로 바꾸고 portable 루트에서 실행하세요.

**서빙** — 아무 정적 파일 서버. 프로덕션에선 HTTPS 필수 (WebGPU와 OPFS가 secure context 요구, `localhost`는 테스트용 예외).

**클라이언트** — WebGPU 지원 브라우저(데스크톱 Chrome/Edge), `shader-f16` 지원 GPU, VRAM 8GB+ 권장. 최초 방문 시 ~2.5GB 다운로드 (이후 OPFS 캐시).

## 1. 모델 변환

필요 파일: `anima-base-v1.0.safetensors`, 공식 **Anima Turbo LoRA**(고속 변형용, 선택), `qwen_image_vae.safetensors`. Qwen3-0.6B-Base는 HF에서 자동 다운로드됩니다.

```bash
# DiT — 베이스 (--fp32-act 권장: fp16도 되지만 ~1~2초 느림 — '함정' 참고)
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
  --out out/vae/qwen_image_vae_decoder_dyn32_2d.onnx --dynamic --size 1024
# T=1 전제로 CausalConv3d를 등가 2D conv로 폴딩 (--keep-3d로 끔, --verify로 동치 검증)
```

참고: VAE의 `--size 512`는 trace 크기일 뿐이고(그래프는 동적), fp32 VAE를 1024로 trace하면 16GB 카드에서 OOM이 날 수 있습니다. `export_dit.py --verify`를 주면 두 해상도에서 ONNX vs PyTorch 출력 비교까지 수행합니다.

**VAE attention chunk (고해상도 OOM 대응)** — Wan VAE 디코더의 mid-block에는 full-latent 해상도 self-attention이 있어, score 행렬 `(1,1,S,S)`가 1280²(S=25600)에서 ~2.4 GB로 브라우저 단일 버퍼 한계(2 GB)를 넘겨 OOM이 납니다. `chunk_vae_attention.py`가 이 SDPA를 쿼리 토큰축으로 N등분(기본 8)해 chunk별 부분 score만 만들도록 전개합니다(softmax는 행 단위라 **수치 동치**). K·V는 공유. 1536²(S=36864)에서 chunk당 score ~0.68 GB로 통과합니다.

```bash
python export/chunk_vae_attention.py \
  --src out/vae/qwen_image_vae_decoder_dyn32_2d.onnx \
  --out out/vae/qwen_image_vae_decoder_dyn32_2dc.onnx --chunks 8
```

> 참고: VAE attention을 `MultiHeadAttention`으로 융합하는 길은 막혔습니다 — head_dim 384가 WebGPU FA2 커널의 workgroup 스토리지 한계(32 KB)를 넘겨 셰이더 컴파일이 실패합니다(48 KB 필요). single-head를 multi-head로 쪼개면 수치가 달라지므로, score chunk가 정공법입니다. chunk는 일반 MatMul+Softmax라 head_dim 제약이 없습니다.

## 2. DiT 양자화 · 기능 추가 · 샤딩

DiT 전체 파이프라인: **양자화 → FlashAttention 융합 → NegPip → 샤딩 → LoRA 슬롯**. fuse는 양자화 직후(NegPip이 v_proj를 재배선하기 전)에 수행하고, NegPip은 샤딩 전(그래프 편집), LoRA 슬롯은 샤딩 *후*(graph-only, 같은 샤드 재사용)에 추가합니다. fuse가 attention의 Softmax 앵커를 흡수하므로, NegPip·LoRA 슬롯은 fuse가 남긴 `.fuse_meta.json`(q/k/v/o proj 노드명)으로 attention을 식별합니다(`--fuse-meta`). 메타가 없으면 기존 Softmax 앵커로 폴백합니다.

```bash
# int8 weight-only + accuracy_level=4 (DP4A 정수 커널 발동 → 샘플러 약 2배)
python export/quantize_dit.py --src out/dit/anima_dit_dyn32.onnx \
  --out out/dit/anima_dit_dyn32_q8a4.onnx --bits 8 --no-exclude --accuracy-level 4

# FlashAttention 융합: 분해 SDPA → com.microsoft.MultiHeadAttention. .fuse_meta.json 생성.
python export/fuse_attention.py --src out/dit/anima_dit_dyn32_q8a4.onnx \
  --out out/dit/anima_dit_dyn32_q8a4f.onnx

# NegPip: negpip_mask 입력 추가, cross-attn v_proj 재배선 (mask=1이면 출력 동일).
#   fuse_meta가 옆에 있으면 메타 기반 식별, 없으면 Softmax 앵커 폴백.
python export/add_negpip.py --src out/dit/anima_dit_dyn32_q8a4f.onnx \
  --out out/dit/anima_dit_dyn32_q8a4fn.onnx

# 가중치를 CDN/브라우저 친화적 샤드로 분할
python export/shard_onnx_data.py --src out/dit/anima_dit_dyn32_q8a4fn.onnx \
  --out out/dit/anima_dit_dyn32_q8a4fns.onnx --shard-mb 400

# LoRA 슬롯: 런타임 LoRA용 저랭크 사이드브랜치(랭크 48). 위 샤드를 그대로 재사용.
#   샤딩으로 파일명이 바뀌므로 fuse_meta 경로를 명시해야 함.
python export/add_lora_slots.py --src out/dit/anima_dit_dyn32_q8a4fns.onnx \
  --out out/dit/anima_dit_dyn32_q8a4fnLs.onnx --rank 48 \
  --fuse-meta out/dit/anima_dit_dyn32_q8a4f.onnx.fuse_meta.json
```

비슬롯(`q8a4fns`)과 슬롯(`q8a4fnLs`) 그래프를 **둘 다** 서버에 두세요: 페이지는 LoRA가 없으면 더 빠른 비슬롯 그래프를 쓰고, LoRA를 장착할 때만 슬롯 그래프로 전환합니다(샤드 동일, 추가 다운로드 없음). Turbo export에도 전체 체인을 반복하세요.

**배치 자동화** — 체크포인트마다 6단계 × 2변형을 손으로 돌리는 건 번거롭습니다. `export/build_dit.bat`(Windows, ComfyUI portable)가 `export_dit → quantize → fuse → negpip → shard → lora-slots` 전체 체인을 base·Turbo 둘 다 한 번에 실행합니다:

```bat
REM ComfyUI portable 루트에서:
export\build_dit.bat ComfyUI\models\diffusion_models\my-finetune.safetensors myft
```

`myft_dit_dyn32_q8a4fns/fnLs.onnx`와 `myft_dit_turbo32_*`(샤드 + manifest 포함)를 생성합니다. Turbo 변형은 export 시점에 Turbo LoRA를 머지합니다(`TURBO_LORA` 경로는 스크립트 상단에서 설정). 체크포인트를 다시 빌드할 때마다 실행하고, 출력을 `index.html`의 `MODELS` 맵에 등록하면 됩니다.

### 페이지에 모델 추가하기

DiT 파일을 `out/dit/`에 둔 뒤, `index.html`의 **두 곳**에 모델을 등록합니다. 두 곳은 **같은 key**를 써야 하며, 다르면 모델이 로드되지 않습니다.

**1. `MODELS` 맵** (`const MODELS` 검색) — base/Turbo 한 쌍 추가:

```js
mymodel: {
  label: "My Model",
  dit:     "out/dit/mymodel_dit_dyn32_q8a4fns.onnx",
  ditLora: "out/dit/mymodel_dit_dyn32_q8a4fnLs.onnx",
  steps: 30, cfg: 5.0, sampler: "er_sde", scheduler: "simple",
},
mymodel_turbo: {
  label: "My Model+Turbo",
  dit:     "out/dit/mymodel_dit_turbo32_q8a4fns.onnx",
  ditLora: "out/dit/mymodel_dit_turbo32_q8a4fnLs.onnx",
  steps: 8, cfg: 1.0, sampler: "euler", scheduler: "simple",
},
```

key는 자유(`[A-Za-z0-9_]`), `dit`/`ditLora` 경로는 실제 파일명과 일치해야 합니다. base는 30스텝 / CFG 5 / er_sde, Turbo는 8스텝 / CFG 1 / euler가 표준입니다.

**2. 모델 드롭다운** (`id="modelSel"` 검색) — key마다 `<option>` 하나씩 추가:

```html
<option value="mymodel">My Model (베이스 · 30스텝 · CFG 5)</option>
<option value="mymodel_turbo">My Model+Turbo (8스텝 · CFG 1)</option>
```

`value`는 `MODELS` key와 정확히 일치해야 합니다.

최소 구성(레거시):

```bash
python export/quantize_dit.py --src out/dit/anima_dit_dyn32.onnx \
  --out out/dit/anima_dit_dyn32_q8.onnx --bits 8 --no-exclude
python export/shard_onnx_data.py --src out/dit/anima_dit_dyn32_q8.onnx \
  --out out/dit/anima_dit_dyn32_q8s.onnx --shard-mb 480
``` `.data`가 CDN 캐시 한도를 넘는 다른 모델도 샤딩하는 게 좋습니다 (fp16 텍스트 인코더가 ~1.2GB):

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
├── dit/           변형별: *_q8a4fns.onnx(비슬롯) + *_q8a4fnLs.onnx(LoRA 슬롯),
│                 각각 .bin 샤드 + .onnx.manifest.json; 슬롯 그래프는 .onnx.lora_manifest.json도.
│                 샤드는 두 그래프가 공유.
├── adapter/       anima_llm_adapter.onnx (+ .data)
├── text_encoder/  qwen3_06b_s.onnx + 샤드 + manifest
└── vae/           qwen_image_vae_decoder_dyn32_2dc.onnx (+ .data)
```

파일명이 다르면 `web/index.html` 상단의 `PATHS` / `MODELS` 상수를 수정하세요. **양자화 전 원본 모델을 웹 루트에 두지 마세요.**

바로 쓸 수 있는 정적 서버가 `deploy/`에 있습니다 (`docker compose up -d`; nginx — 가중치는 `immutable` 캐시 헤더, 페이지는 `no-cache`). Cloudflare 뒤에 두는 경우 `.bin/.onnx/.data/.json`을 캐시 대상으로 지정하는 Cache Rule을 반드시 추가하세요 — 이 확장자들은 기본적으로 캐시되지 않아서, 규칙 없이는 모든 요청이 오리진까지 내려옵니다. 무료 플랜의 파일당 캐시 한도 때문에 샤드는 ~500MB 이하로 유지하세요. 가중치가 `immutable`로 서빙되므로 **모델 파일을 같은 이름으로 덮어쓰면 안 됩니다** — 새 파일명으로 올리고 `index.html`만 갱신하세요.

## 브라우저 기능

- **프롬프트 가중치** — `(태그:1.2)`로 강조, `(태그)` = ×1.1, 중첩은 곱, `\(` `\)`는 리터럴 괄호(단보루 태그용). 어댑터 출력에 토큰 단위로 적용 — ComfyUI와 동일 지점.
- **NegPip** — CFG=1(터보)에서는 부정 프롬프트가 무시됩니다. NegPip 체크 시 부정 프롬프트가 음수 가중치 그룹으로 병합되고, 슬롯 모델의 `negpip_mask`가 해당 토큰의 cross-attn value 부호를 뒤집어 개념을 빼냅니다. `add_negpip.py`로 만든 DiT 필요.
- **런타임 LoRA** — `.safetensors` LoRA를 여러 개, 각자 강도로 추가한 뒤 **LoRA 반영**을 누릅니다(지연 적용 — N개를 N번이 아니라 1번에 컴파일). 변환(safetensors 파싱 + 멀티 LoRA ΔW/SVD 병합)은 Web Worker에서 진행바와 함께 돌고, 끝날 때까지 생성은 비활성화됩니다. **단일** LoRA는 강도가 그래프 입력(`lora_scale`)이라 슬라이더가 재컴파일 없이 다음 생성에 즉시 반영되고, **여러 개**일 때는 강도가 SVD 병합에 들어가 변경 시 재병합합니다. 어느 쪽이든 그래프는 랭크 48 유지. 미지원 키(텍스트 인코더 LoRA, LoKr)와 슬롯 밖 모듈은 조용히 버리지 않고 리포트합니다. `add_lora_slots.py`로 만든 슬롯 모델 필요.
- **사용자 설정 모델** — *사용자 설정* 드롭다운으로 DiT(`.onnx` + 샤드 + manifest)를 디스크에서 골라 메모리에 직접 로드(OPFS 미사용, 모바일 동작). TE/어댑터/VAE는 기본 경로 사용.

## 구현 노트 (다른 환경으로 포팅할 때)

ComfyUI 소스 대조로 검증한 Anima의 추론 계약:

- Flow 모델, `ModelSamplingDiscreteFlow`, **shift = 3.0**, 그리고 결정적으로 **multiplier = 1.0**: DiT의 timestep 입력은 **σ 그 자체(0~1)이지 σ×1000이 아님**. ×1000을 먹이면 순수 노이즈가 나옵니다.
- 시그마 스케줄(`simple`): t = j/1000에 대해 `σ(t)=3t/(1+2t)`로 1000개 테이블을 만들고, 위에서부터 균등 샘플링 후 0을 추가.
- Euler 스텝: `x ← x + (σ_next − σ)·v`; `denoised = x − σ·v` (CONST). CFG는 v 공간과 denoised 공간에서 등가.
- ER-SDE (CONST 경로): λ = σ/(1−σ), α = 1−σ, logit 발산 방지를 위해 σ₀를 σ(1−10⁻⁴)로 오프셋. 3단계 솔버 전체 구현은 `run_pipeline.py` / `index.html` 참고.
- 텍스트 경로는 **이중 토큰화**: 같은 프롬프트가 Qwen3 토크나이저(→ 인코더 hidden states)와 T5 토크나이저(→ 어댑터 쿼리 id, vocab 32128)를 모두 거침. 어댑터 출력은 512 토큰으로 zero-pad.
- 초기 latent: `randn × σ₀`, σ₀ = 1.0. latent 포맷은 Wan21 (16ch, /8 공간 압축), 픽셀 크기는 16의 배수여야 함.

## 밟은 지뢰들 (재구현 전 필독)

1. **DiT 활성 dtype — `--fp32-act`는 NaN 방지가 아니라 *속도* 선택 (2026-06 정정).** 이전엔 "WebGPU에서 fp16 활성이 NaN → fp32-act 필수"로 알고 있었으나 틀렸다. `--fp32-act` **없이**(fp16 활성) export해도 NaN 없이 정상 동작한다. 다만 그 경우 모델 내부 autocast(블록 forward fp32 / 서브모듈 fp16)와 `if x.dtype==fp16: x=x.float()` 분기가 그래프에 fp16↔fp32 왕복 Cast를 대량으로 박는다(실측: Cast 6개 → 763개). 그 결과 weight(int8)·DP4A 경로가 동일한데도 추론이 오히려 ~1~2초 *느려진다*(12s → 14s). 그래서 `--fp32-act`가 속도 면에서 권장 기본값이다. 그 Cast를 안전하게 제거하려면 그래프 전반을 수술해야 하는데(리스크 큼, 메모리 이득 불확실) 현재로선 가치가 없다. **별개로**, VAE는 fp32로 export하라 — Wan VAE는 fp16에서 실제로 오버플로해 고전적인 검은 이미지 버그가 난다. (참고: `add_lora_slots.py`는 LoRA 브랜치를 fp32로 하드코딩하므로 현 상태로는 fp16 활성 DiT와 비호환 — 해당 파일의 경고 주석 참고.)
2. **ORT Web이 `Float16Array`를 반환.** 네이티브 `Float16Array`가 있는 브라우저에서는 fp16 텐서의 `.data`가 uint16 비트 패턴이 아니라 *실제 float*입니다. 비트 연산으로 디코드하면 전부 깨집니다(0 + NaN). 감지해서 분기하세요 (`index.html`의 `toF32`/`makeF16Tensor`).
3. **naive int4 RTN은 diffusion DiT를 망가뜨립니다** — 민감 레이어(adaLN/t_embedder/final_layer)를 제외해도 마찬가지. int8을 쓰거나, int4가 필요하면 HQQ를 쓰세요.
4. **dynamo ONNX exporter는 이름을 익명화합니다** (`node_linear`, `val_46`) — 이름 기반 레이어 매칭이 불가능. `quantize_dit.py`는 그래프 위상(`timesteps`에서 도달 가능하고 `latent`에서 도달 불가)과 가중치 shape으로 민감 레이어를 식별합니다.
5. **`aten._fused_rms_norm`에 ONNX lowering이 없습니다** (최신 PyTorch). export 전에 `F.rms_norm`을 수동 분해 구현으로 몽키패치합니다 (`common.py`).
6. **브라우저 단일 ArrayBuffer 한계(~2GB)** 때문에 큰 `.data`를 통째로 로드할 수 없습니다 — 그래서 샤딩이며, ORT Web의 `externalData`는 파일 목록을 네이티브로 지원합니다.
7. ComfyUI 커스텀 op(`comfy_kitchen.apply_rope_split_half`)는 trace 전에 순수 PyTorch 등가 구현으로 치환해야 합니다.
8. **onnxruntime-web은 WebGPU 빌드를 두 갈래로 배포합니다.** 기본(`ort.webgpu.*`, JSEP)에는 DP4A·FlashAttention 커널이 **없고**, `ort.jspi.*` 빌드가 그 커널을 가진 네이티브 C++ WebGPU EP입니다. 이 페이지는 `WebAssembly.Suspending`이 있으면(Chrome 137+) JSPI 빌드(`1.26.0` 고정)를 쓰고, 없으면 폴백합니다. `subgroup-matrix` 경로는 wasm 빌드에서 컴파일 자체가 제외되니 좇지 마세요.
9. **DP4A는 노드 속성 `accuracy_level=4`가 필요합니다**(`quantize_dit.py --accuracy-level 4`). 추가로 `K%128==0 && N%16==0 && block_size%32==0`. 이는 활성화를 int8로 동적 양자화하는 것 — 가중치 양자화와는 *다른* 정확도 트레이드오프라 브라우저에서 이미지 품질을 확인하세요(`run_pipeline.py`의 CPU EP는 이 경로를 안 탑니다).
10. **VAE `CausalConv3d`는 fp32 전용 rank-5**라 ORT가 느린 naive conv3d 커널로 돌립니다. 단일 이미지(T=1)에서는 마지막 시간탭의 2D conv와 수학적으로 동일 — `export_vae_decoder.py`가 가중치를 4D로 실체화해 Conv2d 커널이 돌게 만들어 스텝당 디코드를 거의 즉시로 만듭니다. 비디오(T>1)에는 사용 불가.
11. **RoPE가 내부적으로 `max(H, W, T)`를 씁니다** — 파이썬 `int` 비교라 dynamo가 trace 시점 값으로 굳혀, trace 때의 H/W 대소관계가 박히고 **landscape(W>H) 해상도가 Reshape 에러로 실패**합니다(정사각형·portrait는 통과). 비정사각형 trace로도 안 고쳐지므로(dynamo가 심볼을 안 나눔), `common.py`의 `patch_comfy_for_export`가 RoPE 임베딩을 축별 독립 `arange`로 monkeypatch합니다(수치 동일). `export_dit.py --verify`가 이제 landscape 케이스도 검증합니다.
12. **VAE 디코더의 mid-block attention이 고해상도 OOM의 원인**입니다. score 행렬 `(1,1,S,S)`가 full-latent 해상도(1024²면 S=16384, 1280²면 25600)에서 단일 버퍼로 잡혀, 1280²에서 ~2.4 GB로 2 GB 한계를 넘습니다. `MultiHeadAttention` 융합은 head_dim 384가 WebGPU FA2의 workgroup 한계(32 KB)를 초과해 컴파일이 실패하고, single-head를 쪼개면 수치가 달라집니다. 해법은 쿼리 토큰축 chunk(`chunk_vae_attention.py`) — softmax가 행 단위라 수치 동치이면서 score 버퍼를 1/N로 줄입니다. 이 페이지는 1536² 이하만 생성하도록 두고 N=8을 씁니다.

## 성능 (참고치)

RTX 5070 Ti, 1024×1024, JSPI 빌드 + DP4A + 2D VAE 기준: **Turbo ≈ 11초**(8스텝, CFG 1), **베이스 ≈ 70초**(30스텝, CFG 5 → DiT 60회) — 로컬 ComfyUI 대비 약 3배(이전 ~10배에서 단축). DP4A가 샘플러 시간을 대략 절반으로 줄였고, VAE를 2D로 폴딩해 디코드가 사실상 즉시가 됐습니다. LoRA 장착 시 슬롯 디스패치로 ~9초 추가됩니다. 새 해상도의 첫 생성은 셰이더 특화 때문에 느리고 이후 캐시됩니다.

**FlashAttention 융합(DiT)의 실측 가치는 속도가 아니라 메모리였습니다.** 현 해상도 범위에서 총 생성 시간은 거의 동률(±수 초)이지만, attention score 행렬을 통째로 들지 않게 되어 DiT의 VRAM 상주 사용량이 절반으로 평탄해지고, unfused로는 OOM 나던 1024 버킷 초과 해상도가 DiT 단계에서 통과합니다. **VAE chunked attention**까지 적용해, 고해상도에서 VAE가 score 단일 버퍼로 OOM 나던 마지막 병목도 해소 — RTX 5070 Ti에서 **1280² ≈ 29초, 1536² ≈ 58초**로 생성됩니다.

> 참고: DiT를 반복 재컴파일하면(예: LoRA를 여러 번 토글) 현재 ORT Web에서 페이지 새로고침 전까지 GPU 메모리가 누수될 수 있습니다 — 알려진 업스트림 이슈. 페이지는 재컴파일을 최소화(LoRA 지연 반영, 단일 상주 세션)해 정상 사용에선 피하도록 했습니다.

## 라이선스

- **이 레포의 코드**: MIT (`LICENSE` 참고).
- **Anima 모델 및 모든 변환/양자화/머지 파생물**: [CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md). 비상업 사용만 가능하며, 파생물 배포 시 지정 고지문 표시, "수정함" 명시, 공식 승인 오인 금지 의무가 따릅니다 — `web/index.html`에 필수 고지가 포함돼 있습니다. 생성 이미지(Outputs)는 라이선스에 따라 상업적 사용이 가능합니다. Cosmos 파생 모델로서 NVIDIA Open Model License도 적용됩니다 ("Built on NVIDIA Cosmos").
- **Qwen3-0.6B-Base**: Apache 2.0.

이 레포는 모델 가중치를 **포함하지 않습니다.** 공식 배포본에서 직접 변환하고, 결과물을 배포할 때 라이선스 의무를 유지하세요.

## 감사

CircleStone Labs & Comfy Org (Anima), NVIDIA (Cosmos-Predict2), Alibaba (Qwen3, Qwen-Image VAE), ComfyUI 프로젝트 (샘플러/스케줄러 레퍼런스 구현), ONNX Runtime Web과 transformers.js 팀.
