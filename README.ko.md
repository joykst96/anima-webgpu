# Anima WebGPU — 브라우저에서 돌리는 애니메 이미지 생성

[English](README.md) | 한국어

[Anima](https://huggingface.co/circlestone-labs/Anima)(CircleStone Labs / Comfy Org, 2B DiT)를 브라우저 안에서 통째로 실행합니다.
서버 추론도 설치도 없습니다. 가중치를 처음 한 번 받으면 이후 모든 생성은 방문자의 GPU에서 WebGPU로 돌고, 프롬프트는 기기 밖으로 나가지 않습니다.

이 레포에는 전체 과정이 들어 있습니다: ComfyUI/PyTorch → ONNX 변환, 양자화·그래프 후처리, 샤드 패키징, 웹 프론트엔드(빌드 도구 없는 ES 모듈), 정적 서버 설정, 검증 도구.

## 한눈에

RTX 5070 Ti(16GB), Chrome 152, ONNX Runtime Web 1.26 JSPI 빌드, 웜 상태 중앙값:

| 작업 | 시간 |
|---|---|
| Turbo 1024² (8스텝, CFG 1) | **12.4초** (DiT 1.43초/스텝 + VAE 0.75초) |
| Turbo 1024², 고급 옵션 '빠르게' | 11.1~11.5초 |
| Turbo 1024², SPEED | 6.7~6.9초 |
| Turbo 1280² / 1536² | 24.4초 / 약 45초 |
| 베이스 1024² (er_sde 30스텝, CFG 5 = DiT 60회) | 약 88초 |
| 베이스 + EasyCache / cfgpp_ud10_ab 20스텝 | 60~66초 / 56~61초 |
| LoRA 장착 | 장당 +1.5초 (반영 7초) |

- 첫 방문 다운로드는 약 3.6GB(DiT 2.05GB + 텍스트 인코더·어댑터·VAE). 이후 OPFS 캐시에서 로드합니다.
- 로드 직후 256² 워밍업으로 셰이더를 미리 만들어, 첫 생성도 13초대로 끝납니다.
- 같은 입력에서 DiT 1회 출력은 ComfyUI(PyTorch fp32) 원본과 **베이스 0.8~1.7%, Turbo 1.6~3.0%** 차이입니다(상대 L2, σ 1.0~0.1). 샘플러·스케줄러·프롬프트 가중치는 ComfyUI와 같은 계산을 합니다(아래 "정확도").

## 기능

**생성**
- 모델 드롭다운: 베이스(er_sde · 30스텝 · CFG 5)와 Turbo(Turbo LoRA 병합, euler · 8스텝 · CFG 1). 배포 구성에 파인튠을 더 넣을 수 있습니다.
- 해상도: 16의 배수, 종횡비 프리셋. 한 그래프가 모든 해상도를 처리합니다(동적 H/W).
- 샘플러: `euler`, `er_sde`, `cfgpp_ud10_ab`(베이스용, CFG 2 권장 — 20스텝이면 er_sde 30스텝보다 31~34% 빠름). 스케줄러: `simple`, `normal`.
- 배치·큐: 한 번에 1~100장. `-1` 시드는 장마다 새로 뽑습니다. 대기 중인 작업은 개별 취소 가능.
- **SPEED**(옵션): 저해상도에서 시작해 중간에 DCT 스펙트럼 확장으로 풀 해상도로 키웁니다([Xiao et al.](https://arxiv.org/abs/2605.18736)). Turbo 1024²/1280²에서 약 45% 단축, 손 같은 세부가 약간 손실될 수 있습니다. 1024²·1280² 프리셋에서만.

**프롬프트**
- 가중치 문법은 ComfyUI 규칙 그대로: `(태그:1.2)`(숫자는 바깥 괄호 가중치를 덮어씀), `(태그)` = 바깥 ×1.1, `\(` `\)`는 리터럴 괄호. 토큰화도 ComfyUI처럼 가중치 구간마다 따로 합니다.
- **NegPip**: CFG 1(Turbo)에서는 부정 프롬프트가 원래 무시됩니다. 체크하면 부정 프롬프트를 음수 가중치로 긍정 프롬프트에 합쳐, 그 개념을 빼냅니다.
- 태그 자동완성: Danbooru 태그 + 한글 별칭(20만 행)을 브라우저 안에서 검색합니다(Web Worker, IndexedDB 캐시).

**LoRA**
- `.safetensors` LoRA(kohya 또는 `diffusion_model.` 포맷)를 여러 개, 각자 강도로 붙입니다. 재변환 없이 랭크 48 슬롯 그래프에 주입합니다.
- 여러 개는 ΔW 합 → SVD로 정확히 병합합니다(Web Worker). 하나일 때는 강도 슬라이더가 재컴파일 없이 바로 반영됩니다.
- 텍스트 인코더 LoRA·LoKr 등 쓸 수 없는 키는 버리지 않고 보고합니다.

**고급 옵션(톱니바퀴)** — 모두 기본은 꺼짐/표준이고, 끄면 종전과 같은 결과입니다.
- 연산 정밀도 '빠르게': 블록 내부를 fp16으로(정규화·잔차·int8 입력은 fp32 유지). 스텝 −10%, 그림은 세부만 달라집니다.
- cross-attention 패딩 압축: 512토큰 패딩을 "패딩 1개 + 점수 bias ln(m)"으로 바꿉니다(수학적으로 동치). 스텝 −5.6%.
- EasyCache: 변화가 작은 스텝의 DiT 호출을 건너뛰고 직전 결과로 추정합니다(ComfyUI 노드 포팅). 베이스 −25~29%, Turbo 0~11%.

**결과 관리**
- 갤러리·라이트박스·ZIP 다운로드. 각 항목에 생성 조건(시드·크기·시간·켠 옵션)이 표시됩니다.
- 폴더 자동저장(Chromium): 고른 폴더에 `{prefix}_00001.png`로 바로 저장합니다. 파일 이름만 읽어 번호를 이어 가고, 기존 파일 내용은 읽지 않습니다.
- 사용자 설정 모델: 폴더를 고르면 그 안(하위 폴더 포함)에서 모델을 찾아 목록으로 보여 줍니다. 브라우저 변환 결과든 빌드 산출물이든 되고, 고른 모델은 메모리로 바로 로드합니다(OPFS 미사용). Chromium은 고른 폴더를 기억해 다음 방문에 다시 엽니다.

## 동작 구조

| 단계 | 모델 | 형식 | 비고 |
|---|---|---|---|
| 텍스트 인코딩 | Qwen3-0.6B-Base | ONNX fp16, 샤드 | last hidden state |
| 컨텍스트 어댑터 | Anima LLMAdapter | ONNX fp16 | 이중 토큰화(Qwen3 → hidden, T5 → 쿼리), 출력은 512×1024로 0 패딩, T5 토큰 가중치를 곱함 |
| 디노이저 | Anima DiT 2B (Cosmos-Predict2) | ONNX, 가중치 int8(MatMulNBits) + 활성 int8(DP4A), fp16 입출력 | 동적 H/W, RoPE 그래프 내장, 어텐션은 `MultiHeadAttention`(FlashAttention)으로 융합, NegPip·LoRA 슬롯·패딩 압축 입력 |
| 디코드 | Qwen-Image(Wan 2.1) VAE 디코더 | ONNX fp32 | CausalConv3d를 2D로 접음, head(전역 어텐션) / tail(국소 conv) 분할 |

- 샘플러·스케줄러는 JavaScript로 돕니다. 스텝마다 DiT를 1회(CFG 1) 또는 2회(CFG > 1) 호출합니다.
- ORT는 `WebAssembly.Suspending`이 있으면(Chrome 137+) **JSPI 빌드**(네이티브 C++ WebGPU EP)를 씁니다. DP4A int8 matmul과 FlashAttention 커널은 이 빌드에만 있습니다. 없으면 JSEP 빌드로 폴백합니다. 버전은 1.26.0 고정(`?ort=x.y.z`로 다른 버전 시험 가능).
- VAE는 head를 한 번 돌리고 tail을 가로 띠로 나눠(halo 16 latent) 돌립니다. 단일 호출과 비트 단위로 같으면서 정점 메모리가 띠 크기로 묶입니다. 1024²를 넘으면 head의 어텐션도 쿼리 청크별로 나눠 돌립니다.
- DiT를 갈아 끼울 때(모델 전환·정밀도 변경·LoRA 반영)는 텍스트 인코더·어댑터·VAE도 함께 내렸다 올립니다. ORT WebGPU의 버퍼 캐시는 GPU 컨텍스트를 쓰는 세션이 전부 해제돼야 비워지기 때문입니다(함정 14).

## 정확도 (ComfyUI 대비)

ComfyUI v0.43 + anima-base-v1.0을 기준(PyTorch fp32)으로 직접 대조한 결과입니다.

| 항목 | 결과 |
|---|---|
| 스케줄러 `simple`·`normal`, 샘플러 `euler`·`er_sde`·`cfgpp_ud10_ab` | 원본 수식을 옮긴 기준과 ≤2.5e-7 (float32 반올림 수준) |
| SPEED(DCT·스펙트럼 확장) | 원본 `spd_core` 대비 4~6e-8 |
| 프롬프트 가중치·토큰화 | ComfyUI `token_weights`와 구간·가중치 일치, Qwen 토큰열 일치 |
| DiT 1회 출력 (같은 입력) | 베이스 0.8~1.7%, Turbo 1.6~3.0% |
| 그 오차의 출처 | 가중치 int8(블록 32)은 1% 미만. 활성 int8(DP4A)이 대부분이고, 그중 가장 큰 타임스텝 임베딩 2개 층은 fp32로 둡니다 |

같은 시드라도 ComfyUI와 **같은 그림은 아닙니다.** 초기 잡음 난수(mulberry32 vs `torch.randn`)가 다르기 때문입니다. 같은 초기 잡음을 넣으면 같은 그림이 나옵니다.

## 포함하지 않은 것 — 직접 채우기

이 레포에는 코드만 들어 있습니다. 아래는 일부러 뺐고, '선택'인 것은 없어도 페이지가 동작합니다.

| 무엇 | 위치 | 채우는 법 |
|---|---|---|
| 모델 가중치·그래프 | `out/dit/`, `out/text_encoder/`, `out/adapter/`, `out/vae/` | 공식 배포본에서 '모델 변환'의 절차로 만듭니다(라이선스 조건 적용). |
| 토크나이저 (선택) | `tokenizers/{qwen3,t5}/` | `export/prepare_tokenizers.py`. 없으면 Hugging Face Hub에서 받습니다. |
| 태그 자동완성 데이터 (선택) | `out/danbooru_ko.csv` | 배포하지 않습니다. 직접 준비한 Danbooru 태그 목록을 넣으세요. 형식: 헤더 `name,category,post_count,description` 다음 한 줄에 태그 하나, 한글 별칭은 따옴표 친 설명 안의 `키워드:` 뒤에 콤마로 구분. 없으면 자동완성만 꺼집니다. |
| 내 모델 목록 | `web/js/config.js` | 들어 있는 파일은 Anima만 담고 있습니다. 자기 목록(`export/model-list.yml` 또는 따로 둔 배포 목록)으로 `gen_models.py`를 돌려 만드세요. |
| 변환 매핑표 | `web/convert/anima2b.map.json` | 들어 있는 매핑표는 관리자의 템플릿 빌드에 맞춰져 있고, 변환기가 템플릿 해시를 확인합니다. 자기 빌드로 `export/make_convert_map.py`를 돌려 다시 만들고, 그 그래프를 템플릿으로 올리세요. |
| 골든 기준 | `tools/golden.json` | 픽셀 해시는 빌드·ORT 버전·GPU·드라이버마다 다릅니다. 처음 실행할 때 직접 만드세요('개발·검증 도구' 참고). |

## 모델 변환

### 환경

- Anima가 도는 ComfyUI(portable 권장). 확인한 구성은 v0.43.0 + Python 3.13 + PyTorch 2.14(cu130)입니다.
- 추가 패키지: `onnx`, `onnxruntime`, `onnxscript`(portable이면 `python_embeded\python.exe -m pip install ...`).
- 파일(Hugging Face): `circlestone-labs/Anima`의 `anima-base-v1.0.safetensors`, `qwen_3_06b_base.safetensors`, `qwen_image_vae.safetensors`, `circlestone-labs/Anima-Official-LoRAs`의 `anima-turbo-lora-v0.2.safetensors`.
- GPU 16GB면 충분합니다(DiT fp32 export 정점 ~10GB). RAM은 양자화 때 10GB 안팎을 씁니다.
- Windows에서 로그를 파일로 돌릴 때는 `PYTHONIOENCODING=utf-8`을 주세요(torch.onnx 진행 표시가 cp949에서 실패).

### DiT: 한 번에

ComfyUI portable 루트에서:

```bat
export\build_dit.bat ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors anima
```

베이스와 Turbo(스크립트 상단 `TURBO_LORA`를 병합) 각각에 대해 6단계를 돕니다:

1. `export_dit.py --dynamic --fp32-act` — 동적 해상도 ONNX(fp32 활성, fp16 입출력)
2. `quantize_dit.py --bits 8 --no-exclude --accuracy-level 4 --fp32-t-embed` — int8 가중치 + DP4A, t_embedder 2개 층만 활성 fp32
3. `fuse_attention.py` — 분해된 SDPA → `MultiHeadAttention`(FlashAttention). `.fuse_meta.json` 생성
4. `add_negpip.py` — `negpip_mask` 입력 추가(mask=1이면 출력 불변)
5. `shard_onnx_data.py --shard-mb 400` — 가중치를 400MB 샤드로
6. `add_lora_slots.py --rank 48` — LoRA 슬롯 그래프(샤드는 공유)

결과: `out/dit/{prefix}_dit_dyn32_q8a4fns.onnx`(일반) / `_q8a4fnLs.onnx`(LoRA 슬롯), Turbo는 `dit_turbo32`.

### DiT: 페이지용 후처리

페이지는 `<그래프>_x.onnx`(패딩 압축 입력)와 `<그래프>_f16.onnx`('빠르게')를 먼저 찾고, 없으면 원래 그래프로 폴백합니다. 일반·슬롯 그래프마다:

```bash
python -I tools/graph/xattn_compact.py <그래프>.onnx <그래프>_x.onnx
python -I tools/graph/fp16_zones.py <그래프>_x.onnx <그래프>_f16.onnx --fast
```

그래프만 바꾸고 샤드는 원래 것을 그대로 참조합니다. 이미 서버에 있는 모델 전체에 한 번에 적용하려면 `tools/graph/make_fast_graphs.py`를 쓰세요.

### 텍스트 인코더·어댑터·VAE

```bash
python export/export_adapter.py --comfyui <ComfyUI> --ckpt <anima-base-v1.0.safetensors> --out out/adapter/anima_llm_adapter.onnx
python export/export_text_encoder.py --out out/text_encoder/qwen3_06b.onnx
python export/shard_onnx_data.py --src out/text_encoder/qwen3_06b.onnx --out out/text_encoder/qwen3_06b_s.onnx --shard-mb 400

python export/export_vae_decoder.py --comfyui <ComfyUI> --vae <qwen_image_vae.safetensors> \
  --out out/vae/qwen_image_vae_decoder_dyn32_2d.onnx --dynamic --size 512
python export/chunk_vae_attention.py --src out/vae/qwen_image_vae_decoder_dyn32_2d.onnx \
  --out out/vae/qwen_image_vae_decoder_dyn32_2dc.onnx --chunks 8
python export/split_vae.py --src out/vae/qwen_image_vae_decoder_dyn32_2dc.onnx   # → _head / _tail / _head_a / _attn / _head_b
```

VAE는 fp32로 export합니다. `--size`는 trace 크기일 뿐이고(그래프는 동적), fp32 VAE를 1024로 trace하면 16GB에서 OOM이 날 수 있습니다.

### 토크나이저

```bash
python export/prepare_tokenizers.py --out tokenizers
```

Qwen3와 T5 토크나이저를 transformers.js용 `tokenizer.json`으로 저장합니다. 페이지는 `tokenizers/`를 먼저 찾고 없으면 Hugging Face Hub로 폴백합니다.

### 브라우저에서 변환 (ComfyUI 없이)

`web/convert/`는 Anima 2B(28블록) 구조의 `.safetensors` 체크포인트를 브라우저 안에서 페이지용 DiT(그래프 6개 + 400MB 샤드 6개 + manifest)로 바꾸는 독립 페이지입니다. 결과는 고른 폴더에 `<이름>_base/`(Turbo LoRA를 함께 넣으면 `<이름>_turbo/`도)로 저장되고, 생성 페이지의 '사용자 설정'에서 그 폴더를 골라 씁니다. 폴더 저장 API가 없는 브라우저는 파일별 다운로드 링크를 줍니다.

- 그래프 구조는 모든 Anima 2B 체크포인트가 같으므로, 위 파이프라인으로 만든 그래프 12개를 **템플릿**으로 두고 가중치 바이트만 다시 만듭니다(bf16 → fp32 → int8 블록 양자화를 ORT 양자화기와 같은 방식으로, Turbo는 ComfyUI와 같은 fp16 경로로 LoRA 병합). 같은 체크포인트면 파이프라인 산출물과 **바이트 단위로 같습니다**(베이스와 파인튠 두 종으로 확인).
- 템플릿: `out/dit/` 아래 베이스·Turbo 그래프 12개(`<stem>_q8a4fn{s,Ls}{,_x,_f16}.onnx`)와 `<stem>_q8a4fnLs.onnx.lora_manifest.json` 2개. 페이지 기준 `../out/dit/`에서 찾고, `?templates=<URL>`로 바꿀 수 있습니다. 샤드·TE·VAE는 필요 없습니다.
- 매핑표 `web/convert/anima2b.map.json`은 `export/make_convert_map.py`가 빌드 폴더와 원본 체크포인트로 만듭니다(`--prefix` = 빌드 접두사, 변환 결과 파일 이름도 이것을 따름. 템플릿을 해시로 확인하므로 그래프를 다시 빌드하면 함께 다시 만드세요).
- 어댑터(`llm_adapter`)는 변환하지 않습니다. 페이지는 모든 모델에 공용 어댑터를 씁니다.
- 다른 서브도메인에 따로 둘 수 있습니다: `convert/` + `js/convert-core.js` + `workers/convert-*.js` + `css/app.css` + 템플릿. 생성 페이지의 링크 주소는 모델 목록의 `convert_url`로 정합니다.

### 페이지에 모델 등록

모델 목록은 `export/model-list.yml`에 두고 `gen_models.py`가 `web/js/config.js`에 주입합니다. 배포마다 다른 것은 이 파일 하나라, 실제 배포 목록은 따로 두고(예: git에서 제외한 `export/model-list.deploy.yml`) 배포 때 생성하면 됩니다.

```bash
python export/gen_models.py export/model-list.yml web/js/config.js
```

```yaml
models:
  - key: myft                 # = build_dit.bat prefix → out/dit/myft_dit_{dyn32|turbo32}_q8a4f{ns,Ls}.onnx
    label: My Model
    variants: [base, turbo]
    source: https://...       # 선택: 하단 출처 링크
    overrides:                # 선택: variant별 steps·cfg·sampler나 그래프 경로(dit/ditLora)
      turbo: { steps: 6 }
```

최상위 `defaults:`는 변형별 스텝·CFG·샘플러 기본값(사용자 설정 모델에도 적용), `selected:`는 처음 선택될 모델, `convert_url:`은 변환 페이지 주소(비우면 `convert/`)입니다.

## 배포

```
index.html  css/  js/  workers/      ← web/ 그대로 (js/config.js가 배포 구성)
tokenizers/{qwen3,t5}/
out/
├── dit/           그래프별 .onnx + _x + _f16, 샤드 .N.bin + .manifest.json, 슬롯 그래프는 .lora_manifest.json (.lora.bin은 페이지가 만듦)
├── adapter/       anima_llm_adapter.onnx (+ .data)
├── text_encoder/  qwen3_06b_s.onnx + 샤드 + manifest
├── vae/           qwen_image_vae_decoder_dyn32_2dc.onnx + _head/_tail/_head_a/_attn/_head_b (+ .data)
└── danbooru_ko.csv  태그 자동완성 데이터
```

- `deploy/`에 바로 쓸 수 있는 nginx 설정이 있습니다(`docker compose up -d`). 가중치·그래프는 1년 `immutable`, 페이지와 `js/`·`css/`·`workers/`는 `no-cache`입니다.
- 그래서 **모델 파일을 같은 이름으로 덮어쓰면 기존 사용자에게 반영되지 않습니다.** 새 이름으로 올리고 `config.js`만 바꾸세요. 페이지는 config가 더 참조하지 않는 옛 캐시를 사용자 브라우저에서 정리합니다.
- HTTPS가 필요합니다(WebGPU·OPFS는 secure context 전용, `localhost`만 예외).
- Cloudflare 뒤에 둘 때는 `.bin/.onnx/.data/.json`을 캐시하는 Cache Rule을 추가하세요(기본적으로 캐시되지 않음). 무료 플랜 파일당 한도 때문에 샤드는 500MB 이하로 둡니다.
- 양자화 전 원본 모델은 웹 루트에 두지 마세요.

## 개발·검증 도구

- `node tools/devserver.mjs --port 5173 --upstream <가중치 서버> --local-out out-local` — `web/`을 서빙하고 `/out/`은 가중치 서버로 프록시합니다(GET만, `--upstream` 대신 환경변수 `ANIMA_UPSTREAM`도 됨). `--local-out`에 같은 경로 파일이 있으면 그것을 줍니다(배포 전 그래프 검증용, HTTP Range 지원).
- `tools/harness.mjs` — 골든 하네스. 9개 생성 케이스 + 자동완성 8개 질의의 픽셀 SHA-256을 `tools/golden.json`과 비교합니다. 수치가 바뀌는 변경은 기준 이미지 대비 PSNR로 봅니다. 기준은 환경마다 달라 레포에 없습니다: 페이지 콘솔에서 하네스를 불러(`const h = await import("/__tools/harness.mjs")`) `h.full()`을 돌리고, 끝나면 `await h.saveGolden()`이 개발 서버를 통해 `tools/golden.json`을 씁니다. 이후로는 `await h.compareToGolden()`으로 비교합니다.
- `tools/profile.mjs` — 세션 프로파일링으로 커널별 GPU 시간.
- `tools/graph/` — 그래프 분석·후처리 스크립트(`info.py`, `xattn_compact.py`, `fp16_zones.py`, 실험용 `batch2.py`·`fuse_matmulnbits.py`·`split_context.py` 등, 각 파일 머리에 결과 기록).
- 브라우저 변환기 검증: `tools/convert-check.mjs`(변환 코어를 Node로 돌려 빌드 산출물과 바이트 대조), `tools/convert-folder.mjs`(변환 페이지와 같은 결과 폴더를 디스크에), `tools/convert-compare.py`(그 폴더와 다른 이름으로 빌드한 파이프라인 산출물을 이름만 맞춰 바이트 대조).

## 추론 계약 (다른 환경으로 옮길 때)

- Flow 모델, `ModelSamplingDiscreteFlow`, **shift 3.0**, **multiplier 1.0** — DiT의 timestep 입력은 σ 그 자체(0~1)입니다. σ×1000을 넣으면 잡음만 나옵니다.
- `denoised = x − σ·v`(CONST). CFG는 v 공간과 denoised 공간에서 같습니다.
- `simple`: t = j/1000의 `σ(t) = 3t/(1+2t)` 테이블 1000개에서 위부터 균등 추출 후 0 추가. `normal`: σ_max~σ_min을 선형 분할한 뒤 shift를 한 번 더 적용(ComfyUI 그대로).
- er_sde(CONST): λ = σ/(1−σ), α = 1−σ, σ₀ = 1이면 shift(1−10⁻⁴)로 옮김.
- 텍스트 경로는 이중 토큰화(Qwen3 → hidden, T5 → 어댑터 쿼리), 어댑터 출력 × T5 토큰 가중치 후 512로 0 패딩.
- 초기 latent는 `randn`(σ₀ = 1), Wan21 latent(16채널, 공간 /8). 픽셀은 16의 배수.

## 함정 (다시 구현하기 전에)

1. **`--fp32-act`는 속도 선택입니다.** fp16 활성도 NaN 없이 돌지만, 모델 내부 autocast가 fp16↔fp32 Cast를 763개 박아 1~2초 느려집니다. VAE는 반드시 fp32로 export하세요(fp16이면 오버플로·검은 이미지로 알려져 있고, ORT 1.26에서는 fp16 conv가 오히려 느립니다).
2. **ORT Web은 fp16 텐서를 `Float16Array`(실제 값)로 돌려줄 수 있습니다.** 비트 패턴(uint16)으로 가정하면 깨집니다(`js/ort.js`의 `toF32`).
3. **int4 RTN은 이 DiT를 망가뜨립니다.** 민감 층을 빼도 마찬가지입니다. int8 RTN(블록 32)은 1% 미만으로 사실상 무손실이고, 블록 64·128은 더 나쁩니다.
4. **DP4A(`accuracy_level=4`)는 활성도 int8로 양자화합니다.** 가중치 양자화와 별개의 오차이고, M=1인 타임스텝 임베딩 2개 층이 그 대부분을 만듭니다. 그 둘만 `accuracy_level=1`로 두면(`--fp32-t-embed`) 시간 비용 없이 오차가 1/3~1/5로 줄어듭니다. CPU EP는 이 경로를 타지 않으니 화질은 브라우저에서 확인하세요.
5. **dynamo exporter는 노드 이름을 익명화합니다**(`node_linear_3`, `val_46`). 층은 그래프 구조로 찾아야 합니다. 예: t_embedder는 "타임스텝에서만 닿는 선형층 중 K·N이 모두 1024 이상인 것"(adaLN은 전부 랭크 256 병목).
6. **ComfyUI 커스텀 op은 export 전에 순수 PyTorch로 바꿔야 합니다.** `comfy_kitchen`의 `rms_rope_split_half`(v0.43, q/k RMSNorm + RoPE)와 예전 `apply_rope_split_half`, 그리고 `F.rms_norm`(`aten._fused_rms_norm`에 ONNX lowering 없음) — `export/common.py`가 처리합니다.
7. **RoPE가 `max(H, W, T)`를 파이썬 int로 비교해 trace 때 값이 굳습니다.** 가로로 긴 해상도에서 Reshape가 깨지므로 축별 독립 `arange`로 바꿉니다(`common.py`).
8. **브라우저 단일 ArrayBuffer는 ~2GB가 한계**라 가중치를 샤딩합니다. ORT Web의 `externalData`는 파일 목록을 받습니다.
9. **onnxruntime-web의 WebGPU 빌드는 두 갈래입니다.** 기본(JSEP)에는 DP4A·FlashAttention 커널이 없고 JSPI 빌드에 있습니다.
10. **VAE `CausalConv3d`(rank 5)는 느린 naive 커널로 돕니다.** T=1이면 마지막 시간 탭의 2D conv와 같으므로 4D로 접어 Conv2d 경로를 태웁니다.
11. **VAE mid-block 어텐션의 점수 행렬이 고해상도 OOM의 원인**입니다(1280²에서 2.4GB). head_dim 384는 FlashAttention 커널의 workgroup 한계를 넘으므로, 쿼리 축으로 나눠(소프트맥스는 행 단위라 수치 동치) 점수 버퍼를 줄입니다.
12. **ORT WebGPU는 한 번의 `run()` 안에서 해제된 버퍼를 재사용하지 않습니다.** VAE 정점이 중간 텐서 합만큼 커지므로 head/tail로 나누고 tail을 띠로 돌립니다.
13. **GC는 VRAM 압박을 모릅니다.** 큰 GPU 버퍼를 감싼 작은 JS 객체라 수거가 늦습니다. 읽은 텐서는 바로 `dispose()`하세요(일부 빌드엔 없으니 가드).
14. **ORT WebGPU의 버퍼 캐시는 세션 하나를 `release()`해도 비워지지 않습니다.** 캐시는 GPU 컨텍스트에 붙어 있고, 그 컨텍스트를 쓰는 세션이 전부 해제돼야 사라집니다. DiT만 갈아 끼우면 상주 VRAM이 5 → 9.7GB로 늘고 정점이 16GB에 닿아 VAE가 2배 느려집니다. 캐시 모드(`storageBufferCacheMode`)는 ORT JS가 실행 장치를 붙인 *뒤에* 설정을 넣어 바꿀 수 없습니다.
15. **WebGPU의 `Split`은 출력마다 storage buffer를 씁니다.** 출력이 많으면(예: 85개) 셰이더 한도를 넘어 실패하므로 `Slice`로 나눕니다.
16. **ORT 버전을 올리면 빨라진다고 가정하지 마세요.** 1.30은 이 DiT에서 전체 생성이 5~6% 느렸습니다(fp16 커널은 1.30이 더 좋음). 새 페이지에서 전체 생성으로 재야 합니다.
17. **같은 체크포인트인지 확인하세요.** 같은 계열의 다른 체크포인트도 그래프 구조·크기가 똑같아 구분이 안 됩니다. 원본 PyTorch 출력과 직접 대조하는 것이 유일한 확인 방법입니다.

## 라이선스

- **이 레포의 코드**: MIT(`LICENSE`).
- **Anima 모델과 그 변환·양자화·병합 파생물**: [CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md). 비상업 용도만 가능하고, 파생물을 배포할 때는 지정 고지문 표시, "수정함" 명시, 공식 승인으로 오인되지 않게 할 의무가 있습니다(`web/index.html`에 필수 고지 포함). 생성 이미지는 라이선스에 따라 상업적으로 쓸 수 있습니다. Cosmos 파생 모델이므로 NVIDIA Open Model License도 적용됩니다("Built on NVIDIA Cosmos").
- **Qwen3-0.6B-Base**: Apache 2.0.

이 레포에는 모델 가중치가 **들어 있지 않습니다.** 공식 배포본에서 직접 변환하고, 결과물을 배포할 때 라이선스 의무를 지키세요.

## 감사

CircleStone Labs & Comfy Org(Anima), NVIDIA(Cosmos-Predict2), Alibaba(Qwen3, Qwen-Image VAE), ComfyUI(샘플러·스케줄러·EasyCache 참조 구현), sorryhyun/ComfyUI-Spectrum-KSampler(SPEED), ONNX Runtime Web·transformers.js 팀.
