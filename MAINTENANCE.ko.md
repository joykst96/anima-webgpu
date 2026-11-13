# 유지보수 문서

[English](MAINTENANCE.md) | 한국어

[README](README.ko.md)는 기능 소개만 담습니다. 이 문서는 모델을 직접 변환·배포하거나 코드를 고칠 때 알아야 하는 내용입니다.

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
- **WebNN 경로(실험, 선택)**: 고급 옵션에서 고르면 DiT와 VAE만 ORT의 WebNN EP(`ort.all` 번들, 표준 경로와 다른 모듈 인스턴스)로 돕니다. 텍스트 인코더·어댑터·샘플러는 그대로입니다. DiT는 정적 1024² fp16 그래프(`<그래프>_wnn1024.onnx`), VAE는 fp16 계산 그래프(`<vae>_wnn16.onnx`)를 쓰고, LoRA는 표준과 같은 슬롯 방식입니다. NegPip은 같은 그래프에 `negpip_mask` 입력을 넣은 그래프(`<그래프>_wnn1024n.onnx`, 샤드 공유)로 씁니다. 이 입력이 없는 이전 그래프만 있는 모델은 그 그래프로 로드하고 NegPip을 무시합니다. Spectrum은 본체가 내놓은 특징을 GPU 텐서로 둔 채 머리 그래프(`…_head.onnx`·`…_headk.onnx`)만 돌려 예측 스텝을 만듭니다. `navigator.ml`이 없거나 파일이 없거나 세션 생성이 실패하면 표준 경로로 로드합니다.

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
| 변환 매핑표 | `web/convert/anima2b.map.json`, `anima2b.wnn.map.json`(WebNN용) | 들어 있는 매핑표는 관리자의 템플릿 빌드에 맞춰져 있고, 변환기가 템플릿 해시를 확인합니다. 자기 빌드로 `export/make_convert_map.py`를 돌려 다시 만들고(WebNN용은 `--webnn`), 그 그래프를 템플릿으로 올리세요. |
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
4. `add_negpip.py` — `negpip_mask` 입력 추가(mask=1이면 출력 불변). WebNN 빌드는 샤딩이 끝난 그래프에 `--graph-only`로, LoRA 슬롯보다 먼저 건다
5. `shard_onnx_data.py --shard-mb 400` — 가중치를 400MB 샤드로
6. `add_lora_slots.py --rank 48` — LoRA 슬롯 그래프(샤드는 공유)

결과: `out/dit/{prefix}_dit_dyn32_q8a4fns.onnx`(일반) / `_q8a4fnLs.onnx`(LoRA 슬롯), Turbo는 `dit_turbo32`.

### DiT: 페이지용 후처리

페이지는 `<그래프>_x.onnx`(패딩 압축 입력)와 `<그래프>_f16.onnx`('빠르게')를 먼저 찾고, 없으면 원래 그래프로 폴백합니다. 일반·슬롯 그래프마다:

```bash
python -I export/xattn_compact.py <그래프>.onnx <그래프>_x.onnx
python -I export/fp16_zones.py <그래프>_x.onnx <그래프>_f16.onnx --fast
```

그래프만 바꾸고 샤드는 원래 것을 그대로 참조합니다. 이미 서버에 있는 모델 전체에 한 번에 적용하려면 `export/make_fast_graphs.py`를 쓰세요.

### WebNN 그래프 (선택)

고급 옵션 'WebNN (실험)'이 쓰는 그래프는 따로 만듭니다. ComfyUI portable 루트에서 `<repo>\export\webnn\build_webnn.bat <ckpt> <prefix>`를 실행하면 베이스·Turbo DiT(정적 1024² fp16, NegPip·LoRA 슬롯 그래프와 Spectrum용 본체·머리 그래프 포함)와 VAE(fp16 계산) 그래프가 나옵니다. 없으면 페이지는 표준 경로로 로드합니다. 자세한 것은 [export/webnn/README.md](export/webnn/README.md).

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
- **WebNN용 파일(선택)**: 변환 페이지의 'WebNN용 파일도 만들기'를 켜면 같은 폴더에 fp16 그래프 10개(일반·슬롯 × NegPip 유무, 각각의 `_sp`, Spectrum 머리 2종)와 샤드 10개, 머리의 `.data`를 더 씁니다(변형당 약 3.7GB). 매핑표는 `web/convert/anima2b.wnn.map.json`(`make_convert_map.py --webnn`으로 WebNN 빌드 폴더에서 생성), 템플릿은 `out/dit/`의 WebNN 그래프 20개(`<stem>_q8a4fn{s,Ls}_wnn1024{,n}{,_sp}.onnx`, `<stem>_q8a4fns_wnn1024_head{,k}.onnx`)와 `<stem>_q8a4fnLs_wnn1024{,n}.onnx.lora_manifest.json` 4개입니다. 이것도 `build_webnn.bat` 산출물과 바이트 단위로 같습니다. 생성 페이지는 '사용자 설정' 모델에서도 고급 옵션 'WebNN (실험)'을 고르면 이 파일들로 로드합니다(LoRA·Spectrum 포함).
- 어댑터(`llm_adapter`)는 변환하지 않습니다. 페이지는 모든 모델에 공용 어댑터를 씁니다.
- 다른 서브도메인에 따로 둘 수 있습니다: `convert/` + `js/convert-core.js` + `workers/convert-*.js` + `css/tokens.css` + `css/app.css` + 템플릿. 생성 페이지의 링크 주소는 모델 목록의 `convert_url`로 정합니다.

### 페이지에 모델 등록

모델 목록은 `export/model-list.yml`에 두고 `gen_models.py`가 `web/js/config.js`에 주입합니다. 배포마다 다른 것은 이 파일 하나이고, `index.html`에는 배포별 내용이 없습니다. 레포의 `config.js`는 이 기본 목록(Anima만)으로 만든 것입니다.

자기 배포 목록은 따로 두고(git 제외) 결과도 따로 쓰면 레포의 `config.js`를 건드리지 않습니다. 배포할 때 그 파일을 `js/config.js`로 올리고, 로컬에서는 `devserver.mjs --config`로 바꿔 끼웁니다.

```bash
python export/gen_models.py export/model-list.yml web/js/config.js                         # 레포 기본 구성
python export/gen_models.py <내 목록>.yml web/js/config.js --out <내 config>.js    # 배포용 구성을 다른 파일에
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

최상위 `defaults:`는 변형별 스텝·CFG·샘플러 기본값(사용자 설정 모델에도 적용), `selected:`는 처음 선택될 모델, `convert_url:`은 변환 페이지 주소(비우면 `convert/`)입니다. `tag_helper_url:`을 적으면 긍정 프롬프트 옆에 그 주소로 가는 '태그 도우미' 배지가 보이고(비우면 숨김), `sources_title:`은 하단 출처 링크 묶음의 제목입니다(기본 "파인튜닝 모델 출처").

## 배포

```
index.html  css/  js/  workers/  convert/   ← web/ 그대로 (js/config.js가 배포 구성)
tokenizers/{qwen3,t5}/
out/
├── dit/           그래프별 .onnx + _x + _f16, 샤드 .N.bin + .manifest.json, 슬롯 그래프는 .lora_manifest.json (.lora.bin은 페이지가 만듦)
│                  (선택) WebNN: 그래프별 _wnn1024{,n}.onnx + _wnn1024{,n}_sp.onnx(n = NegPip 입력), 샤드 .N.bin + .manifest.json(+ 슬롯은 .lora_manifest.json), 변형별 _wnn1024_head.onnx + .data, _wnn1024_headk.onnx
├── adapter/       anima_llm_adapter.onnx (+ .data)
├── text_encoder/  qwen3_06b_s.onnx + 샤드 + manifest
├── vae/           qwen_image_vae_decoder_dyn32_2dc.onnx + _head/_tail/_head_a/_attn/_head_b (+ .data), (선택) WebNN: _2dc_wnn16.onnx
└── danbooru_ko.csv  태그 자동완성 데이터
```

- `deploy/`에 바로 쓸 수 있는 nginx 설정이 있습니다(`docker compose up -d`). 가중치·그래프는 1년 `immutable`, 페이지와 `js/`·`css/`·`workers/`는 `no-cache`입니다.
- 그래서 **모델 파일을 같은 이름으로 덮어쓰면 기존 사용자에게 반영되지 않습니다.** 새 이름으로 올리고 `config.js`만 바꾸세요. 페이지는 config가 더 참조하지 않는 옛 캐시를 사용자 브라우저에서 정리합니다.
- HTTPS가 필요합니다(WebGPU·OPFS는 secure context 전용, `localhost`만 예외).
- Cloudflare 뒤에 둘 때는 `.bin/.onnx/.data/.json`을 캐시하는 Cache Rule을 추가하세요(기본적으로 캐시되지 않음). 무료 플랜 파일당 한도 때문에 샤드는 500MB 이하로 둡니다.
- 양자화 전 원본 모델은 웹 루트에 두지 마세요.

## 개발·검증 도구

- `node tools/devserver.mjs --port 5173 --upstream <가중치 서버> --local-out out-local` — `web/`을 서빙하고 `/out/`은 가중치 서버로 프록시합니다(GET만, `--upstream` 대신 환경변수 `ANIMA_UPSTREAM`도 됨). `--local-out`에 같은 경로 파일이 있으면 그것을 줍니다(배포 전 그래프 검증용, HTTP Range 지원). `--config <파일>`을 주면 `/js/config.js` 요청에 그 파일을 줍니다(자기 배포 구성으로 띄울 때). 레포 루트에 `tokenizers/`가 있으면 토크나이저도 로컬 것을 줍니다.
- `tools/harness.mjs` — 골든 하네스. 9개 생성 케이스 + 자동완성 8개 질의의 픽셀 SHA-256을 `tools/golden.json`과 비교합니다. 수치가 바뀌는 변경은 기준 이미지 대비 PSNR로 봅니다. 기준은 환경마다 달라 레포에 없습니다: 페이지 콘솔에서 하네스를 불러(`const h = await import("/__tools/harness.mjs")`) `h.full()`을 돌리고, 끝나면 `await h.saveGolden()`이 개발 서버를 통해 `tools/golden.json`을 씁니다. 이후로는 `await h.compareToGolden()`으로 비교합니다.
- `tools/profile.mjs` — 세션 프로파일링으로 커널별 GPU 시간.
- `tools/graph/` — 그래프 분석·실험 스크립트(`info.py`, 실험용 `batch2.py`·`fuse_matmulnbits.py`·`split_context.py` 등, 각 파일 머리에 결과 기록). 페이지용 후처리(`xattn_compact.py`, `fp16_zones.py`)를 포함한 빌드 단계는 `export/`에 있습니다.
- `tools/webnn/` — WebNN 경로 측정 도구(벤치 페이지, WebNN 플래그를 켠 Chrome 실행기).
- `node tools/spectrum-check.mjs` — Spectrum의 스케줄·예측식(`web/js/spectrum.js`)을 원 구현의 기준값(`tools/spectrum-ref-30steps.json`)과 대조합니다. 모델 없이 돕니다.
- 브라우저 변환기 검증: `tools/convert-check.mjs`(변환 코어를 Node로 돌려 빌드 산출물과 바이트 대조. WebNN은 `--map web/convert/anima2b.wnn.map.json`), `tools/convert-folder.mjs`(변환 페이지와 같은 결과 폴더를 디스크에), `tools/convert-compare.py`(그 폴더와 다른 이름으로 빌드한 파이프라인 산출물을 이름만 맞춰 바이트 대조).

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
17. **같은 체크포인트인지 확인하세요.** 같은 계열의 다른 체크포인트도 그래프 구조·크기가 똑같아 구분이 안 됩니다. 원본 PyTorch 출력과 직접 대조하거나, 체크포인트를 `tools/convert-folder.mjs`로 변환해 배포된 샤드와 바이트를 대조하세요(변환기 결과가 파이프라인과 바이트 단위로 같으므로, 샤드가 같으면 같은 체크포인트입니다).
18. **Chrome의 WebNN(DirectML)은 정적 shape만 받고, 받지 않는 형태가 있습니다.** 5D MatMul, `allowzero` Reshape, fp16 LayerNorm은 세션 생성이 실패하거나 끝나지 않습니다(`export/webnn/webnn_fix.py`가 고침). 그래서 WebNN 그래프는 해상도를 고정해 따로 만들고, 페이지는 세션 생성에 시간 제한을 두고 표준 경로로 폴백합니다.
19. **그래프 내부의 fp16 텐서를 그대로 출력으로 내면 결과가 달라질 수 있습니다.** ORT Web이 하던 Cast 쌍 정리가 막혀 계산 경로가 바뀝니다(DiT 출력이 0.08% 달라졌음). 중간 특징을 꺼낼 때는 fp32 쪽에서 새 `Cast`를 달아 내보내세요(`export/webnn/spectrum_split.py`).
