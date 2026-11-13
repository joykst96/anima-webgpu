# Anima WebGPU — 브라우저에서 돌리는 애니메 이미지 생성

[English](README.md) | 한국어

[Anima](https://huggingface.co/circlestone-labs/Anima)(CircleStone Labs / Comfy Org, 2B DiT)를 브라우저 안에서 통째로 실행합니다.
서버 추론도 설치도 없습니다. 가중치를 처음 한 번 받으면 이후 모든 생성은 방문자의 GPU에서 WebGPU로 돌고, 프롬프트는 기기 밖으로 나가지 않습니다.

이 레포에는 전체 과정이 들어 있습니다: ComfyUI/PyTorch → ONNX 변환, 양자화·그래프 후처리, 샤드 패키징, 웹 프론트엔드(빌드 도구 없는 ES 모듈), 정적 서버 설정, 검증 도구.

## 한눈에

RTX 5070 Ti(16GB) 기준:

| 작업 | 시간 |
|---|---|
| Turbo 1024² (8스텝, CFG 1) | **약 12.5초** |
| Turbo 1024², 고급 옵션 '빠르게' | 11.1~11.5초 |
| Turbo 1024², SPEED | 6.7~6.9초 |
| Turbo 1280² / 1536² | 24.4초 / 약 45초 |
| 베이스 1024² (er_sde 30스텝, CFG 5 = DiT 60회) | 약 88초 |
| 베이스 + EasyCache / cfgpp_ud10_ab 20스텝 | 60~66초 / 56~61초 |
| LoRA 장착 | 장당 +1.5초 (반영 7초) |
| (실험) WebNN, Turbo 1024² | 약 3.4초 |
| (실험) WebNN, 베이스 1024² 30스텝 / + Spectrum | 약 24초 / 약 15초 |

- WebGPU가 되는 브라우저가 필요합니다(Chrome 137 이상 권장).
- 첫 방문 다운로드는 약 3.6GB입니다. 이후에는 브라우저 캐시(OPFS)에서 로드합니다.
- 샘플러·스케줄러·프롬프트 가중치는 ComfyUI와 같은 계산을 합니다. 같은 입력에서 DiT 1회 출력은 ComfyUI(PyTorch fp32) 원본과 베이스 0.8~1.7%, Turbo 1.6~3.0% 차이입니다([정확도](MAINTENANCE.ko.md#정확도-comfyui-대비)).

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
- 연산 정밀도 'WebNN (실험)': DiT와 VAE를 브라우저의 WebNN으로 fp16 계산합니다. 1024² 한 장이 Turbo 약 3.4초, 베이스(30스텝) 약 24초입니다(RTX 5070 Ti). Chrome의 WebNN 플래그와 WebNN 그래프를 배포한 모델이 필요하고, 1024×1024 전용이며 SPEED·NegPip은 쓸 수 없습니다(LoRA는 됩니다). 쓸 수 없는 환경에서는 표준으로 로드됩니다. 변환 페이지에서 WebNN용 파일도 함께 만들면 직접 변환한 모델에서도 쓸 수 있습니다.
- Spectrum(WebNN 전용): 일부 스텝은 모델의 블록 계산을 건너뛰고, 앞 스텝들의 특징으로 예측한 특징을 출력층에만 통과시킵니다([Han et al.](https://arxiv.org/abs/2603.01623)). 베이스 30스텝이면 17스텝만 실제로 계산해 약 24초 → 15초, 그림은 세부만 달라집니다. euler·er_sde 샘플러에서 동작합니다.
- cross-attention 패딩 압축: 512토큰 패딩을 "패딩 1개 + 점수 bias ln(m)"으로 바꿉니다(수학적으로 동치). 스텝 −5.6%.
- EasyCache: 변화가 작은 스텝의 DiT 호출을 건너뛰고 직전 결과로 추정합니다(ComfyUI 노드 포팅). 베이스 −25~29%, Turbo 0~11%.

**결과 관리**
- 갤러리·라이트박스·ZIP 다운로드. 각 항목에 생성 조건(시드·크기·시간·켠 옵션)이 표시됩니다.
- 폴더 자동저장(Chromium): 고른 폴더에 `{prefix}_00001.png`로 바로 저장합니다. 파일 이름만 읽어 번호를 이어 가고, 기존 파일 내용은 읽지 않습니다.
- 사용자 설정 모델: 폴더를 고르면 그 안(하위 폴더 포함)에서 모델을 찾아 목록으로 보여 줍니다. 브라우저 변환 결과든 빌드 산출물이든 되고, 고른 모델은 메모리로 바로 로드합니다(OPFS 미사용). Chromium은 고른 폴더를 기억해 다음 방문에 다시 엽니다.

## 문서

이 레포에는 코드만 있고 모델 가중치는 들어 있지 않습니다. 직접 변환해 올리려면 아래 문서를 보세요.

- [MAINTENANCE.ko.md](MAINTENANCE.ko.md) — 동작 구조, 정확도 검증, 모델 변환, 배포, 개발·검증 도구, 추론 계약, 함정.
- 폴더별 설명: [export/webnn/](export/webnn/README.md)(WebNN 그래프 빌드), [tools/graph/](tools/graph/README.md)(그래프 분석·실험), [tools/webnn/](tools/webnn/README.md)(WebNN 측정).

## 라이선스

- **이 레포의 코드**: MIT(`LICENSE`).
- **Anima 모델과 그 변환·양자화·병합 파생물**: [CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md). 비상업 용도만 가능하고, 파생물을 배포할 때는 지정 고지문 표시, "수정함" 명시, 공식 승인으로 오인되지 않게 할 의무가 있습니다(`web/index.html`에 필수 고지 포함). 생성 이미지는 라이선스에 따라 상업적으로 쓸 수 있습니다. Cosmos 파생 모델이므로 NVIDIA Open Model License도 적용됩니다("Built on NVIDIA Cosmos").
- **Qwen3-0.6B-Base**: Apache 2.0.

이 레포에는 모델 가중치가 **들어 있지 않습니다.** 공식 배포본에서 직접 변환하고, 결과물을 배포할 때 라이선스 의무를 지키세요.

## 감사

CircleStone Labs & Comfy Org(Anima), NVIDIA(Cosmos-Predict2), Alibaba(Qwen3, Qwen-Image VAE), ComfyUI(샘플러·스케줄러·EasyCache 참조 구현), sorryhyun/ComfyUI-Spectrum-KSampler(SPEED·Spectrum), ONNX Runtime Web·transformers.js 팀.
