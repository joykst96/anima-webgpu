# export/webnn — WebNN용 그래프 빌드 (실험 경로)

생성 페이지의 고급 옵션 › 연산 정밀도 › "WebNN (실험)"이 쓰는 그래프를 만든다.
운영(WebGPU) 빌드는 상위 폴더 `export/`(`build_dit.bat`)이고, **WebNN에만 필요한 것은 전부 이 폴더에 있다.**

WebNN 전용 `export_dit`는 없다. DiT는 운영과 같은 `export/export_dit.py`를 옵션만 달리해 쓴다
(`--dynamic`·`--fp32-act` 없이 `--size 1024` → 체크포인트를 fp16으로 로드한 정적 그래프).

| 파일 | 용도 |
|---|---|
| `build_webnn.bat` | 진입점. 베이스·Turbo DiT(export → `webnn_fix` → 샤딩 → LoRA 슬롯 → Spectrum 분할)와 VAE(`vae_fp16`)를 한 번에 만든다 |
| `webnn_fix.py` | DirectML(= Chrome WebNN 백엔드)이 거부하는 형태를 고친다: 5D MatMul, `allowzero` Reshape, fp16 LayerNorm. 그래프만 |
| `spectrum_split.py` | Spectrum용 분할: 본체 그래프에 특징 출력(fp32·fp16)을 더한 `_sp.onnx`, 머리 그래프(특징, timesteps → 출력), 가중합 머리(`…_headk.onnx`: 특징 16개 + 계수 → 출력). 그래프만 — 머리 가중치 35MB만 따로 쓴다 |
| `vae_fp16.py` | VAE 디코더 그래프를 fp16 계산으로 바꾼다(`all`). 그래프만 — 가중치는 원본 `.data` 공유 |
| `webnn_q8.py` | (선택, 페이지 미사용) fp16 가중치를 int8 블록 32로 줄여 다운로드 3.64 → 2.05GiB. 속도 같음, 오차는 운영 수준 |

## 빌드

ComfyUI portable 루트에서(운영 빌드와 같은 환경):

```bat
<repo>\export\webnn\build_webnn.bat ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors animav10
```

| 산출물 | 페이지가 찾는 이름 |
|---|---|
| `out\dit\<prefix>_dit_{dyn32,turbo32}_q8a4fns_wnn1024.onnx` + `.N.bin` + `.manifest.json` | `<운영 DiT 그래프>_wnn1024.onnx` (`loader.js loadWebnnDit`). `dyn32` = 베이스, `turbo32` = Turbo |
| `…_q8a4fnLs_wnn1024.onnx` + `.manifest.json` + `.lora_manifest.json` | `<운영 LoRA 슬롯 그래프>_wnn1024.onnx` — LoRA를 붙였을 때. 샤드는 위와 공유 |
| `…_q8a4fns_wnn1024_sp.onnx`, `…_q8a4fnLs_wnn1024_sp.onnx` | 위 두 그래프 + Spectrum 특징 출력. 있으면 페이지가 이쪽을 올린다(샤드·manifest는 위 그래프 것) |
| `…_q8a4fns_wnn1024_head.onnx` + `.data`, `…_headk.onnx` | Spectrum 머리와 가중합 머리(변형마다 하나씩, LoRA와 무관, `.data` 공유). 가중합 머리가 있으면 특징을 GPU에 둔 채 예측한다(없으면 CPU에서 합산 — 느리지만 동작) |
| `out\vae\<vae>_wnn16.onnx` | `<운영 VAE 그래프>_wnn16.onnx` (`loader.js loadWebnnVae`) |

- 베이스와 Turbo를 둘 다 만든다(변형당 샤드 약 3.6GB). 네 번째 인자 `base` / `turbo`로 하나만 고를 수 있다.
- Spectrum(`web/js/spectrum.js`)은 `_sp` 본체와 머리가 둘 다 있을 때만 켤 수 있다. 없으면 그 기능만 빠진다.
- LoRA 슬롯은 운영과 같은 `export/add_lora_slots.py`로 넣는다(랭크 48). 융합 전 그래프라 `--fuse-meta` 없이 식별하고, 타깃이 fp16이라 브랜치도 fp16이다.
  `lora.bin` 레이아웃(`lora_manifest`)이 운영 슬롯 그래프와 같아 페이지의 변환 워커·강도 입력(`lora_scale`)을 그대로 쓴다. 함께 나오는 0으로 찬 `.lora.bin`은 배포하지 않는다(페이지가 만든다).
- 파일이 서버에 없으면 페이지는 표준(WebGPU) 경로로 로드한다.
- 중간 산출물 `out\webnn-work\`(약 4GB)는 지워도 된다.
- 브라우저 변환 페이지의 'WebNN용 파일도 만들기'는 이 빌드의 그래프 12개(+ 슬롯 그래프의 `lora_manifest` 2개)를 템플릿으로 쓴다. 그래프를 다시 빌드하면 매핑표도 다시 만든다:
  `python export/make_convert_map.py --build <out\dit> --ckpt <체크포인트> --prefix <prefix> --out web/convert/anima2b.wnn.map.json --webnn --check`,
  확인은 `node tools/convert-check.mjs --map web/convert/anima2b.wnn.map.json --ckpt <체크포인트> --lora <turbo LoRA> --ref <out\dit>`(변환 코어 결과를 이 빌드 산출물과 바이트 대조).

## 측정·실험 도구

빌드가 아닌 것은 `tools/webnn/`에 있다(벤치 페이지, 플래그를 켠 Chrome 실행기, 합성 그래프 생성기).
