# export/webnn — WebNN용 그래프 빌드 (실험 경로)

생성 페이지의 고급 옵션 › 연산 정밀도 › "WebNN (실험)"이 쓰는 그래프를 만든다.
운영(WebGPU) 빌드는 상위 폴더 `export/`(`build_dit.bat`)이고, **WebNN에만 필요한 것은 전부 이 폴더에 있다.**

WebNN 전용 `export_dit`는 없다. DiT는 운영과 같은 `export/export_dit.py`를 옵션만 달리해 쓴다
(`--dynamic`, `--fp32-act` 없이 → 체크포인트를 fp16으로 로드한 동적 그래프. 페이지가 세션을 만들 때 latent 크기를 `freeDimensionOverrides`로 고정하고, 해상도가 바뀌면 다시 만든다).

| 파일 | 용도 |
|---|---|
| `build_webnn.bat` | 진입점. 베이스·Turbo DiT(export → `webnn_fix` → 샤딩 → NegPip(`add_negpip --graph-only`) → LoRA 슬롯 → int8 ConvRot → Spectrum 분할)와 VAE(`vae_fp16`)를 한 번에 만든다 |
| `webnn_fix.py` | DirectML(= Chrome WebNN 백엔드)이 거부하는 형태를 고친다: 5D MatMul, `allowzero` Reshape, fp16 LayerNorm. 그래프만 |
| `spectrum_split.py` | Spectrum용 분할: 본체 그래프에 특징 출력(fp32·fp16)을 더한 `_sp.onnx`, 머리 그래프(특징, timesteps → 출력), 가중합 머리(`…_headk.onnx`: 특징 16개 + 계수 → 출력). 그래프만 — 머리 가중치 35MB만 따로 쓴다. 동적 그래프면 특징 H·W가 이름 있는 차원(`feat_h`, `feat_w`)이 되고, 머리 안의 `Shape(latent)`를 2 × 특징 H·W로 바꿔 latent 입력을 없앤다 |
| `vae_fp16.py` | VAE 디코더 그래프를 fp16 계산으로 바꾼다(`all`). 그래프만 — 가중치는 원본 `.data` 공유 |
| `webnn_q8.py` | 선형층 가중치를 int8(블록 32)로: `--dq-cast --rot 64`(ConvRot — 입력축 64개씩 Hadamard 회전 후 양자화, 그래프는 입력에 같은 회전)으로 int8 그래프와 자기 샤드(3.64 → 2.05GiB)를 만들고, `--like <int8 그래프>`로 NegPip·LoRA 슬롯 그래프를 그래프만 바꿔 같은 샤드를 가리키게 한다. 회전은 float64로 정확히 계산해 브라우저 변환기가 바이트 단위로 재현한다 |

## 빌드

ComfyUI portable 루트에서(운영 빌드와 같은 환경):

```bat
<repo>\export\webnn\build_webnn.bat ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors animav10
```

| 산출물 | 페이지가 찾는 이름 |
|---|---|
| `out\dit\<prefix>_dit_{dyn32,turbo32}_q8a4fns_wnnd.onnx` + `.N.bin`(10개, 3.64GiB) + `.manifest.json` | `<운영 DiT 그래프>_wnnd.onnx` — fp16 가중치(`loader.js loadWebnnDit`). `dyn32` = 베이스, `turbo32` = Turbo |
| `…_q8a4fns_wnndr.onnx` + `.N.bin`(6개, 2.05GiB) + `.manifest.json` | `<운영 DiT 그래프>_wnndr.onnx` — 고급 옵션의 'int8 가중치 (ConvRot)'. 자기 샤드를 쓴다. 없으면 페이지는 fp16으로 |
| `…_q8a4fnLs_{wnnd,wnndr}.onnx` + `.manifest.json` + `.lora_manifest.json` | LoRA를 붙였을 때의 슬롯 그래프. 샤드는 같은 가중치의 위 그래프 것을 공유 |
| `…_{wnnd,wnndr}n.onnx`(일반·슬롯, `.manifest.json`, 슬롯은 `.lora_manifest.json`) | 위 그래프에 NegPip 입력(`negpip_mask`)을 넣은 것. 있으면 페이지가 이쪽을 먼저 쓴다. 샤드는 공유 |
| `…_sp.onnx`(위 여덟 그래프 모두) | Spectrum 특징 출력을 더한 본체. 있으면 페이지가 이쪽을 올린다(샤드·manifest는 원래 그래프 것) |
| `…_q8a4fns_wnnd_head.onnx` + `.data`, `…_headk.onnx` | Spectrum 머리와 가중합 머리(fp16, 변형마다 하나, LoRA와 무관). fp16·int8 본체가 같이 쓴다 |
| `out\vae\<vae>_wnn16.onnx` | `<운영 VAE 그래프>_wnn16.onnx` (`loader.js loadWebnnVae`) — 원래 동적 그래프라 해상도마다 다시 만든다 |

- 베이스와 Turbo를 둘 다 만든다(변형당 샤드 fp16 약 3.6GB + int8 약 2.0GB). 네 번째 인자 `base` / `turbo`로 하나만 고를 수 있다.
- 해상도 전환: 페이지가 세션을 내린 뒤 가중치를 OPFS 캐시에서 다시 읽어 새 크기로 만든다(약 8초, 다운로드 없음 — `loader.js resizeWebnnDit`).
  이전 정적 그래프(`_wnn1024*`, 1024² 전용)만 있는 모델도 페이지는 그대로 로드한다(이 스크립트는 더 만들지 않는다).
- Spectrum(`web/js/spectrum.js`)은 `_sp` 본체와 머리가 둘 다 있을 때만 켤 수 있다. 없으면 그 기능만 빠진다.
- LoRA 슬롯은 운영과 같은 `export/add_lora_slots.py`로 넣는다(랭크 48). 융합 전 그래프라 `--fuse-meta` 없이 식별하고, 타깃이 fp16이라 브랜치도 fp16이다.
  `lora.bin` 레이아웃(`lora_manifest`)이 운영 슬롯 그래프와 같아 페이지의 변환 워커·강도 입력(`lora_scale`)을 그대로 쓴다. 함께 나오는 0으로 찬 `.lora.bin`은 배포하지 않는다(페이지가 만든다).
- NegPip은 운영과 같은 `export/add_negpip.py`를 `--graph-only`로 샤딩된 그래프에 건다(가중치는 읽지 않고, 샤드 manifest는 새 이름으로 복사).
  WebNN 그래프는 어텐션이 융합되지 않아 Softmax 앵커로 cross-attention v_proj 28개를 찾고, v_proj 입력이 fp16 `context` 그대로라 마스크 곱도 fp16이다(±1이라 정확).
  NegPip 슬롯 그래프는 NegPip 그래프에서 만든다 — v_proj의 LoRA 브랜치도 마스크를 곱한 context를 받아야 해서(운영 빌드와 같은 순서).
- int8 ConvRot(6단계)은 NegPip·슬롯을 넣은 fp16 그래프에서 만든다: 일반 그래프를 `webnn_q8.py --dq-cast --rot 64`로 양자화·샤딩하고, 나머지 셋은 `--like`로 같은 노드 변경을 그래프만 적용한다
  (LoRA 슬롯 가중치는 양자화하지 않고, 슬롯 브랜치는 회전 전 입력을 그대로 받는다. NegPip 그래프는 마스크를 곱한 context에 회전이 걸린다).
  매 실행 GPU에서 int8 → fp16으로 풀어 쓰므로 fp16보다 약 20~25% 느리고, GPU 메모리는 약 1.6GB 덜 쓴다. 회전 덕분에 PyTorch 기준 오차가 int8만 할 때보다 8~24% 작다.
- 파일이 서버에 없으면 페이지는 표준(WebGPU) 경로로 로드한다.
- 중간 산출물 `out\webnn-work\`(변형당 약 8GB)는 지워도 된다.
- 브라우저 변환 페이지의 'WebNN fp16 / int8 (ConvRot) 파일도 만들기'는 이 빌드의 그래프(변형마다 fp16 10개·int8 10개, 머리 포함)와 슬롯 그래프의 `lora_manifest`를 템플릿으로 쓴다. 그래프를 다시 빌드하면 매핑표도 다시 만든다:
  `python export/make_convert_map.py --build <out\dit> --ckpt <체크포인트> --prefix <prefix> --out web/convert/anima2b.wnn.map.json --webnn --check`,
  `… --out web/convert/anima2b.wnnr.map.json --webnn-int8 --check`.
  확인은 `node tools/convert-check.mjs --map <매핑표> --ckpt <체크포인트> --lora <turbo LoRA> --ref <out\dit>`(변환 코어 결과를 이 빌드 산출물과 바이트 대조).

## 측정·실험 도구

빌드가 아닌 것은 `tools/webnn/`에 있다(벤치 페이지, 플래그를 켠 Chrome 실행기, 합성 그래프 생성기).
