# tools/graph — 그래프 분석·실험 변형 (전수조사용)

운영(WebGPU) 그래프를 분석하거나 실험 변형을 만드는 스크립트. **빌드 단계는 여기 없다** — 운영 빌드(페이지용 후처리 `xattn_compact.py`·`fp16_zones.py`·`make_fast_graphs.py` 포함)는 `export/`,
WebNN 빌드는 `export/webnn/`, WebNN 측정은 `tools/webnn/`에 있다. 표에 없는 `fuse_elementwise.py`·`sattn_downsample.py`도 미적용 실험이다(도구만 보존).

가중치 없이 **그래프 .onnx만**(파일당 ~5MB) 다룬다. 그래프는 가중치 서버에서 HTTP GET으로만 받는다(서버 상태 불변):

```bash
curl -s -o anima_dit_turbo32_q8a4fns.onnx http://<가중치 서버>/out/dit/anima_dit_turbo32_q8a4fns.onnx
```

파이썬은 `onnx` 패키지만 필요 (`uv venv && uv pip install onnx numpy`).

| 파일 | 용도 |
|---|---|
| `info.py` | op 개수·주요 op 속성·Cast 대상 dtype |
| `shape_subgraph.py` | shape 계산 부분그래프(=ORT가 CPU에 두는 노드 후보), timesteps 전용 노드 |
| `attn_variants.py` | MHA만 바꾼 실험 그래프: `unfused`(분해 어텐션), `mhaf16`(q/k/v fp16 Cast → FA fp16) |
| `fuse_matmulnbits.py` | 같은 입력의 MatMulNBits를 N 방향으로 합침(self qkv·adaLN·cross K/V, 비트 동일·속도 차이 없음 — 미적용) |
| `split_context.py` | 프롬프트 전용 노드를 pre 그래프로 분리(비트 동일, −0.3% — 미적용) |
| `batch2.py` | DiT batch 축을 입력에서 받게 수정(CFG batch 2 실험, 비트 동일·속도 이득 없음 — 미적용) |
| `convrot_weights.py` | 회전(ConvRot) + 정수 가중치 양자화 변형별 DiT 오차(가중치만, 미적용) |
| `convrot_act_layers.py` | 층별 활성 int8 양자화 오차와 회전 효과(실제 활성 기준, 미적용) |
| `convrot_w8a8_sim.py` | 가중치·활성 int8을 그래프 안에서 흉내 낸 전체 DiT 오차(회전 유무, 미적용) |
| `serve.mjs` | 실험 그래프를 CORS로 서빙 (`node tools/graph/serve.mjs <dir> 5174`) |

## 실험 그래프로 A/B 하는 법 (페이지 코드 수정 없음)

외부 데이터(샤드) 참조 이름은 원본과 같으므로, 페이지가 DiT 세션을 만들 때 그래프 바이트만 바꿔 끼운다.
새로 연 페이지에서 '모델 로드' **전에** 콘솔에서:

```js
const ort = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.jspi.bundle.min.mjs");
const orig = ort.InferenceSession.create;
ort.InferenceSession.create = async function (graph, opts = {}) {
  if ((opts.externalData || []).some((e) => /anima_dit_turbo32_q8a4fns\.onnx\.\d+\.bin/.test(e.path)))
    graph = new Uint8Array(await (await fetch("http://127.0.0.1:5174/dit_mhaf16.onnx")).arrayBuffer());
  return orig.call(this, graph, opts);
};
```

그 뒤 `tools/harness.mjs`의 `bench()` / `compareToReference()`로 시간·PSNR을 잰다.
주의: PSNR은 8스텝 샘플링이 작은 수치 차이를 증폭하므로 **수학적으로 동치인 변형끼리도 ~19dB**가 나온다.
