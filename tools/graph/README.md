# tools/graph — 그래프 분석·실험 변형 (전수조사용)

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
| `vae_fp16.py` | VAE 분할 그래프 fp16 변형(conv/all/nonorm, 1.26에선 느림 — 미적용) |
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
