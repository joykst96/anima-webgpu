# tools/webnn — WebNN 측정 도구

WebNN 경로의 속도·오차를 재는 도구. 그래프를 만드는 스크립트는 `export/webnn/`에 있다.

| 파일 | 용도 |
|---|---|
| `webnn-bench.html` | 벤치 페이지. `bench`(합성 그래프), `benchDit`(실제 DiT), `benchVae`(실제 VAE), `traceWebNN`(세션 생성이 멈출 때 원인 추적), `benchDyn`(샤드 manifest·LoRA 슬롯 그래프를 해상도 목록마다 세션 생성→실행→해제, PyTorch 기준 오차, `ep: "webgpu"`로 표준 그래프도) |
| `webnn-bench-run.mjs` | WebNN 플래그를 켠 Chrome을 임시 프로필로 띄워 식 하나를 평가하고 결과를 출력한다. `--mon`은 GPU 전체 사용량과 그 Chrome의 메모리(커밋·물리, GPU 프로세스·렌더러별)를 기록해 결과의 `marks` 구간마다 최댓값을 붙인다 |
| `webnn_bench.py` | DiT 블록을 본뜬 정적 합성 그래프(WebNN vs WebGPU 비교용) |
| `webnn_bench_quant.py` | 양자화 변형 합성 그래프(정수 계산·역양자화 방식 비교용) |

```bash
# devserver-local(--local-out out-local)을 띄운 뒤
node tools/webnn/webnn-bench-run.mjs http://localhost:5173/__tools/webnn/webnn-bench.html <식 파일>
```

그래프와 입력은 `out-local/webnn-bench/`에서 읽는다(git 미포함). Chrome 157 기준 `navigator.ml`은
`--enable-features=WebMachineLearningNeuralNetwork`가 있어야 생기고, 실행기가 이 플래그를 붙인다.
