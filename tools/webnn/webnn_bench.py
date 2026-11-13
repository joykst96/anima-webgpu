"""DiT 블록을 본뜬 정적 shape 테스트 그래프 (WebNN vs WebGPU 시간 비교용).

구조는 Anima DiT 2B의 self-attention + MLP 블록과 같게: hidden 2048, heads 16(head_dim 128),
MLP 4배, 토큰 4096(= 1024² 이미지). 가중치는 난수이고 블록끼리 공유한다(파일 크기만 줄고 연산량은 같음).
입력은 [1,1,2048]만 받아 상수 [1,4096,2048]에 더하고, 출력은 [1,1,8] 조각만 내보내 전송 시간을 뺀다.

  mode fp32 / fp16 : MatMul + 분해 어텐션 (WebNN·WebGPU 양쪽에서 도는 공통 그래프)
  mode best        : MatMulNBits(int8, accuracy_level 4 = DP4A) + com.microsoft.MultiHeadAttention
                     (지금 페이지가 쓰는 WebGPU 최선 경로. WebNN EP는 못 돌림)
  part block / mlp / attn : 블록 전체 / MLP만 / 어텐션만

만들기:  uv run --no-project --python 3.12 --with onnx --with numpy tools/webnn/webnn_bench.py out-local/webnn-bench
재기:    devserver-local을 띄우고  node tools/webnn/webnn-bench-run.mjs http://localhost:<port>/__tools/webnn/webnn-bench.html <식 파일>
         (식 예: bench({graph:'block_fp16', ep:'webnn-gpu', bundle:'ort.all.bundle.min.mjs', runs:7}))

결과 (2026-11-13, RTX 5070 Ti, Chrome 157 + --enable-features=WebMachineLearningNeuralNetwork, ORT 1.26.0,
      4블록·4096토큰, 웜 7회 중앙값 ms, 변형마다 새 브라우저):
  graph        WebGPU(jspi)   WebNN gpu(ort.all)
  block_fp32      242.8          310.4     (출력 소수 4자리까지 동일)
  block_fp16      170.9           34.0     (출력은 fp16 오차 안에서 일치)
  block_best      164.6            -       (= 페이지의 현재 경로. 블록당 41ms, 실제 DiT는 51ms/블록)
  mlp_fp32         98.1          244.8
  mlp_fp16         71.6           12.0
  mlp_best         39.8            -
  attn_fp16         -             22.7
  attn_best       138.0            -
→ WebNN은 fp16에서만 빠르다(블록 4.8배). fp32는 WebGPU보다 느리다. 실제 DiT 그래프로는 아직 안 쟀다.
"""
import sys, os
import numpy as np
import onnx
from onnx import helper as oh, numpy_helper as nh, TensorProto as TP

H, HEADS, HD, MLP, TOK, BLOCKS = 2048, 16, 128, 8192, 4096, 4
out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
rng = np.random.default_rng(0)


def build(mode, part):
    ft = np.float16 if mode == "fp16" else np.float32
    tp = TP.FLOAT16 if mode == "fp16" else TP.FLOAT
    nodes, inits = [], []
    made = set()

    def init(name, arr):
        if name not in made:
            inits.append(nh.from_array(arr, name)); made.add(name)
        return name

    def lin(x, w, K, N, out):
        if mode == "best":
            nb = K // 32
            q = np.clip(np.round(rng.normal(128, 30, (N, nb, 32))), 0, 255).astype(np.uint8)
            init(w + "_q", q)
            init(w + "_s", np.full((N * nb,), 0.02 / 30, np.float32))
            nodes.append(oh.make_node("MatMulNBits", [x, w + "_q", w + "_s"], [out], domain="com.microsoft",
                                      K=K, N=N, bits=8, block_size=32, accuracy_level=4))
        else:
            init(w, rng.normal(0, 0.02, (K, N)).astype(ft))
            nodes.append(oh.make_node("MatMul", [x, w], [out]))
        return out

    init("base", rng.normal(0, 1, (1, TOK, H)).astype(ft))
    init("g", np.ones(H, ft)); init("b", np.zeros(H, ft))
    init("s4", np.array([1, TOK, HEADS, HD], np.int64)); init("s3", np.array([1, TOK, H], np.int64))
    init("scale", np.array(1 / np.sqrt(HD), ft))
    nodes.append(oh.make_node("Add", ["base", "x"], ["h0"]))
    h = "h0"
    for i in range(BLOCKS):
        p = f"b{i}_"
        if part in ("block", "attn"):
            nodes.append(oh.make_node("LayerNormalization", [h, "g", "b"], [p + "n1"], axis=-1))
            q, k, v = (lin(p + "n1", "W" + c, H, H, p + c) for c in "qkv")
            if mode == "best":
                nodes.append(oh.make_node("MultiHeadAttention", [q, k, v], [p + "o"], domain="com.microsoft",
                                          num_heads=HEADS, scale=float(1 / np.sqrt(HD))))
            else:
                for c, perm in ((q, [0, 2, 1, 3]), (k, [0, 2, 3, 1]), (v, [0, 2, 1, 3])):
                    nodes.append(oh.make_node("Reshape", [c, "s4"], [c + "r"]))
                    nodes.append(oh.make_node("Transpose", [c + "r"], [c + "t"], perm=perm))
                nodes.append(oh.make_node("MatMul", [q + "t", k + "t"], [p + "sc"]))
                nodes.append(oh.make_node("Mul", [p + "sc", "scale"], [p + "ss"]))
                nodes.append(oh.make_node("Softmax", [p + "ss"], [p + "pr"], axis=-1))
                nodes.append(oh.make_node("MatMul", [p + "pr", v + "t"], [p + "ov"]))
                nodes.append(oh.make_node("Transpose", [p + "ov"], [p + "ot"], perm=[0, 2, 1, 3]))
                nodes.append(oh.make_node("Reshape", [p + "ot", "s3"], [p + "o"]))
            lin(p + "o", "Wo", H, H, p + "a")
            nodes.append(oh.make_node("Add", [h, p + "a"], [p + "h1"])); h = p + "h1"
        if part in ("block", "mlp"):
            nodes.append(oh.make_node("LayerNormalization", [h, "g", "b"], [p + "n2"], axis=-1))
            lin(p + "n2", "W1", H, MLP, p + "m1")
            nodes.append(oh.make_node("Gelu", [p + "m1"], [p + "m2"]))
            lin(p + "m2", "W2", MLP, H, p + "m3")
            nodes.append(oh.make_node("Add", [h, p + "m3"], [p + "h2"])); h = p + "h2"
    for n, a in (("st", [0, 0, 0]), ("en", [1, 1, 8]), ("ax", [0, 1, 2])):
        init(n, np.array(a, np.int64))
    nodes.append(oh.make_node("Slice", [h, "st", "en", "ax"], ["y"]))
    g = oh.make_graph(nodes, f"bench_{part}_{mode}", [oh.make_tensor_value_info("x", tp, [1, 1, H])],
                      [oh.make_tensor_value_info("y", tp, [1, 1, 8])], inits)
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 20), oh.make_opsetid("com.microsoft", 1)])
    m.ir_version = 9
    path = os.path.join(out_dir, f"{part}_{mode}.onnx")
    onnx.save(m, path)
    print(f"{path}  {os.path.getsize(path) / 1e6:.0f}MB  nodes={len(nodes)}")


for part in ("block", "mlp", "attn"):
    for mode in ("fp32", "fp16", "best"):
        build(mode, part)
