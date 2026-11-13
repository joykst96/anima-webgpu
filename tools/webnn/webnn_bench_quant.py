"""양자화 변형 테스트 그래프 (MLP 4블록·4096토큰, tools/webnn/webnn_bench.py와 같은 뼈대, 활성 fp16).
  dq8b  : 가중치 int8 + DequantizeLinear(axis=0, block_size=32, scale fp16) → fp16 MatMul   (블록 32 = 지금 운영 양자화와 같은 단위)
  dq8c  : 가중치 int8 + DequantizeLinear(axis=1, 열별 scale fp16) → fp16 MatMul             (블록 없는 per-channel)
  dq4b  : 가중치 int4 + DequantizeLinear(블록 32)
  mmi   : 정수 연산 — 활성을 DynamicQuantizeLinear(uint8)로, MatMulInteger(uint8 × int8) → int32 → float (열별 scale)
  qdq   : QDQ 형식 — 활성 QuantizeLinear→DequantizeLinear, 가중치 DequantizeLinear, MatMul, 출력 Q→DQ (백엔드가 정수 커널로 융합하는지)
"""
import sys, os, numpy as np, onnx
from onnx import helper as oh, numpy_helper as nh, TensorProto as TP
H, MLP, TOK, BLOCKS = 2048, 8192, 4096, 4
out_dir = sys.argv[1]; rng = np.random.default_rng(0)

def build(mode):
    nodes, inits, made = [], [], set()
    def init(name, arr):
        if name not in made: inits.append(nh.from_array(arr, name)); made.add(name)
        return name
    def raw_int4(name, arr):  # int4는 numpy에 없어 직접 채운다(2개씩 한 바이트)
        if name in made: return name
        flat = (arr.astype(np.int8).ravel() & 0xF).astype(np.uint8)
        if flat.size % 2: flat = np.append(flat, 0)
        t = onnx.TensorProto(name=name, data_type=TP.INT4, dims=list(arr.shape), raw_data=(flat[0::2] | (flat[1::2] << 4)).tobytes())
        inits.append(t); made.add(name); return name
    def lin(x, w, K, N, out):
        if mode in ("dq8b", "dq4b"):
            if mode == "dq8b": init(w + "_q", np.clip(np.round(rng.normal(0, 30, (K, N))), -127, 127).astype(np.int8)); sc = 0.02 / 30
            else: raw_int4(w + "_q", np.clip(np.round(rng.normal(0, 2.5, (K, N))), -8, 7)); sc = 0.02 / 2.5
            init(w + "_s", np.full((K // 32, N), sc, np.float16))
            nodes.append(oh.make_node("DequantizeLinear", [w + "_q", w + "_s"], [w + "_dq" + out], axis=0, block_size=32))
            nodes.append(oh.make_node("MatMul", [x, w + "_dq" + out], [out]))
        elif mode == "dqm8":   # 수동 역양자화: Cast(int8→fp16) → [K/32,32,N] × scale[K/32,1,N] → [K,N]. DequantizeLinear 없이 기본 연산만
            init(w + "_q", np.clip(np.round(rng.normal(0, 30, (K // 32, 32, N))), -127, 127).astype(np.int8))
            init(w + "_s", np.full((K // 32, 1, N), 0.02 / 30, np.float16)); init(w + "_shp", np.array([K, N], np.int64))
            nodes.append(oh.make_node("Cast", [w + "_q"], [w + "_c" + out], to=TP.FLOAT16))
            nodes.append(oh.make_node("Mul", [w + "_c" + out, w + "_s"], [w + "_m" + out]))
            nodes.append(oh.make_node("Reshape", [w + "_m" + out, w + "_shp"], [w + "_dq" + out]))
            nodes.append(oh.make_node("MatMul", [x, w + "_dq" + out], [out]))
        elif mode == "dq8bf":  # 블록 DequantizeLinear인데 scale fp32 → 출력 fp32 → Cast fp16
            init(w + "_q", np.clip(np.round(rng.normal(0, 30, (K, N))), -127, 127).astype(np.int8))
            init(w + "_s", np.full((K // 32, N), 0.02 / 30, np.float32))
            nodes.append(oh.make_node("DequantizeLinear", [w + "_q", w + "_s"], [w + "_d32" + out], axis=0, block_size=32))
            nodes.append(oh.make_node("Cast", [w + "_d32" + out], [w + "_dq" + out], to=TP.FLOAT16))
            nodes.append(oh.make_node("MatMul", [x, w + "_dq" + out], [out]))
        elif mode == "dq8c":
            init(w + "_q", np.clip(np.round(rng.normal(0, 30, (K, N))), -127, 127).astype(np.int8))
            init(w + "_s", np.full((N,), 0.02 / 30, np.float16))
            nodes.append(oh.make_node("DequantizeLinear", [w + "_q", w + "_s"], [w + "_dq" + out], axis=1))
            nodes.append(oh.make_node("MatMul", [x, w + "_dq" + out], [out]))
        elif mode == "mmi":
            init(w + "_q", np.clip(np.round(rng.normal(0, 30, (K, N))), -127, 127).astype(np.int8))
            init(w + "_s", np.full((N,), 0.02 / 30, np.float32))
            nodes.append(oh.make_node("Cast", [x], [out + "_x32"], to=TP.FLOAT))
            nodes.append(oh.make_node("DynamicQuantizeLinear", [out + "_x32"], [out + "_xq", out + "_xs", out + "_xz"]))
            nodes.append(oh.make_node("MatMulInteger", [out + "_xq", w + "_q", out + "_xz"], [out + "_i32"]))
            nodes.append(oh.make_node("Cast", [out + "_i32"], [out + "_f32"], to=TP.FLOAT))
            nodes.append(oh.make_node("Mul", [out + "_f32", out + "_xs"], [out + "_m1"]))
            nodes.append(oh.make_node("Mul", [out + "_m1", w + "_s"], [out + "_m2"]))
            nodes.append(oh.make_node("Cast", [out + "_m2"], [out], to=TP.FLOAT16))
        elif mode == "qdq":
            init(w + "_q", np.clip(np.round(rng.normal(0, 30, (K, N))), -127, 127).astype(np.int8))
            init(w + "_s", np.array(0.02 / 30, np.float32)); init(w + "_z", np.array(0, np.int8))
            init("as", np.array(8 / 127, np.float32)); init("az", np.array(0, np.int8))   # 활성 범위 ±8 고정(실험용)
            nodes.append(oh.make_node("Cast", [x], [out + "_x32"], to=TP.FLOAT))
            nodes.append(oh.make_node("QuantizeLinear", [out + "_x32", "as", "az"], [out + "_xq"]))
            nodes.append(oh.make_node("DequantizeLinear", [out + "_xq", "as", "az"], [out + "_xd"]))
            nodes.append(oh.make_node("DequantizeLinear", [w + "_q", w + "_s", w + "_z"], [w + "_dq" + out]))
            nodes.append(oh.make_node("MatMul", [out + "_xd", w + "_dq" + out], [out + "_y"]))
            nodes.append(oh.make_node("QuantizeLinear", [out + "_y", "as", "az"], [out + "_yq"]))
            nodes.append(oh.make_node("DequantizeLinear", [out + "_yq", "as", "az"], [out + "_yd"]))
            nodes.append(oh.make_node("Cast", [out + "_yd"], [out], to=TP.FLOAT16))
        return out
    ft = np.float16
    init("base", rng.normal(0, 1, (1, TOK, H)).astype(ft)); init("g", np.ones(H, ft)); init("b", np.zeros(H, ft))
    nodes.append(oh.make_node("Add", ["base", "x"], ["h0"])); h = "h0"
    for i in range(BLOCKS):
        p = f"b{i}_"
        nodes.append(oh.make_node("LayerNormalization", [h, "g", "b"], [p + "n2"], axis=-1))
        lin(p + "n2", "W1", H, MLP, p + "m1"); nodes.append(oh.make_node("Gelu", [p + "m1"], [p + "m2"]))
        lin(p + "m2", "W2", MLP, H, p + "m3"); nodes.append(oh.make_node("Add", [h, p + "m3"], [p + "h2"])); h = p + "h2"
    for n, a in (("st", [0, 0, 0]), ("en", [1, 1, 8]), ("ax", [0, 1, 2])): init(n, np.array(a, np.int64))
    nodes.append(oh.make_node("Slice", [h, "st", "en", "ax"], ["y"]))
    g = oh.make_graph(nodes, "q_" + mode, [oh.make_tensor_value_info("x", TP.FLOAT16, [1, 1, H])], [oh.make_tensor_value_info("y", TP.FLOAT16, [1, 1, 8])], inits)
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 21)]); m.ir_version = 10
    p = os.path.join(out_dir, f"mlp_{mode}.onnx"); onnx.save(m, p); print(p, round(os.path.getsize(p) / 1e6), "MB")
for mode in sys.argv[2].split(","): build(mode)
