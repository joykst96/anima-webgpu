# WebNN용 fp16 정적 그래프(webnn_fix.py 산출물)의 선형층 가중치를 int8(블록 32, 대칭 RTN)로 바꾼다 — 다운로드 크기용.
#   python -I export/webnn/webnn_q8.py <src.onnx> <dst.onnx>      (src 옆에 .data, dst 옆에 <dst>.data를 쓴다)
#
# 계산은 여전히 fp16이다. 가중치만 int8로 두고 그래프 안에서 풀어 쓴다("수동 역양자화"):
#   Wq int8 [K/32, 32, N] → Cast(fp32) → Mul(scale fp32 [K/32, 1, N]) → Cast(fp16) → Reshape([K, N]) → MatMul
# DequantizeLinear를 쓰지 않는 이유(2026-11-13 실측):
#   - 블록 단위 DequantizeLinear(axis=0, block_size=32)는 네이티브 DML EP는 받지만 Chrome 157 WebNN 백엔드가 세션 생성에서 거부한다.
#   - 열 단위(per-channel)는 ORT Web 1.26의 WebNN EP가 scale 랭크 오류를 낸다.
#   - Cast/Mul/Reshape만 쓰면 어디서나 받고, 속도도 fp16 가중치와 같다.
# 양자화 단위는 운영 그래프(MatMulNBits bits 8, block 32)와 같다: 입력 차원 K를 32개씩 묶고 출력 열마다 scale 하나.
import sys, os, numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto as TP

BLOCK, MIN_ELEMS = 32, 1 << 16
args = [a for a in sys.argv[1:] if not a.startswith("--")]
src, dst = args[0], args[1]
# --dq-cast: int8→fp32를 Cast 대신 DequantizeLinear(scale 1, 스칼라)로 한다. ORT의 상수 접기는 DequantizeLinear를 건너뛰므로
# 로드 때 wasm에서 가중치를 fp16으로 풀어 버리는 일(전체 모델에서 4GB 힙 초과 → std::bad_alloc)이 없다.
# 동적 그래프는 shape 계산을 접으려면 최적화를 켜야 해서 이 옵션이 필요하다(정적 그래프는 graphOptimizationLevel:"disabled"로도 됨).
DQ_CAST = "--dq-cast" in sys.argv
m = onnx.load(src)  # 외부 데이터까지 메모리로
g = m.graph
inits = {t.name: t for t in g.initializer}
users = {}
for n in g.node:
    for i in n.input:
        users.setdefault(i, []).append(n)

new_nodes, done, n_q, bytes_before, bytes_after = [], {}, 0, 0, 0
for n in g.node:
    w = inits.get(n.input[1]) if n.op_type == "MatMul" and len(n.input) == 2 else None
    if w is None or w.data_type != TP.FLOAT16 or len(w.dims) != 2 or w.dims[0] % BLOCK or w.dims[0] * w.dims[1] < MIN_ELEMS \
            or any(u.op_type != "MatMul" or u.input[1] != w.name for u in users[w.name]):
        new_nodes.append(n)
        continue
    if w.name not in done:
        K, N = w.dims
        W = numpy_helper.to_array(w).astype(np.float32).reshape(K // BLOCK, BLOCK, N)
        scale = np.abs(W).max(axis=1, keepdims=True) / 127.0          # [K/32, 1, N]
        scale[scale == 0] = 1.0
        q = np.clip(np.rint(W / scale), -127, 127).astype(np.int8)
        bytes_before += K * N * 2
        bytes_after += q.nbytes + scale.nbytes
        qn, sn, shp, out = w.name + "_q8", w.name + "_q8s", w.name + "_q8shape", w.name + "_deq"
        g.initializer.remove(w)
        g.initializer.extend([numpy_helper.from_array(q, qn), numpy_helper.from_array(scale.astype(np.float32), sn),
                              numpy_helper.from_array(np.array([K, N], np.int64), shp)])
        new_nodes += [helper.make_node("DequantizeLinear", [qn, "wq8_one", "wq8_zero"], [out + "_f32"], name=out + "_dq") if DQ_CAST else
                      helper.make_node("Cast", [qn], [out + "_f32"], to=TP.FLOAT, name=out + "_cast"),
                      helper.make_node("Mul", [out + "_f32", sn], [out + "_s"], name=out + "_mul"),
                      helper.make_node("Cast", [out + "_s"], [out + "_f16"], to=TP.FLOAT16, name=out + "_cast16"),
                      helper.make_node("Reshape", [out + "_f16", shp], [out], name=out + "_reshape")]
        done[w.name] = out
        n_q += 1
    n.input[1] = done[w.name]
    new_nodes.append(n)
del g.node[:]
g.node.extend(new_nodes)
if DQ_CAST:
    # 영점을 생략하면 ORT Web 1.26의 WebNN EP가 scale과 모양이 다른 기본 영점을 만들어 dequantizeLinear가 거부된다 → 명시
    g.initializer.extend([numpy_helper.from_array(np.array(1.0, np.float32), "wq8_one"), numpy_helper.from_array(np.array(0, np.int8), "wq8_zero")])
data = os.path.basename(dst) + ".data"
if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(dst)), data)):
    os.remove(os.path.join(os.path.dirname(os.path.abspath(dst)), data))
onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True, location=data, size_threshold=1024)
print(f"[webnn_q8] MatMul 가중치 {n_q}개를 int8(블록 {BLOCK})로: {bytes_before / 2**30:.2f} → {bytes_after / 2**30:.2f} GiB -> {dst}")
