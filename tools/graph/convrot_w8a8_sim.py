"""W8A8(가중치 int8 블록 32 + 활성 int8 행 단위)를 그래프 안의 가짜 양자화로 흉내 내고, 회전(ConvRot식 그룹 Hadamard) 유무로
DiT 1회 출력 오차를 비교한다. 네이티브 DML로 실행, 기준 = PyTorch fp32.
  rot_a8.py <fixed static fp16 graph> <feeds dir> <variants: w8,plain,rot16,rot64,rot256[,…]> [abits=8]
  w8    : 가중치만 int8 (활성은 fp16 그대로)          plain : + 활성 int8, 회전 없음 (지금 WebGPU 경로에 해당한다고 가정)
  rot<G>: 활성을 G개씩 Hadamard 회전한 뒤 int8, 가중치는 회전한 뒤 int8
가정: 활성 scale은 행(토큰)마다 하나(absmax/127). ORT DP4A 커널의 실제 단위는 확인하지 않았다.
"""
import sys, os, numpy as np, onnx, onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference, TensorProto as TP
gpath, fd, variants = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
AQ = 2 ** (int(sys.argv[4]) - 1) - 1 if len(sys.argv) > 4 else 127
d = os.path.dirname(os.path.abspath(gpath))
full = onnx.load(gpath); WT = {}
users = {}
for n in full.graph.node:
    for i in n.input: users.setdefault(i, []).append(n)
ii = {t.name: t for t in full.graph.initializer}
for n in full.graph.node:
    w = ii.get(n.input[1]) if n.op_type == "MatMul" and len(n.input) == 2 else None
    if w is None or w.data_type != TP.FLOAT16 or len(w.dims) != 2 or w.dims[0] % 32 or w.dims[0] * w.dims[1] < (1 << 16): continue
    if any(u.op_type != "MatMul" or u.input[1] != w.name for u in users[w.name]): continue
    WT[w.name] = numpy_helper.to_array(w).astype(np.float32)
del full, ii, users
def hadamard(G):
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], np.float32); h = h4
    while h.shape[0] < G: h = np.kron(h, h4)
    return h / np.sqrt(G)
def q_blk(W, qmax=127, B=32):
    K, N = W.shape; Wb = W.reshape(K // B, B, N); s = np.abs(Wb).max(axis=1, keepdims=True) / qmax; s[s == 0] = 1
    return (np.clip(np.rint(Wb / s), -qmax, qmax) * s).reshape(K, N)
lat = np.fromfile(os.path.join(fd, "feed_latent.f32"), np.float32).astype(np.float16).reshape(1, 16, 1, 128, 128)
ctx = np.fromfile(os.path.join(fd, "feed_context.f32"), np.float32).astype(np.float16).reshape(1, 512, 1024)
refs = {s: np.fromfile(os.path.join(fd, f"ref_s{s}.f32"), np.float32) for s in ("1.0", "0.5", "0.1")}
print(f"{'변형':10s} {'DiT 오차 σ1.0 / 0.5 / 0.1':>26s}   회전·양자화한 입력 수, 스텝 시간", flush=True)
for v in variants:
    G = int(v[3:]) if v.startswith("rot") else 0; H = hadamard(G) if G else None
    m = onnx.load(gpath, load_external_data=False); g = m.graph
    inf = shape_inference.infer_shapes(m); rank = {x.name: len(x.type.tensor_type.shape.dim) for x in list(inf.graph.value_info) + list(inf.graph.input)}
    shp = {x.name: [q.dim_value for q in x.type.tensor_type.shape.dim] for x in list(inf.graph.value_info) + list(inf.graph.input)}
    new_nodes, new_inits, fq, new_w, nrot, nskip = [], [], {}, {}, 0, 0
    new_inits += [numpy_helper.from_array(np.array([-1], np.int64), "fq_axis"), numpy_helper.from_array(np.array(1.0 / AQ, np.float32), "fq_inv"), numpy_helper.from_array(np.array(1e-20, np.float32), "fq_eps")]
    for n in g.node:
        if n.op_type == "MatMul" and len(n.input) == 2 and n.input[1] in WT:
            W = WT[n.input[1]]; K = W.shape[0]; x = n.input[0]
            use = bool(G) and K % G == 0 and rank.get(x) == 3
            # 운영 그래프와 같게: 타임스텝 임베딩 2개 층(토큰 1개, K·N 모두 1024 이상)은 활성을 양자화하지 않는다(--fp32-t-embed)
            t_embed = shp.get(x, [0, 0])[-2] == 1 and K >= 1024 and W.shape[1] >= 1024
            if t_embed: use = False; nskip += 1
            if v != "w8" and not t_embed:
                key = (x, use)
                if key not in fq:
                    p = f"fq{len(fq)}_"; cur = x
                    if use:
                        if f"rs_{K}" not in {t.name for t in new_inits}:
                            new_inits += [numpy_helper.from_array(np.array([0, 0, K // G, G], np.int64), f"rs_{K}"), numpy_helper.from_array(np.array([0, 0, K], np.int64), f"rb_{K}")]
                        if "fq_H" not in {t.name for t in new_inits}: new_inits.append(numpy_helper.from_array(H.astype(np.float16), "fq_H"))
                        new_nodes += [helper.make_node("Reshape", [cur, f"rs_{K}"], [p + "g"]), helper.make_node("MatMul", [p + "g", "fq_H"], [p + "gr"]), helper.make_node("Reshape", [p + "gr", f"rb_{K}"], [p + "r"])]
                        cur = p + "r"; nrot += 1
                    new_nodes += [helper.make_node("Cast", [cur], [p + "f"], to=TP.FLOAT), helper.make_node("Abs", [p + "f"], [p + "a"]),
                                  helper.make_node("ReduceMax", [p + "a", "fq_axis"], [p + "mx"], keepdims=1), helper.make_node("Mul", [p + "mx", "fq_inv"], [p + "s0"]),
                                  helper.make_node("Add", [p + "s0", "fq_eps"], [p + "s"]), helper.make_node("Div", [p + "f", p + "s"], [p + "d"]),
                                  helper.make_node("Round", [p + "d"], [p + "rd"]), helper.make_node("Mul", [p + "rd", p + "s"], [p + "q"]),
                                  helper.make_node("Cast", [p + "q"], [p + "o"], to=TP.FLOAT16)]
                    fq[key] = p + "o"
                n.input[0] = fq[key]
            wname = n.input[1] + ("_rot" if use else "_q")       # 같은 가중치가 회전/비회전 양쪽으로 쓰일 일은 없지만 이름은 나눠 둔다
            if wname not in new_w:
                Wr = (H @ W.reshape(K // G, G, -1)).reshape(K, -1) if use else W
                new_w[wname] = q_blk(Wr).astype(np.float16)
                new_inits.append(numpy_helper.from_array(new_w[wname], wname))
            n.input[1] = wname
        new_nodes.append(n)
    del g.node[:]; g.node.extend(new_nodes); g.initializer.extend(new_inits)
    p = os.path.join(d, "_a8.onnx"); onnx.save(m, p, save_as_external_data=True, all_tensors_to_one_file=True, location="_a8.onnx.data", size_threshold=1 << 20, convert_attribute=False)
    so = ort.SessionOptions(); so.log_severity_level = 4
    s = ort.InferenceSession(p, so, providers=["DmlExecutionProvider"]); assert s.get_providers()[0].startswith("Dml"), s.get_providers()
    errs, ms = [], []
    import time
    for sg, ref in refs.items():
        for _ in range(2):
            t = time.perf_counter(); o = s.run(None, {"latent": lat, "timesteps": np.array([float(sg)], np.float32), "context": ctx})[0]; ms.append((time.perf_counter() - t) * 1000)
        o = o.astype(np.float32).ravel(); errs.append(100 * np.linalg.norm(o - ref) / np.linalg.norm(ref))
    print(f"{v:10s} {errs[0]:8.3f} /{errs[1]:7.3f} /{errs[2]:7.3f}     입력 {len(fq)}개 중 회전 {nrot}, t_embed 제외 {nskip}, {min(ms):.0f}ms", flush=True)
    del s
    for f in ("_a8.onnx", "_a8.onnx.data"):
        try: os.remove(os.path.join(d, f))
        except OSError: pass
