"""활성 int8 양자화(행=토큰 단위 scale, DP4A·ConvRot W8A8과 같은 단위라고 가정)의 오차가 회전으로 얼마나 줄어드는지,
실제 DiT의 선형층 입력으로 층 단위에서 잰다. 가중치는 정확(fp32)하게 두고 활성만 양자화한다.
  rot_act.py <fixed static fp16 graph> <feeds dir> <sigma>
"""
import sys, os, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper, shape_inference, TensorProto as TP
gpath, fd, sigma = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = os.path.dirname(os.path.abspath(gpath))
m = onnx.load(gpath, load_external_data=False)
inf = shape_inference.infer_shapes(m); vi = {v.name: v for v in inf.graph.value_info}
g = m.graph; inits = {t.name: t for t in g.initializer}
cand = [n for n in g.node if n.op_type == "MatMul" and len(n.input) == 2 and n.input[1] in inits and len(inits[n.input[1]].dims) == 2
        and inits[n.input[1]].dims[0] % 256 == 0 and inits[n.input[1]].dims[0] * inits[n.input[1]].dims[1] >= (1 << 20) and n.input[0] in vi]
pick = cand[::max(1, len(cand) // 36)]
g.output.extend([vi[n.input[0]] for n in pick if n.input[0] not in {o.name for o in g.output}])
p = os.path.join(d, "_act.onnx"); onnx.save(m, p)
lat = np.fromfile(os.path.join(fd, "feed_latent.f32"), np.float32).astype(np.float16).reshape(1, 16, 1, 128, 128)
ctx = np.fromfile(os.path.join(fd, "feed_context.f32"), np.float32).astype(np.float16).reshape(1, 512, 1024)
so = ort.SessionOptions(); so.log_severity_level = 4
s = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
outs = dict(zip([o.name for o in s.get_outputs()], s.run(None, {"latent": lat, "timesteps": np.array([sigma], np.float32), "context": ctx})))
os.remove(p)
full = onnx.load(gpath); W = {t.name: t for t in full.graph.initializer}
def hadamard(G):
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], np.float32); h = h4
    while h.shape[0] < G: h = np.kron(h, h4)
    return h / np.sqrt(G)
def qrow(x, qmax=127):
    sc = np.abs(x).max(axis=1, keepdims=True) / qmax; sc[sc == 0] = 1
    return np.clip(np.rint(x / sc), -qmax, qmax) * sc
rows = []
for n in pick:
    x = outs[n.input[0]].astype(np.float32); K = x.shape[-1]; x = x.reshape(-1, K)
    w = numpy_helper.to_array(W[n.input[1]]).astype(np.float32); y = x @ w; ny = np.linalg.norm(y)
    r = {"K": K, "N": w.shape[1], "rows": x.shape[0], "kurt": float(((x - x.mean()) ** 4).mean() / (x.var() ** 2 + 1e-30)), "crest": float(np.median(np.abs(x).max(axis=1) / (x.std(axis=1) + 1e-12)))}
    r["plain"] = 100 * np.linalg.norm((qrow(x) - x) @ w) / ny
    for G in (16, 64, 256):
        H = hadamard(G); xr = (x.reshape(-1, K // G, G) @ H).reshape(-1, K); wr = (H @ w.reshape(K // G, G, -1)).reshape(K, -1)
        r[f"rot{G}"] = 100 * np.linalg.norm((qrow(xr) - xr) @ wr) / ny
    rows.append(r)
print(f"층 {len(rows)}개 (σ={sigma}). 활성만 int8(행 단위)로 했을 때 그 층 출력의 상대 오차 %")
print(f"{'K→N':>12s} {'토큰':>5s} {'최대/표준편차':>12s} {'회전 없음':>9s} {'rot16':>7s} {'rot64':>7s} {'rot256':>7s}")
for r in rows: print(f"{r['K']:5d}→{r['N']:<5d} {r['rows']:5d} {r['crest']:12.1f} {r['plain']:9.3f} {r['rot16']:7.3f} {r['rot64']:7.3f} {r['rot256']:7.3f}")
for k in ("plain", "rot16", "rot64", "rot256"): print(f"  {k:7s} 중앙값 {np.median([r[k] for r in rows]):.3f}%  평균 {np.mean([r[k] for r in rows]):.3f}%  최대 {np.max([r[k] for r in rows]):.3f}%")
