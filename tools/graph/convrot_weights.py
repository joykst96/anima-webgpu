"""회전(ConvRot: 그룹 Hadamard) + 정수 가중치 양자화의 효과를 실제 Anima Turbo DiT에서 잰다 (가중치 양자화만, 계산은 fp16).
  rot_exp.py <fixed static fp16 graph.onnx> <feeds dir> <schemes: 예 blk32-8,row-8,rowrot256-8,...>
가중치를 양자화→역양자화(→역회전)한 fp16 값으로 바꿔 끼워(add_external_initializers, 디스크에 안 씀) 네이티브 DML로 DiT 1회 출력을
PyTorch fp32 기준(ref_s*.f32)과 비교한다. 대상 가중치는 export/webnn/webnn_q8.py와 같은 453개.
  blk32     : 입력축 32개 블록 × 출력 열마다 scale (지금 방식)
  row       : 출력 열마다 scale 하나 (ConvRot int8의 '행 단위' — 우리 [K,N] 배치에선 열)
  rot<G>    : 접두로 붙이면 양자화 전에 입력축을 G개씩 묶어 정규 Hadamard로 회전, 역양자화 뒤 되돌림 (K가 G의 배수일 때만)
"""
import sys, os, time, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper, TensorProto as TP
gpath, fd, schemes = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
m = onnx.load(gpath); g = m.graph
inits = {t.name: t for t in g.initializer}; users = {}
for n in g.node:
    for i in n.input: users.setdefault(i, []).append(n)
targets = {}
for n in g.node:
    w = inits.get(n.input[1]) if n.op_type == "MatMul" and len(n.input) == 2 else None
    if w is None or w.data_type != TP.FLOAT16 or len(w.dims) != 2 or w.dims[0] % 32 or w.dims[0] * w.dims[1] < (1 << 16): continue
    if any(u.op_type != "MatMul" or u.input[1] != w.name for u in users[w.name]): continue
    targets[w.name] = numpy_helper.to_array(w).astype(np.float32)
del m
print(f"대상 가중치 {len(targets)}개, {sum(w.size for w in targets.values()) / 1e9:.2f}B 파라미터", flush=True)

def hadamard(G):
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], np.float32); h = h4
    while h.shape[0] < G: h = np.kron(h, h4)
    assert h.shape[0] == G, "정규 Hadamard는 4의 거듭제곱 크기만"
    return h / np.sqrt(G)
def rot(W, H):  # [K,N] → 입력축을 G개씩 묶어 H 적용 (H 대칭·직교 → 같은 함수가 역변환)
    G = H.shape[0]; K, N = W.shape
    return (H @ W.reshape(K // G, G, N)).reshape(K, N)
def q_blk(W, qmax, B=32):
    K, N = W.shape; Wb = W.reshape(K // B, B, N)
    s = np.abs(Wb).max(axis=1, keepdims=True) / qmax; s[s == 0] = 1
    return (np.clip(np.rint(Wb / s), -qmax, qmax) * s).reshape(K, N), (K // B) * N
def q_row(W, qmax):
    s = np.abs(W).max(axis=0, keepdims=True) / qmax; s[s == 0] = 1
    return np.clip(np.rint(W / s), -qmax, qmax) * s, W.shape[1]

lat = np.fromfile(os.path.join(fd, "feed_latent.f32"), np.float32).astype(np.float16).reshape(1, 16, 1, 128, 128)
ctx = np.fromfile(os.path.join(fd, "feed_context.f32"), np.float32).astype(np.float16).reshape(1, 512, 1024)
refs = {s: np.fromfile(os.path.join(fd, f"ref_s{s}.f32"), np.float32) for s in ("1.0", "0.5", "0.1")}
def evaluate(new):
    so = ort.SessionOptions(); so.log_severity_level = 4
    if new:
        names = list(new); vals = [ort.OrtValue.ortvalue_from_numpy(new[k]) for k in names]
        so.add_external_initializers(names, vals)
    s = ort.InferenceSession(gpath, so, providers=["DmlExecutionProvider"]); assert s.get_providers()[0].startswith("Dml")
    out = []
    for sg, ref in refs.items():
        o = s.run(None, {"latent": lat, "timesteps": np.array([float(sg)], np.float32), "context": ctx})[0].astype(np.float32).ravel()
        out.append(100 * np.linalg.norm(o - ref) / np.linalg.norm(ref))
    return out
print(f"{'방식':16s} {'가중치 오차':>10s}  {'DiT 오차 σ1.0/0.5/0.1':>24s}  {'크기(GiB)':>9s}  회전 적용", flush=True)
for sc in schemes:
    t0 = time.time()
    if sc == "fp16":
        e = evaluate(None); print(f"{'fp16 (원본)':16s} {'-':>10s}  {e[0]:7.3f} /{e[1]:6.3f} /{e[2]:6.3f}  {sum(w.size for w in targets.values()) * 2 / 2**30:9.2f}", flush=True); continue
    name, bits = sc.split("-"); bits = int(bits); qmax = 2 ** (bits - 1) - 1
    G = int(name.split("rot")[1]) if "rot" in name else 0; base = name.split("rot")[0] if G else name
    H = hadamard(G) if G else None
    new, num, den, nbytes, nrot = {}, 0.0, 0.0, 0, 0
    for k, W in targets.items():
        use = G and W.shape[0] % G == 0
        Wr = rot(W, H) if use else W
        Wq, nsc = q_blk(Wr, qmax) if base == "blk32" else q_row(Wr, qmax)
        Wh = rot(Wq, H) if use else Wq
        num += float(((Wh - W) ** 2).sum()); den += float((W ** 2).sum()); nbytes += W.size * bits / 8 + nsc * 4; nrot += bool(use)
        new[k] = Wh.astype(np.float16)
    e = evaluate(new)
    print(f"{sc:16s} {100 * np.sqrt(num / den):9.3f}%  {e[0]:7.3f} /{e[1]:6.3f} /{e[2]:6.3f}  {nbytes / 2**30:9.2f}  {nrot}/{len(targets)}  ({time.time() - t0:.0f}s)", flush=True)
