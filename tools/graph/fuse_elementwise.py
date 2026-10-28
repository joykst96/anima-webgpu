# DiT 그래프의 분해된 원소별 연산을 ORT 융합 op로 바꾼다 (그래프만, 가중치 그대로, 재export 불필요).
#   GELU   : Div(x, √2) → Erf → Add(1) → Mul(0.5) → Mul(x)            → com.microsoft.Gelu(x)
#   RMSNorm: Pow(x,2) → ReduceMean(마지막 축) → Add(ε) → Sqrt → Reciprocal → Mul(x) → Mul(γ)
#                                                                       → SimplifiedLayerNormalization(x, γ, axis=-1, ε)
# 두 op 모두 ORT 1.26 WebGPU EP에 GPU 커널이 있음(fp32·fp16 확인). 수식은 같고 누산 순서만 달라 결과는 미세하게 바뀐다.
# fp16 구역 변환(fp16_zones.py)은 이 뒤에 적용한다.
#
#   python -I tools/graph/fuse_elementwise.py <src.onnx> <dst.onnx>
import sys, collections
import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
g = m.graph
init = {i.name: i for i in g.initializer}
prod = {o: n for n in g.node for o in n.output}
cons = collections.defaultdict(list)
for n in g.node:
    for i in n.input:
        cons[i].append(n)


def const(t):
    if t in init and init[t].data_location != onnx.TensorProto.EXTERNAL:
        return numpy_helper.to_array(init[t])
    p = prod.get(t)
    if p is not None and p.op_type == "Constant":
        return numpy_helper.to_array(p.attribute[0].t)
    return None


def scalar(t, v, tol=1e-5):
    c = const(t)
    return c is not None and c.size == 1 and abs(float(c.ravel()[0]) - v) < tol


def only(t):
    cs = cons[t]
    return cs[0] if len(cs) == 1 else None


def other(n, t):
    return n.input[1] if n.input[0] == t else n.input[0]


def rank_of(t, depth=0):
    """Reshape(상수 shape) / Transpose(perm) / 원소별 체인을 거슬러 올라가 rank를 구한다."""
    p = prod.get(t)
    if p is None or depth > 8:
        return None
    if p.op_type == "Reshape":
        c = const(p.input[1])
        if c is not None:
            return int(c.size)
        # 실행 중 Concat으로 만드는 shape — 입력이 모두 1원소(상수 1개 또는 Unsqueeze 결과)면 입력 수 = rank
        sp = prod.get(p.input[1])
        if sp is not None and sp.op_type == "Concat":
            def one(i):
                q = prod.get(i)
                if const(i) is not None:
                    return const(i).size == 1
                if q is None:
                    return False
                if q.op_type == "Unsqueeze":
                    return True
                return q.op_type == "Reshape" and const(q.input[1]) is not None and const(q.input[1]).tolist() in ([1], [-1])
            ok = all(one(i) for i in sp.input)
            return len(sp.input) if ok else None
        return None
    if p.op_type == "Transpose":
        perm = next((a.ints for a in p.attribute if a.name == "perm"), None)
        return len(perm) if perm else None
    if p.op_type in ("Mul", "Add", "Div", "Sub", "Cast"):
        return rank_of(p.input[0], depth + 1)
    return None


dead, new_nodes, n_gelu, n_rms, skipped = set(), {}, 0, 0, collections.Counter()

# GELU
for erf in [n for n in g.node if n.op_type == "Erf"]:
    div = prod.get(erf.input[0])
    if div is None or div.op_type != "Div" or not scalar(div.input[1], 2 ** 0.5):
        skipped["gelu:div"] += 1; continue
    x = div.input[0]
    add = only(erf.output[0])
    if add is None or add.op_type != "Add" or not scalar(other(add, erf.output[0]), 1.0):
        skipped["gelu:add"] += 1; continue
    m1 = only(add.output[0])
    if m1 is None or m1.op_type != "Mul":
        skipped["gelu:mul1"] += 1; continue
    m2 = only(m1.output[0])
    o1 = other(m1, add.output[0])
    if m2 is None or m2.op_type != "Mul":
        skipped["gelu:mul2"] += 1; continue
    o2 = other(m2, m1.output[0])
    if not ((scalar(o1, 0.5) and o2 == x) or (o1 == x and scalar(o2, 0.5))):
        skipped["gelu:shape"] += 1; continue
    if len(cons[div.output[0]]) != 1:
        skipped["gelu:fanout"] += 1; continue
    for n in (div, erf, add, m1, m2):
        dead.add(id(n))
    new_nodes[id(m2)] = helper.make_node("Gelu", [x], [m2.output[0]], domain="com.microsoft", name=m2.name + "_gelu")
    n_gelu += 1

# RMSNorm
for pw in [n for n in g.node if n.op_type == "Pow"]:
    if not scalar(pw.input[1], 2.0):
        skipped["rms:pow"] += 1; continue
    x = pw.input[0]
    rm = only(pw.output[0])
    if rm is None or rm.op_type != "ReduceMean":
        skipped["rms:mean"] += 1; continue
    axes = const(rm.input[1]) if len(rm.input) > 1 else None
    keep = next((a.i for a in rm.attribute if a.name == "keepdims"), 1)
    r = rank_of(x)
    if axes is None or axes.size != 1 or keep != 1 or r is None or int(axes.ravel()[0]) % r != r - 1:
        skipped["rms:axis"] += 1; continue
    add = only(rm.output[0])
    if add is None or add.op_type != "Add":
        skipped["rms:add"] += 1; continue
    eps = const(other(add, rm.output[0]))
    if eps is None or eps.size != 1:
        skipped["rms:eps"] += 1; continue
    sq = only(add.output[0])
    rc = only(sq.output[0]) if sq is not None and sq.op_type == "Sqrt" else None
    mx = only(rc.output[0]) if rc is not None and rc.op_type == "Reciprocal" else None
    if mx is None or mx.op_type != "Mul" or other(mx, rc.output[0]) != x:
        skipped["rms:mulx"] += 1; continue
    mg = only(mx.output[0])
    if mg is None or mg.op_type != "Mul":
        skipped["rms:mulg"] += 1; continue
    gamma = other(mg, mx.output[0])
    if gamma not in init:
        skipped["rms:gamma"] += 1; continue
    for n in (pw, rm, add, sq, rc, mx, mg):
        dead.add(id(n))
    new_nodes[id(mg)] = helper.make_node("SimplifiedLayerNormalization", [x, gamma], [mg.output[0]], axis=-1,
                                         epsilon=float(eps.ravel()[0]), stash_type=1, name=mg.name + "_rms")
    n_rms += 1

out = []
for n in g.node:
    if id(n) in new_nodes:
        out.append(new_nodes[id(n)])
    elif id(n) not in dead:
        out.append(n)
del g.node[:]
g.node.extend(out)
if not any(o.domain == "com.microsoft" for o in m.opset_import):
    m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
onnx.save(m, dst)
print(f"GELU {n_gelu}개 · RMSNorm {n_rms}개 융합, 노드 {len(out)}개 (건너뜀: {dict(skipped)}) → {dst}")
