# 같은 입력을 받는 MatMulNBits 여러 개를 N 방향으로 하나로 합친다 (그래프만, 가중치 그대로).
#   python -I tools/graph/fuse_matmulnbits.py <src.onnx> <dst.onnx> [self] [adaln] [cross]
#     self  : self-attention q/k/v 3개 → N=6144 1개 (블록당)
#     adaln : timestep 임베딩(silu) 하나를 받는 adaLN 선형층 85개 → 1개
#     cross : context(와 negpip context)를 받는 cross-attention K/V 투영 28개씩 → 1개씩
# 가중치(B)와 scales는 Concat 노드로 이어 붙인다 — 입력이 전부 initializer라 ORT가 세션 생성 때 상수로 접는다.
# 합친 출력은 Split으로 원래 이름에 돌려준다. MatMulNBits는 행(N)마다 독립이라 수학적으로 같은 계산이다
# (활성 int8 양자화도 입력 행 블록 단위라 같은 입력이면 같은 값).
import sys, collections, onnx
import numpy as np
from onnx import helper, numpy_helper

src, dst, *kinds = sys.argv[1:]
kinds = set(kinds or ["self"])
m = onnx.load(src, load_external_data=False)
g = m.graph
cons = collections.defaultdict(list)
for n in g.node:
    for i in n.input:
        cons[i].append(n)
attr = lambda n, k: next(helper.get_attribute_value(a) for a in n.attribute if a.name == k)


def kind_of(inp, mm):
    if inp == "silu_1":
        return "adaln"
    if attr(mm[0], "K") == 1024:
        return "cross"
    return "self" if len(mm) == 3 else None


fused, drop, new_inits = [], set(), []
for inp, ns in list(cons.items()):
    mm = [n for n in ns if n.op_type == "MatMulNBits"]
    if len(mm) < 2 or kind_of(inp, mm) not in kinds:
        continue
    keys = {(attr(n, "K"), attr(n, "bits"), attr(n, "block_size"), attr(n, "accuracy_level"), len(n.input)) for n in mm}
    assert len(keys) == 1 and len(mm[0].input) == 3, (inp, keys)
    Ns = [attr(n, "N") for n in mm]
    tag = f"fz_{mm[0].name}"
    fused.append(helper.make_node("Concat", [n.input[1] for n in mm], [tag + "_B"], axis=0, name=tag + "_catB"))
    fused.append(helper.make_node("Concat", [n.input[2] for n in mm], [tag + "_S"], axis=0, name=tag + "_catS"))
    a = {x.name: helper.get_attribute_value(x) for x in mm[0].attribute}
    a["N"] = sum(Ns)
    fused.append(helper.make_node("MatMulNBits", [inp, tag + "_B", tag + "_S"], [tag + "_Y"], domain="com.microsoft", name=tag, **a))
    if len(mm) <= 8:
        new_inits.append(numpy_helper.from_array(np.array(Ns, dtype=np.int64), tag + "_split"))
        fused.append(helper.make_node("Split", [tag + "_Y", tag + "_split"], [n.output[0] for n in mm], axis=-1, name=tag + "_split"))
    else:  # WebGPU Split은 출력마다 storage buffer를 써서 한 셰이더당 한도(8~10)를 넘으면 실패 → 출력별 Slice
        new_inits.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), tag + "_ax"))
        off = 0
        for k, (n, w) in enumerate(zip(mm, Ns)):
            new_inits.append(numpy_helper.from_array(np.array([off], dtype=np.int64), f"{tag}_s{k}"))
            new_inits.append(numpy_helper.from_array(np.array([off + w], dtype=np.int64), f"{tag}_e{k}"))
            fused.append(helper.make_node("Slice", [tag + "_Y", f"{tag}_s{k}", f"{tag}_e{k}", tag + "_ax"], [n.output[0]], name=f"{tag}_slice{k}"))
            off += w
    drop.update(id_ for id_ in (n.name for n in mm))
    print(kind_of(inp, mm), inp, len(mm), "→ N", sum(Ns))

nodes = [n for n in g.node if n.name not in drop] + fused
avail = set([i.name for i in g.input] + [t.name for t in g.initializer] + [t.name for t in new_inits])
order, pending = [], nodes
while pending:
    rest = []
    for n in pending:
        if all((not i) or i in avail for i in n.input):
            order.append(n)
            avail.update(n.output)
        else:
            rest.append(n)
    if len(rest) == len(pending):
        raise SystemExit(f"정렬 실패 {len(rest)}")
    pending = rest
del g.node[:]
g.node.extend(order)
g.initializer.extend(new_inits)
onnx.save(m, dst)
print("nodes", len(order))
