# DiT 그래프(가중치 외부 참조 그대로)의 MHA 노드만 바꾼 실험용 변형을 만든다.
#   unfused : MHA → Reshape/Transpose/MatMul/Softmax/MatMul (fuse_attention 이전 형태 근사, fp32)
#   mhaf16  : MHA 입력 q/k/v를 fp16으로 Cast, 출력을 fp32로 Cast (FlashAttention을 fp16으로)
# 외부 데이터 위치(location)는 그대로 → 페이지가 넘기는 샤드 externalData와 맞물린다.
import sys, onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

src, mode, dst = sys.argv[1], sys.argv[2], sys.argv[3]
m = onnx.load(src, load_external_data=False)
g = m.graph
new_nodes, inits = [], []

def const_i64(name, vals):
    inits.append(numpy_helper.from_array(np.array(vals, dtype=np.int64), name))
    return name

n_mha = 0
for n in g.node:
    if n.op_type != "MultiHeadAttention":
        new_nodes.append(n)
        continue
    n_mha += 1
    attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
    H = attrs["num_heads"]; scale = attrs["scale"]
    q, k, v = n.input[0], n.input[1], n.input[2]
    assert len([i for i in n.input if i]) == 3, n.input
    out = n.output[0]
    p = n.name + "_x_"
    if mode == "mhaf16":
        for nm in (q, k, v):
            new_nodes.append(helper.make_node("Cast", [nm], [p + nm + "_h"], to=TensorProto.FLOAT16, name=p + nm + "_cast"))
        new_nodes.append(helper.make_node("MultiHeadAttention", [p + q + "_h", p + k + "_h", p + v + "_h"], [p + "o_h"],
                                          domain="com.microsoft", name=n.name, num_heads=H, scale=scale))
        new_nodes.append(helper.make_node("Cast", [p + "o_h"], [out], to=TensorProto.FLOAT, name=p + "o_cast"))
    elif mode == "unfused":
        sh4 = const_i64(p + "sh4", [0, 0, H, -1])
        sh3 = const_i64(p + "sh3", [0, 0, -1])
        new_nodes += [
            helper.make_node("Reshape", [q, sh4], [p + "q4"], name=p + "rq"),
            helper.make_node("Transpose", [p + "q4"], [p + "qt"], perm=[0, 2, 1, 3], name=p + "tq"),
            helper.make_node("Reshape", [k, sh4], [p + "k4"], name=p + "rk"),
            helper.make_node("Transpose", [p + "k4"], [p + "kt"], perm=[0, 2, 3, 1], name=p + "tk"),
            helper.make_node("Reshape", [v, sh4], [p + "v4"], name=p + "rv"),
            helper.make_node("Transpose", [p + "v4"], [p + "vt"], perm=[0, 2, 1, 3], name=p + "tv"),
            helper.make_node("MatMul", [p + "qt", p + "kt"], [p + "s"], name=p + "qk"),
        ]
        inits.append(numpy_helper.from_array(np.array(scale, dtype=np.float32), p + "scale"))
        new_nodes += [
            helper.make_node("Mul", [p + "s", p + "scale"], [p + "ss"], name=p + "sc"),
            helper.make_node("Softmax", [p + "ss"], [p + "pr"], axis=-1, name=p + "sm"),
            helper.make_node("MatMul", [p + "pr", p + "vt"], [p + "o4"], name=p + "pv"),
            helper.make_node("Transpose", [p + "o4"], [p + "ot"], perm=[0, 2, 1, 3], name=p + "to"),
            helper.make_node("Reshape", [p + "ot", sh3], [out], name=p + "ro"),
        ]
    else:
        raise SystemExit("mode?")

del g.node[:]
g.node.extend(new_nodes)
g.initializer.extend(inits)
onnx.save(m, dst)
print(mode, "replaced", n_mha, "MHA ->", dst, len(g.node), "nodes")
