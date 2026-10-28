# ONNX 그래프 요약: 입출력, op 개수, 주요 op 속성(MatMulNBits/MHA/LayerNorm), Cast 대상 dtype.
#   python -I tools/graph/info.py <그래프.onnx>... (가중치 없이 그래프만 로드)
import sys, collections, onnx
from onnx import helper

for p in sys.argv[1:]:
    m = onnx.load(p, load_external_data=False)
    g = m.graph
    print("=" * 80)
    print(p.split("\\")[-1].split("/")[-1], "opset", [(o.domain, o.version) for o in m.opset_import], "ir", m.ir_version, "producer", m.producer_name, m.producer_version)
    def dims(t):
        return [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]
    for i in g.input:
        print("  in ", i.name, onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type), dims(i))
    for o in g.output:
        print("  out", o.name, onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type), dims(o))
    c = collections.Counter((n.domain or "ai.onnx") + ":" + n.op_type for n in g.node)
    print("  nodes", len(g.node), "initializers", len(g.initializer))
    for k, v in c.most_common():
        print(f"    {v:6d} {k}")
    # attrs of interesting ops
    seen = set()
    for n in g.node:
        if n.op_type in ("MatMulNBits", "MultiHeadAttention", "GroupQueryAttention", "Attention", "SkipLayerNormalization", "LayerNormalization", "SimplifiedLayerNormalization", "RotaryEmbedding"):
            a = tuple(sorted((x.name, str(helper.get_attribute_value(x))) for x in n.attribute if x.name not in ("K", "N")))
            key = (n.op_type, a, len(n.input))
            if key not in seen:
                seen.add(key)
                print("   attr", n.op_type, dict(a), "ninputs", len(n.input), [i for i in n.input][:8])
    # Cast summary
    casts = collections.Counter(helper.get_attribute_value(n.attribute[0]) for n in g.node if n.op_type == "Cast")
    if casts:
        print("  Cast to:", {onnx.TensorProto.DataType.Name(k): v for k, v in casts.items()})
