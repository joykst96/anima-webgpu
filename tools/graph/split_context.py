# DiT에서 프롬프트(context·negpip_mask·xattn_*)에만 의존하는 노드를 별도 그래프로 떼어낸다.
#   python -I tools/graph/split_context.py <src.onnx> <pre.onnx> <main.onnx>
#   pre : 프롬프트 입력 → 경계 텐서들 (job마다 한 번)
#   main: latent·timesteps + 경계 텐서 (+ 스텝 쪽과 섞여 쓰이는 프롬프트 입력, 예: xattn_bias_row) → v_pred (스텝마다)
# 경계 텐서 = 프롬프트 전용 노드의 출력 중 나머지 노드가 쓰는 것(cross-attention K/V 투영·k-norm 결과 등).
# 두 그래프 모두 원래의 외부 데이터(샤드) 참조를 그대로 쓴다 — 페이지의 DiT 세션 옵션(externalData)을 그대로 넘기면 된다.
import sys, collections, onnx
from onnx import helper, TensorProto

src, pre_p, main_p = sys.argv[1:4]
m = onnx.load(src, load_external_data=False)
g = m.graph
CTX = {"context", "negpip_mask", "xattn_index", "xattn_bias_row"}
STEP = {"latent", "timesteps"}
init = {t.name: t for t in g.initializer}
inputs = {i.name: i for i in g.input}

dep = {}  # 텐서 → {"ctx", "step"} 부분집합
for k in CTX:
    dep[k] = {"ctx"}
for k in STEP:
    dep[k] = {"step"}
for n in g.node:
    d = set()
    for i in n.input:
        d |= dep.get(i, set())
    for o in n.output:
        dep[o] = d
ctx_only = [n for n in g.node if any(dep.get(i) == {"ctx"} for i in n.input) and all(not (dep.get(i, set()) & {"step"}) for i in n.input)]
ctx_out = {o for n in ctx_only for o in n.output}
rest = [n for n in g.node if n.name not in {c.name for c in ctx_only}]
boundary = sorted({i for n in rest for i in n.input if i in ctx_out} | {o.name for o in g.output if o.name in ctx_out})

# dtype 전파 (경계 텐서를 그래프 입력으로 선언할 때 필요)
dt = {k: inputs[k].type.tensor_type.elem_type for k in inputs}
dt.update({k: t.data_type for k, t in init.items()})
for n in g.node:
    if n.op_type == "Cast":
        t = next(a.i for a in n.attribute if a.name == "to")
    elif n.op_type == "Shape":
        t = TensorProto.INT64
    elif n.op_type in ("MatMulNBits", "Gather", "GatherElements", "Expand", "Reshape", "Transpose", "Unsqueeze", "Squeeze", "Slice", "Split", "Concat", "Tile"):
        t = dt.get(n.input[0])
    else:
        t = next((dt[i] for i in n.input if i in dt and dt[i] in (TensorProto.FLOAT, TensorProto.FLOAT16)), dt.get(n.input[0]) if n.input else None)
    for o in n.output:
        dt[o] = t


def used_inits(nodes):
    used = {i for n in nodes for i in n.input}
    return [t for t in g.initializer if t.name in used]


def make(nodes, ins, outs, name):
    gg = helper.make_graph(nodes, name, ins, outs, used_inits(nodes))
    mm = helper.make_model(gg, opset_imports=m.opset_import, ir_version=m.ir_version)
    return mm


order = [i.name for i in g.input]
bvi = [helper.make_tensor_value_info(b, dt[b], None) for b in boundary]
pre_in = [inputs[k] for k in order if k in CTX]
main_used = {i for n in rest for i in n.input}
main_in = [inputs[k] for k in order if k in STEP or (k in CTX and k in main_used)] + bvi  # 프롬프트 입력 중 main이 직접 쓰는 것(xattn_bias_row 등)은 남긴다
onnx.save(make(ctx_only, pre_in, bvi, "dit_pre"), pre_p)
onnx.save(make(rest, main_in, list(g.output), "dit_main"), main_p)
print("ctx-only nodes", len(ctx_only), collections.Counter(n.op_type for n in ctx_only).most_common(6))
print("boundary", len(boundary), collections.Counter(TensorProto.DataType.Name(dt[b]) for b in boundary))
print("main nodes", len(rest))
