# VAE 분할 그래프(tail 등) fp16 변형 — 그래프만 바꾸고 가중치는 그대로(fp32 initializer에 Cast를 걸면 ORT가 세션 생성 때 접는다).
#   python -I tools/graph/vae_fp16.py <src.onnx> <dst.onnx> <conv|all|nonorm> <데이터 입력 이름,...>
#   예) ... qwen_image_vae_decoder_dyn32_2dc_tail.onnx tail_all.onnx all add_257
#
# ※ 미적용 실험 (2026-10-27) — "VAE는 fp32 필수(fp16 오버플로·검은 이미지)" 주장은 재현되지 않았다:
#   tail 전부 fp16(all)도 1024² 실제 latent에서 NaN/Inf 없음, fp32 대비 PSNR 53dB(최대 24/255).
#   그러나 ORT 1.26 WebGPU의 fp16 Conv2dMM·원소별 커널이 fp32보다 느려(Conv 380 → 795ms, Mul 3배) tail 0.68 → 1.10s(conv만 2.33s).
#   ORT 1.30에서는 fp16 커널이 정상이라 all이 0.67 → 0.58s(−13%)지만, 1.30은 DiT가 6% 느려 전체로는 손해.
#   → ORT를 올릴 때 다시 볼 후보. 그때 head(어텐션 포함)도 같은 변환이 필요하다.
#   mode conv  : Conv만 fp16 (입력·가중치 Cast, 출력 fp32로 되돌림)
#   mode all   : 데이터 경로 전부 fp16
#   mode nonorm: 전부 fp16, 단 RMS 정규화(ReduceL2·Clip·Expand·Div)는 fp32
# 데이터 경로 = 입력 텐서에서 데이터로 이어진 노드(Shape 경유 제외). 그래프 입출력 dtype은 그대로.
import sys, collections, onnx
from onnx import helper, TensorProto
src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
data_inputs = sys.argv[4].split(",")
m = onnx.load(src, load_external_data=False)
g = m.graph
init = {t.name: t for t in g.initializer}
F32, F16 = TensorProto.FLOAT, TensorProto.FLOAT16
dtype = {i.name: i.type.tensor_type.elem_type for i in g.input}
dtype.update({t.name: t.data_type for t in g.initializer})
data = set(data_inputs)
for n in g.node:
    if n.op_type == "Shape": continue
    if any(i in data for i in n.input): data.update(n.output)
# 노드별 정밀도 결정
NORM = {"ReduceL2", "Clip", "Expand", "Div"}
def want16(n):
    if not any(i in data for i in n.input): return False
    if n.op_type in ("Shape", "Range"): return False
    if mode == "conv": return n.op_type == "Conv"
    if mode == "nonorm" and n.op_type in NORM: return False
    return True
nodes = list(g.node)
mode16 = [want16(n) for n in nodes]
# float 텐서 추적: 각 텐서의 현재 dtype (fp16로 바뀐 것 기록)
is16 = {}  # tensor -> True면 fp16로 생산됨
for i in g.input: is16[i.name] = dtype[i.name] == F16
new_nodes, ncast = [], collections.Counter()
cast_cache = {}
def cast(name, to):
    key = (name, to)
    if key not in cast_cache:
        out = f"{name}__to{'16' if to == F16 else '32'}"
        new_nodes.append(helper.make_node("Cast", [name], [out], to=to, name=out))
        cast_cache[key] = out; ncast[to] += 1; is16[out] = to == F16
    return cast_cache[key]
def is_float_tensor(name):
    if name in init: return init[name].data_type in (F32, F16)
    return name in is16
for idx, n in enumerate(nodes):
    want = mode16[idx]
    if n.op_type == "Cast":
        to = next(a.i for a in n.attribute if a.name == "to")
        if want and to == F32:
            for a in n.attribute:
                if a.name == "to": a.i = F16
            to = F16
        new_nodes.append(n)
        for o in n.output:
            if to in (F32, F16): is16[o] = to == F16
        continue
    for k, i in enumerate(n.input):
        if not i or not is_float_tensor(i): continue
        cur16 = is16.get(i, False) if i not in init else init[i].data_type == F16
        # 정수 인덱스/shape 입력은 위에서 걸러짐(is_float_tensor). Clip min/max·Pow 지수 등 상수도 같은 dtype이어야 한다.
        if want and not cur16: n.input[k] = cast(i, F16)
        elif not want and cur16: n.input[k] = cast(i, F32)
    new_nodes.append(n)
    for o in n.output:
        if n.op_type in ("Shape", "Range") or (n.op_type in ("Gather", "GatherND", "Concat", "Reshape", "Squeeze", "Unsqueeze", "Expand", "Max", "Mul", "Add", "Div") and not any(is_float_tensor(i) for i in n.input)):
            continue  # 정수 경로
        is16[o] = want
# 출력은 원래 dtype으로
for o in g.output:
    if is16.get(o.name) and o.type.tensor_type.elem_type == F32:
        tmp = o.name + "__16"
        for n in new_nodes:
            for k, x in enumerate(n.output):
                if x == o.name: n.output[k] = tmp
            for k, x in enumerate(n.input):
                if x == o.name: n.input[k] = tmp
        new_nodes.append(helper.make_node("Cast", [tmp], [o.name], to=F32, name=o.name + "__out32"))
# 위상 정렬
avail = set([i.name for i in g.input] + list(init))
order, pending = [], new_nodes
while pending:
    rest = []
    for n in pending:
        if all((not i) or i in avail for i in n.input): order.append(n); avail.update(n.output)
        else: rest.append(n)
    if len(rest) == len(pending): raise SystemExit(f"정렬 실패 {[n.name or n.op_type for n in rest[:5]]}")
    pending = rest
del g.node[:]; g.node.extend(order); del g.value_info[:]
print(mode, "fp16 노드", sum(mode16), "/", len(mode16), "Cast 추가", dict(ncast))
onnx.save(m, dst)
