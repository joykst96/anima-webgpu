# DiT 그래프의 "블록 내부 연산"만 fp16으로 바꾸는 그래프 후처리 (가중치 파일은 그대로, 재export 불필요).
#
# ComfyUI의 공식 fp16 경로(comfy/ldm/cosmos/predict2.py, 커밋 6a26328)와 같은 구조:
#   - fp32 유지: 잔차 스트림(블록 출력 합 Add 84개), 그 합에 들어가는 gate Mul, LayerNorm, 시간(adaLN) 경로,
#                블록 앞(x_embedder·RoPE 표) / 뒤(final_layer)
#   - fp16: 각 LayerNorm 출력에서 gate Mul 직전까지 = 변조(modulate) · q/k/v/o 투영 · q/k RMSNorm · RoPE ·
#           어텐션(MHA→FlashAttention fp16) · MLP(GELU 포함)
#   - MatMulNBits는 그대로(DP4A, accuracy_level 4) — 입력만 fp16, scales는 Cast(로드 시 상수 접힘)
# 경계에는 Cast만 넣는다(fp32→fp16 입력, fp16→fp32 출력).
#
#   python -I tools/graph/fp16_zones.py <src.onnx> <dst.onnx> --fast
#   --fast = --keep-matmul --keep-mod --keep-norm (페이지 '빠르게' 모드, 실측: 현재 대비 오차 2.9/1.8/1.6%, DiT −11%)
#   개별 옵션은 실험용 — 전부 fp16(옵션 없음)은 DP4A에 fp16 입력이 들어가 오차 18%.
import sys, collections
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
FAST = "--fast" in sys.argv             # 페이지 '빠르게' 모드 = --keep-matmul --keep-mod --keep-norm
KEEP_NORM = FAST or "--keep-norm" in sys.argv
KEEP_MM = FAST or "--keep-matmul" in sys.argv   # MatMulNBits는 fp32 입력으로 (DP4A 활성 양자화를 fp32에서)
KEEP_MLP = "--keep-mlp" in sys.argv     # MLP(up→GELU→down)는 fp32로
KEEP_MOD = FAST or "--keep-mod" in sys.argv     # LN 직후 변조(×(1+scale)+shift)는 fp32 (ComfyUI: _fn을 residual dtype으로 계산 후 cast)
m = onnx.load(src, load_external_data=False)
g = m.graph
nodes = list(g.node)
prod = {o: n for n in nodes for o in n.output}
cons = collections.defaultdict(list)
for n in nodes:
    for i in n.input:
        if i:
            cons[i].append(n)
init = {i.name: i for i in g.initializer}
F32, F16 = TensorProto.FLOAT, TensorProto.FLOAT16

# ── dtype 추론 (이 그래프에 쓰인 op 범위) ──
dtype = {i.name: i.type.tensor_type.elem_type for i in g.input}
dtype.update({i.name: i.data_type for i in g.initializer})
SAME_AS_IN0 = {"Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Reciprocal", "Erf", "Sigmoid", "Neg", "Cos", "Sin",
               "ReduceMean", "Reshape", "Transpose", "Slice", "Concat", "Split", "Squeeze", "Unsqueeze", "Expand",
               "Tile", "Pad", "Gather", "LayerNormalization", "MatMulNBits", "MultiHeadAttention", "Range", "Mod", "Max",
               "MatMul",  # MatMul = LoRA 슬롯 브랜치(add_lora_slots.py)
               "Gelu", "SimplifiedLayerNormalization"}  # fuse_elementwise.py 산출
for n in nodes:
    if n.op_type == "Cast":
        t = next(a.i for a in n.attribute if a.name == "to")
    elif n.op_type == "Shape":
        t = TensorProto.INT64
    elif n.op_type == "Constant":
        t = next(a.t.data_type for a in n.attribute if a.name == "value")
    elif n.op_type in SAME_AS_IN0:
        t = dtype.get(n.input[0])
    else:
        raise SystemExit(f"dtype 규칙 없음: {n.op_type}")
    for o in n.output:
        dtype[o] = t

# ── 구역 식별 ──
lns = [n for n in nodes if n.op_type == "LayerNormalization"]
residual = {n.input[0] for n in lns}                      # 잔차 스트림 텐서 (LN 입력)
res_adds = [n for n in nodes if n.op_type == "Add" and n.output[0] in residual]
gates = set()
for a in res_adds:
    for i in a.input:
        if i not in residual and i in prod and prod[i].op_type == "Mul":
            gates.add(id(prod[i]))
block_lns = [n for n in lns if any(c.op_type != "LayerNormalization" for c in cons[n.output[0]])][:-1] \
    if False else lns[:-1]                                # 마지막 LN = final_layer → fp32 유지
stop = gates | {id(a) for a in res_adds} | {id(n) for n in lns}

zone = set()
frontier = [c for ln in block_lns for c in cons[ln.output[0]]]
while frontier:
    n = frontier.pop()
    if id(n) in zone or id(n) in stop:
        continue
    zone.add(id(n))
    for o in n.output:
        frontier.extend(cons[o])
if KEEP_MOD:
    # LN 출력 → Mul(scale) → Add(shift) 두 단계 (각 소비자가 Mul/Add인 동안)
    for ln in block_lns:
        frontier = list(cons[ln.output[0]])
        for _ in range(2):
            nxt = []
            for n in frontier:
                if n.op_type in ("Mul", "Add"):
                    zone.discard(id(n))
                    for o in n.output:
                        nxt.extend(cons[o])
            frontier = nxt
if KEEP_MM:
    for n in nodes:
        if n.op_type == "MatMulNBits":
            zone.discard(id(n))
if KEEP_MLP:
    # MLP up(N=8192) → GELU(Div/Erf/Add/Mul) → down(K=8192): up 출력에서 down까지 전방 도달 노드
    def attr(n, k):
        return next((a.i for a in n.attribute if a.name == k), None)
    for n in nodes:
        if n.op_type == "MatMulNBits" and attr(n, "N") == 8192:
            stack, seen = [n], set()
            while stack:
                x = stack.pop()
                if id(x) in seen:
                    continue
                seen.add(id(x))
                zone.discard(id(x))
                if x.op_type == "MatMulNBits" and attr(x, "K") == 8192:
                    continue   # down 투영까지 포함하고 멈춤
                for o in x.output:
                    stack.extend(cons[o])
if KEEP_NORM:
    for n in nodes:
        if id(n) in zone and n.op_type in ("Pow", "ReduceMean", "Sqrt", "Reciprocal", "SimplifiedLayerNormalization"):
            zone.discard(id(n))
    # 체인 사이의 eps Add도 fp32 (입력이 ReduceMean 출력)
    for n in nodes:
        if id(n) in zone and n.op_type == "Add" and any(prod.get(i) is not None and prod[i].op_type == "ReduceMean" for i in n.input):
            zone.discard(id(n))

# ── 변환 ──
new_nodes, new_inits = [], []
cast16 = {}   # fp32 텐서 → fp16 버전 이름

def to16(name):
    if name in cast16:
        return cast16[name]
    if name in init and init[name].data_location != TensorProto.EXTERNAL and init[name].data_type == F32:
        arr = numpy_helper.to_array(init[name]).astype(np.float16)
        nm = name + "__f16"
        new_inits.append(numpy_helper.from_array(arr, nm))
    else:
        nm = name + "__f16"
        new_nodes.append(helper.make_node("Cast", [name], [nm], to=F16, name=nm + "_cast"))
    cast16[name] = nm
    return nm

out_cast = {}  # 구역 내부 fp16 출력 → 구역 밖 소비자용 fp32
for n in nodes:
    if id(n) in zone:
        for k, i in enumerate(n.input):
            if not i:
                continue
            p = prod.get(i)
            if dtype.get(i) == F32 and (p is None or id(p) not in zone):
                n.input[k] = to16(i)
        if n.op_type == "Cast":
            for a in n.attribute:
                if a.name == "to" and a.i == F32:
                    a.i = F16
        for o in n.output:
            if dtype.get(o) == F32:
                dtype[o] = F16
        new_nodes.append(n)
    else:
        for k, i in enumerate(n.input):
            p = prod.get(i)
            if p is not None and id(p) in zone and dtype.get(i) == F16:
                if i not in out_cast:
                    nm = i + "__f32"
                    new_nodes.append(helper.make_node("Cast", [i], [nm], to=F32, name=nm + "_cast"))
                    out_cast[i] = nm
                n.input[k] = out_cast[i]
        new_nodes.append(n)

# 위상 순서 보정: Cast를 소비자 앞에 두기 위해 단순 위상 정렬
avail = {i.name for i in g.input} | {i.name for i in g.initializer} | {i.name for i in new_inits} | {""}
ordered, pending = [], list(new_nodes)
while pending:
    rest = []
    for n in pending:
        if all(i in avail for i in n.input):
            ordered.append(n)
            avail.update(n.output)
        else:
            rest.append(n)
    if len(rest) == len(pending):
        raise SystemExit(f"위상 정렬 실패: {rest[0].name} 입력 {[i for i in rest[0].input if i not in avail]}")
    pending = rest

del g.node[:]
g.node.extend(ordered)
g.initializer.extend(new_inits)
del g.value_info[:]   # 낡은 타입 정보 제거 (ORT가 다시 추론)
onnx.save(m, dst)
cnt = collections.Counter(n.op_type for n in nodes if id(n) in zone)
print(f"fp16 구역 노드 {len(zone)}개 / 전체 {len(nodes)} (잔차 Add {len(res_adds)}, gate {len(gates)}, LN {len(lns)})")
print("  구역 op:", dict(cnt.most_common(12)))
print(f"  Cast 추가: 입력 {sum(1 for v in cast16 if v not in init)}·초기화값 fp16 {len(new_inits)}·출력 {len(out_cast)} → {dst}")
