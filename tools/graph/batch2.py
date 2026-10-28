# DiT 그래프의 batch 축(=1 고정)을 latent 입력의 batch로 바꾼다 — CFG의 긍정/부정을 batch 2로 한 번에 돌리는 실험 (그래프만, 가중치 그대로).
#   python -I tools/graph/batch2.py <src.onnx> <dst.onnx>
#
# ※ 미적용 실험 (2026-10-27) — 결과는 비트 동일하지만 빠르지 않다.
#   batch 1 입력 = 원래 그래프와 100% 같은 출력, batch 2 [긍정, 부정] = 각각 따로 돌린 출력과 100% 같음(1024², 베이스 _x 그래프).
#   시간: batch 1 두 번 2.86s vs batch 2 한 번 2.93s(+2.5%), 일시 VRAM은 두 배(정점 15.7GB/16GB). DiT가 GPU 계산 한도라 묶어도 이득 없음.
#   그래프를 JS에서 쓰려면 denoise가 latent·context·negpip_mask를 batch로 이어 붙여 한 번 run하고 출력을 반으로 나누면 된다.
#   패딩 압축(xattn_compact)과 함께 쓰려면 긍정·부정의 T가 같아야 한다(바이어스 행을 batch별로 받도록 바꿔야 함 — 미구현).
#
# 바꾸는 것:
#  - batch를 나르는 텐서(latent·context·negpip_mask에서 데이터로 이어진 것, Shape 경유 제외)에 걸린 Reshape:
#    모양의 첫 원소 1 → 0(입력 dim0 복사), allowzero=0. 상수 모양과 Concat으로 만드는 모양 둘 다.
#    (timesteps·RoPE처럼 batch와 무관한 경로는 1을 유지해 브로드캐스트)
#  - 패딩 마스크 채널(Tile repeats [1,1,1,1,1]) → [B,1,1,1,1]
#  - 입출력 dim0 → 'batch'
import sys, collections, onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
init ={t.name: t for t in g.initializer}

BATCH_IN = {"latent", "context", "negpip_mask"}
carry = set(BATCH_IN)
for n in g.node:  # 위상 순서
    if n.op_type == "Shape": continue
    if any(i in carry for i in n.input): carry.update(n.output)

def const_val(name):
    if name in init and init[name].data_location != 1: return numpy_helper.to_array(init[name])
    p = prod.get(name)
    if p is not None and p.op_type == "Constant": return numpy_helper.to_array(p.attribute[0].t)
    return None

new_inits, stats = [], collections.Counter()
zero1 = "b2_zero1"; new_inits.append(numpy_helper.from_array(np.array([0], dtype="int64"), zero1))
cache = {}
for n in g.node:
    if n.op_type != "Reshape" or n.input[0] not in carry: continue
    shp = n.input[1]
    v = const_val(shp)
    if v is not None:
        if v.ndim == 1 and v.size >= 2 and v[0] == 1:
            if shp not in cache:
                nv = v.copy(); nv[0] = 0
                nm = f"b2_{shp}"; new_inits.append(numpy_helper.from_array(nv, nm)); cache[shp] = nm
            n.input[1] = cache[shp]; stats["reshape_const"] += 1
        else: stats["reshape_const_skip" + str(v.tolist()[:2])] += 1
    else:
        c = prod.get(shp)
        if c is not None and c.op_type == "Concat":
            first = const_val(c.input[0])
            if first is not None and first.tolist() == [1]:
                if shp not in cache:
                    nc = helper.make_node("Concat", [zero1] + list(c.input[1:]), [f"b2_{shp}"], axis=0, name=f"b2_{c.name or shp}")
                    g.node.append(nc); cache[shp] = f"b2_{shp}"
                n.input[1] = cache[shp]; stats["reshape_concat"] += 1
            else: stats["reshape_dyn_skip"] += 1
        else: stats["reshape_dyn_other"] += 1
    for a in n.attribute:
        if a.name == "allowzero": a.i = 0
# 패딩 마스크 Tile → batch
bshape = helper.make_node("Shape", ["latent"], ["b2_B"], start=0, end=1, name="b2_shapeB")
rep = helper.make_node("Concat", ["b2_B", "b2_ones4"], ["b2_tile_rep"], axis=0, name="b2_tilerep")
new_inits.append(numpy_helper.from_array(np.array([1, 1, 1, 1], dtype="int64"), "b2_ones4"))
for n in g.node:
    if n.op_type == "Tile" and n.output[0] == "repeat":
        n.input[1] = "b2_tile_rep"; stats["tile"] += 1
g.node.insert(0, rep); g.node.insert(0, bshape)
g.initializer.extend(new_inits)
for vi in list(g.input) + list(g.output):
    if vi.name in BATCH_IN or vi in g.output:
        d = vi.type.tensor_type.shape.dim[0]; d.ClearField("dim_value"); d.dim_param = "batch"
# 남은 모양 정보는 batch 1 가정일 수 있어 지운다
del g.value_info[:]
# 위상 정렬 (새 Concat 노드를 소비자 앞으로)
order, avail = [], set([i.name for i in g.input] + list(init) + [t.name for t in new_inits])
nodes = list(g.node)
pending = nodes
while pending:
    rest = []
    for n in pending:
        if all((not i) or i in avail for i in n.input):
            order.append(n); avail.update(n.output)
        else: rest.append(n)
    if len(rest) == len(pending): raise SystemExit(f"정렬 실패 {len(rest)}")
    pending = rest
del g.node[:]; g.node.extend(order)
print(dict(stats), "carry", len(carry))
onnx.save(m, dst)
