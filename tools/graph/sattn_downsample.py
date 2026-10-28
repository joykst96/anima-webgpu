# self-attention K/V 다운샘플링(ToDo류)을 받을 수 있게 DiT 그래프를 바꾼다 (그래프만, 가중치 그대로).
#
# ※ 폐기된 실험 (2026-10-27, 미적용) — 키를 1/4로 줄이는 변형은 모두 그림이 무너지고, 1/4만 빼도 그림이 바뀐다.
#   구현 문제가 아니라 모델 특성(자기·이웃 토큰 의존)이다.
#   재검토하려면 make_fast_graphs.py에서 xattn_compact 뒤에 넣고, fp16_zones --fast(KEEP_NORM)가 여기 ReduceMean을
#   fp16 구역 밖으로 빼지 않는지 확인해야 한다(미확인).
#
# self-attention이 DiT 스텝의 절반(1024² 728ms/1425ms)을 차지한다. 키·값 토큰 수를 1/4로 줄이면 QKᵀ·PV 계산이 1/4로 준다.
# 쿼리는 그대로라 출력 해상도는 같다.
#   - 새 입력 sattn_index [G, T] int64: K/V 토큰 T개 각각을 만들 원래 토큰 G개 (토큰 순서 = (h, w) 행 우선, patch 2)
#     G=1이면 최근접(행 선택), G=4면 2×2 평균. 끄면 [1, S] = 0..S-1
#   - 각 self MultiHeadAttention의 k/v 입력 앞에 Gather(axis=1) → [1, G, T, D] → ReduceMean(axis 1) → [1, T, D]
# 끄면 Gather는 그대로 복사, 원소 1개의 평균은 자기 자신 → 종전과 같은 결과. 인덱스는 페이지가 스텝마다 넣으므로
# 적용 스텝·방식을 그래프 수정 없이 바꿀 수 있다.
# xattn_compact.py 뒤, fp16_zones.py 앞에 적용한다 (Gather·ReduceMean은 fp16 구역 안으로 들어간다).
#
#   python -I tools/graph/sattn_downsample.py <src.onnx> <dst.onnx>
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
g = m.graph

selfs = [n for n in g.node if n.op_type == "MultiHeadAttention" and "cross" not in n.name]
assert selfs and all(len([x for x in n.input if x]) == 3 for n in selfs), "self MHA 입력 형태가 예상과 다름"
opset = next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))
assert opset >= 18, f"ReduceMean axes 입력은 opset 18+ (현재 {opset})"
g.input.append(helper.make_tensor_value_info("sattn_index", TensorProto.INT64, ["sattn_g", "sattn_len"]))
g.initializer.append(numpy_helper.from_array(np.array([1], np.int64), "sattn_axis1"))

extra = []
for n in selfs:
    q, k, v = n.input[0], n.input[1], n.input[2]
    for x, tag in ((k, "k"), (v, "v")):
        extra.append(helper.make_node("Gather", [x, "sattn_index"], [x + "_sdg"], axis=1, name=f"{n.name}_sd_{tag}"))
        extra.append(helper.make_node("ReduceMean", [x + "_sdg", "sattn_axis1"], [x + "_sd"], keepdims=0,
                                      name=f"{n.name}_sd_{tag}_mean"))
    n.input[1], n.input[2] = k + "_sd", v + "_sd"

# 위상 정렬 (새 노드를 입력이 준비된 뒤로)
nodes = list(g.node) + extra
avail = {i.name for i in g.input} | {i.name for i in g.initializer} | {""}
ordered, pending = [], nodes
while pending:
    rest = []
    for n in pending:
        if all(i in avail for i in n.input):
            ordered.append(n)
            avail.update(n.output)
        else:
            rest.append(n)
    if len(rest) == len(pending):
        raise SystemExit(f"위상 정렬 실패: {rest[0].name}")
    pending = rest
del g.node[:]
g.node.extend(ordered)
del g.value_info[:]
onnx.save(m, dst)
print(f"self MHA {len(selfs)}개: K/V 토큰 묶음 평균(Gather + ReduceMean) 연결 → {dst}")
