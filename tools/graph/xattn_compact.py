# cross-attention 패딩 압축을 받을 수 있게 DiT 그래프를 바꾼다 (그래프만, 가중치 그대로).
#
# context는 512토큰으로 0 패딩된다. 패딩 토큰은 k = k_norm(k_proj(0)) = 0, v = v_proj(0) = 0 (bias 없음)이라
# 점수가 항상 0 → 소프트맥스 분모에 m·exp(0−max)만 더하고 분자엔 0. 따라서 패딩 m개를
# "패딩 토큰 1개 + 점수 bias ln(m)"으로 바꿔도 수학적으로 같다.
#
# K/V 투영은 종전처럼 512행 전체로 계산하고 어텐션 직전에 행만 고른다(투영 입력·출력은 종전과 동일).
# 압축을 켜면 어텐션 합산 순서가 바뀌어 float 반올림 수준(~1e-6)으로 달라지고, 이를 다음 층의 DP4A 활성 int8 반올림이
# 증폭해 DiT 출력이 ~2~3% 달라진다(실측. 활성 fp32 그래프에선 0.002% — 수식은 정확). context를 줄이는 방식도 결과가 같았다.
#   - 새 입력 xattn_index [T] int64: 남길 K/V 행 (실제 토큰 0..n-1 + 패딩 대표 1개). 끄면 0..511
#   - 새 입력 xattn_bias_row [1,1,1,T]: 실제 토큰 0, 패딩 대표 ln(m). 끄면 전부 0
#   - 각 cross MultiHeadAttention의 k/v 입력 앞에 Gather(axis=1, xattn_index), attention_bias(입력 5) = bias 행을 [1,1,S,T]로 Expand
# 끄면(전체 인덱스 + bias 0) Gather는 그대로 복사, +0은 정확 → 종전과 같은 결과(골든 해시 일치로 검증).
# fp16 구역 변환(fp16_zones.py)은 이 뒤에 적용한다.
#
#   python -I tools/graph/xattn_compact.py <src.onnx> <dst.onnx>
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
g = m.graph

cross = [n for n in g.node if n.op_type == "MultiHeadAttention" and "cross" in n.name]
assert cross and all(len([x for x in n.input if x]) == 3 for n in cross), "cross MHA 입력 형태가 예상과 다름"
g.input.append(helper.make_tensor_value_info("xattn_index", TensorProto.INT64, ["xattn_len"]))
g.input.append(helper.make_tensor_value_info("xattn_bias_row", TensorProto.FLOAT, [1, 1, 1, "xattn_len"]))


def c(name, arr):
    g.initializer.append(numpy_helper.from_array(np.array(arr, np.int64), name))
    return name


q0 = cross[0].input[0]
extra = [
    helper.make_node("Shape", [q0], ["xattn_qshape"], name="xattn_qshape"),
    helper.make_node("Gather", ["xattn_qshape", c("xattn_i1", 1)], ["xattn_S"], axis=0, name="xattn_S"),
    helper.make_node("Unsqueeze", ["xattn_S", c("xattn_ax0", [0])], ["xattn_S1"], name="xattn_S1"),
    helper.make_node("Concat", [c("xattn_one", [1]), c("xattn_one2", [1]), "xattn_S1", c("xattn_one3", [1])],
                     ["xattn_shape"], axis=0, name="xattn_shape"),
    helper.make_node("Expand", ["xattn_bias_row", "xattn_shape"], ["xattn_bias"], name="xattn_bias"),
]
for n in cross:
    q, k, v = n.input[0], n.input[1], n.input[2]
    extra.append(helper.make_node("Gather", [k, "xattn_index"], [k + "_xa"], axis=1, name=n.name + "_xa_k"))
    extra.append(helper.make_node("Gather", [v, "xattn_index"], [v + "_xa"], axis=1, name=n.name + "_xa_v"))
    del n.input[:]
    n.input.extend([q, k + "_xa", v + "_xa", "", "", "xattn_bias"])

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
print(f"cross MHA {len(cross)}개: K/V 행 선택(Gather) + attention_bias 연결 → {dst}")
