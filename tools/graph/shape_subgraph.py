# 그래프에서 "shape 계산 부분그래프"를 찾는다: 입력이 전부 (initializer | Constant | Shape 출력 | 이 집합의 출력)인 노드.
# ORT는 이런 노드를 CPU에 배정한다(데이터 의존 없음 → GPU→CPU 동기화 없음).
# 반대로 float 텐서 데이터를 먹는 비-Shape 노드가 CPU로 가면 동기화가 생긴다 → 그 후보도 나열.
import sys, collections, onnx
p = sys.argv[1]
m = onnx.load(p, load_external_data=False)
g = m.graph
init = {i.name for i in g.initializer}
const = set()
for n in g.node:
    if n.op_type == "Constant":
        const.update(n.output)
shape_vals = set()
shape_nodes = []
changed = True
order = list(g.node)
done = set()
while changed:
    changed = False
    for idx, n in enumerate(order):
        if idx in done or n.op_type == "Constant":
            continue
        ins = [i for i in n.input if i]
        if n.op_type == "Shape" or (ins and all(i in init or i in const or i in shape_vals for i in ins)):
            done.add(idx)
            shape_vals.update(n.output)
            shape_nodes.append(n)
            changed = True
print("shape-subgraph nodes", len(shape_nodes))
print(" ", collections.Counter(n.op_type for n in shape_nodes).most_common())
# consumers: shape values feeding into non-shape nodes (via which input slot)
uses = collections.Counter()
for n in g.node:
    if n in shape_nodes:
        continue
    for k, i in enumerate(n.input):
        if i in shape_vals and i not in init:
            uses[(n.op_type, k)] += 1
print("shape values consumed by GPU nodes (op, input slot):", uses.most_common())
# timestep path: nodes depending only on 'timesteps' input (+init/shape) — tiny tensors, likely also CPU-candidate
small = set(["timesteps"])
tnodes = []
for n in g.node:
    ins = [i for i in n.input if i]
    if ins and any(i in small for i in ins) and all(i in small or i in init or i in const or i in shape_vals for i in ins):
        small.update(n.output); tnodes.append(n.op_type)
print("timestep-only nodes", len(tnodes), collections.Counter(tnodes).most_common())
