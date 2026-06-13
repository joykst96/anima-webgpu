"""
fuse 후 그래프에서 cross v_proj의 입력 상태 진단 (negpip 메타 전환 전제 확인).

add_negpip 삽입부는 v_proj.input[0](활성 텐서)에 mask Mul을 끼운다.
fuse가 v_proj의 input을 안 건드렸는지(여전히 원본 context 경로인지), k/v가 같은
활성을 공유하는지 확인. 메타가 준 v_proj 노드명만으로 삽입이 안전한지 판정.

사용 (한 줄):
  python_embeded\python.exe export\diag_vproj.py --src out\dit\anima_dit_dyn32_q8a4f.onnx
"""
import argparse
import json
from collections import Counter

import onnx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--meta", default=None, help="기본: <src>.fuse_meta.json")
    args = p.parse_args()
    meta_path = args.meta or (args.src + ".fuse_meta.json")

    g = onnx.load(args.src, load_external_data=False).graph
    prod = {o: n for n in g.node for o in n.output}
    byname = {n.name: n for n in g.node}
    meta = json.load(open(meta_path))
    cross = [a for a in meta["attentions"] if a["kind"] == "cross"]
    selfa = [a for a in meta["attentions"] if a["kind"] == "self"]

    print(f"[meta] self {len(selfa)} / cross {len(cross)}")

    # 1) cross v_proj input[0] 생산자 op_type 분포
    c = Counter()
    samples = []
    for a in cross:
        v = byname[a["v_proj"]]
        src = v.input[0]
        op = prod[src].op_type if src in prod else "GRAPH_INPUT"
        c.update([op])
        if len(samples) < 3:
            samples.append((a["v_proj"], src, op))
    print(f"[v_proj] input[0] 생산자 op_type 분포: {dict(c)}")
    for s in samples:
        print(f"  ex: v_proj={s[0]}  input[0]={s[1]}  producer={s[2]}")

    # 2) k_proj와 v_proj가 같은 활성 입력 공유하는가 (둘다 context 소비)
    same = all(byname[a["k_proj"]].input[0] == byname[a["v_proj"]].input[0]
               for a in cross)
    print(f"[k==v input] cross k_proj.input[0] == v_proj.input[0] (모두): {same}")

    # 3) v_proj input[0]별 종류 수 (negpip은 활성텐서별로 Mul 1개)
    v_inputs = sorted({byname[a["v_proj"]].input[0] for a in cross})
    print(f"[v_proj] 고유 input[0] 텐서 종류: {len(v_inputs)}개")
    print(f"  (블록마다 다른 context-cast 텐서면 28, 전역 공유면 1 — Mul 개수 결정)")

    # 4) v_proj input[0]의 dtype 추적용: 생산자가 Cast인지
    cast_cnt = sum(1 for a in cross
                   if (byname[a["v_proj"]].input[0] in prod
                       and prod[byname[a["v_proj"]].input[0]].op_type == "Cast"))
    print(f"[dtype] v_proj input[0] 생산자가 Cast인 경우: {cast_cnt}/{len(cross)}")
    print("  (fp32-act 그래프면 context fp16→Cast fp32 패턴 — act_dtype 추적이 Cast 찾음)")


if __name__ == "__main__":
    main()
