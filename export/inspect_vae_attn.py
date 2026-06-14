"""
VAE attention(SDPA) 주변 구조 정밀 진단 — VAE 전용 fuse 설계용.

DiT와 달리 VAE는 attention 1블록뿐. ONNX로 익스포트된 scaled_dot_product_attention의
정확한 노드 체인(Q/K/V proj 경계, scale 위치, reshape/transpose)을 까서
com.microsoft.MultiHeadAttention으로 어떻게 치환할지 경계를 확정한다.

사용 (한 줄):
  python_embeded\python.exe export\inspect_vae_attn.py --vae out\vae\..._2d.onnx
"""
import argparse

import onnx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", required=True)
    p.add_argument("--ctx", type=int, default=6, help="attention 앞뒤로 보여줄 노드 수")
    args = p.parse_args()

    g = onnx.load(args.vae, load_external_data=False).graph
    producer = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    order = {id(n): i for i, n in enumerate(g.node)}
    init_names = {i.name for i in g.initializer}
    init_shapes = {i.name: tuple(i.dims) for i in g.initializer}

    sm = next(n for n in g.node if n.op_type == "Softmax")
    # QK MatMul = softmax 입력 생산자, PV MatMul = softmax 출력 소비자
    qk = producer.get(sm.input[0])
    pv = next((c for c in consumers.get(sm.output[0], []) if c.op_type == "MatMul"), None)

    def desc(n):
        if n is None:
            return "(없음)"
        ins = []
        for i in n.input:
            tag = ""
            if i in init_names:
                tag = f"[init{list(init_shapes[i])}]"
            ins.append(i + tag)
        return f"#{order[id(n)]} {n.op_type}({', '.join(ins)}) -> {list(n.output)}"

    print("=== attention 핵심 노드 ===")
    print("QK :", desc(qk))
    print("SM :", desc(sm))
    print("PV :", desc(pv))

    # QK 입력 후방 추적 (Q측, K측 각각 — scale Mul / Div / reshape / transpose 체인)
    def trace_back(start, depth=10, label=""):
        print(f"\n--- {label} 후방 체인 (from {start}) ---")
        t = start
        for _ in range(depth):
            n = producer.get(t)
            if n is None:
                print(f"  └ {t} = (그래프 입력 또는 initializer)")
                break
            print(f"  {desc(n)}")
            # 활성 입력(initializer 아닌) 중 첫 번째로 계속
            act = [i for i in n.input if i not in init_names]
            if not act:
                break
            t = act[0]

    if qk:
        trace_back(qk.input[0], label="QK.in[0] (Q측)")
        trace_back(qk.input[1], label="QK.in[1] (K측)")
    if pv:
        # PV의 입력 중 softmax 아닌 쪽 = V
        v_in = [i for i in pv.input if i != sm.output[0]]
        if v_in:
            trace_back(v_in[0], label="PV V측")

    # PV 출력 전방 (o_proj / reshape 복원)
    print("\n--- PV 출력 전방 (복원/o_proj) ---")
    t = pv.output[0] if pv else None
    for _ in range(8):
        cs = consumers.get(t, [])
        if not cs:
            print(f"  └ {t} = (그래프 출력 또는 말단)")
            break
        n = cs[0]
        print(f"  {desc(n)}")
        t = n.output[0]

    # scale 단서: QK 주변 Mul/Div 상수
    print("\n--- scale 단서 (attention 부근 Mul/Div 상수) ---")
    for n in g.node:
        if n.op_type in ("Mul", "Div") and abs(order[id(n)] - order[id(sm)]) < 20:
            consts = [i for i in n.input if i in init_names]
            for c in consts:
                arr = onnx.numpy_helper.to_array(
                    next(it for it in g.initializer if it.name == c))
                if arr.size <= 4:
                    print(f"  #{order[id(n)]} {n.op_type} const {c} = {arr.flatten()}")


if __name__ == "__main__":
    main()
