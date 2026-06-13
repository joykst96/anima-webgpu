"""
attention 코어 구조 진단 (읽기 전용 — 그래프 수정 없음).

목적: fuse_attention.py 설계에 필요한 7개 질문에 답한다.
  1. q_proj 출력 ~ QK matmul 입력 사이 노드열 (q_norm 전개 + RoPE + reshape/transpose)
  2. RoPE가 head 분할 전(B,S,hidden)인지 후(B,H,S,D)인지
  3. scale Mul 위치 (있으면 MHA scale 속성으로)
  4. QK matmul 입력 레이아웃 (rank/shape)
  5. Softmax 주변 Cast (fp32-act)
  6. PV 출력 ~ output_proj 입력 사이 reshape/transpose
  7. attention bias Add 유무

add_negpip.py의 build_maps + Softmax 앵커 + is_weight_matmul 재활용.
가중치는 안 올림 (load_external_data=False) — 그래프 위상만 본다.

사용 (한 줄):
  python export/inspect_attention.py --src out/dit/anima_dit_dyn32_q8a4.onnx
"""
import argparse

import onnx
from onnx import TensorProto

WEIGHT_MATMULS = {"MatMulNBits", "MatMul", "Gemm"}


def const_value(name, producer, init_map):
    """name이 가리키는 상수(initializer 또는 Constant 노드)의 스칼라/소형 배열 값.
    scale 판정용 — 1/sqrt(128)≈0.08839 인지 확인. 못 찾으면 None."""
    import numpy as np
    from onnx import numpy_helper
    # initializer 직접
    if name in init_map:
        try:
            arr = numpy_helper.to_array(init_map[name])
            return arr
        except Exception:
            return None
    # Constant 노드
    p = producer.get(name)
    if p is not None and p.op_type == "Constant":
        for a in p.attribute:
            if a.name == "value":
                try:
                    return numpy_helper.to_array(a.t)
                except Exception:
                    return None
    return None


def fmt_const(arr):
    """상수 배열을 짧게 요약 + scale 후보 표시."""
    import numpy as np
    if arr is None:
        return "(상수 못 읽음 — external/동적)"
    flat = np.asarray(arr).ravel()
    if flat.size == 0:
        return "(빈 배열)"
    if flat.size == 1:
        v = float(flat[0])
        tag = ""
        # 1/sqrt(128) = 0.088388, 1/sqrt(64)=0.125, sqrt 역수류
        for hd, val in ((128, 0.0883883), (64, 0.125), (96, 0.1020621)):
            if abs(v - val) < 1e-4:
                tag = f"  <<< 1/sqrt({hd}) = attention scale!"
        return f"scalar={v:.6f}{tag}"
    return (f"shape={flat.shape if hasattr(flat,'shape') else len(flat)} "
            f"size={flat.size} 첫값={float(flat[0]):.6f} "
            f"(min={float(flat.min()):.4f} max={float(flat.max()):.4f})")


def n_consumers(tensor, consumers):
    return len(consumers.get(tensor, []))


def build_maps(graph):
    producer, consumers = {}, {}
    for n in graph.node:
        for o in n.output:
            producer[o] = n
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    return producer, consumers


def is_weight_matmul(node, init_names):
    if node.op_type == "MatMulNBits":
        return True
    if node.op_type in ("MatMul", "Gemm") and len(node.input) >= 2:
        return node.input[1] in init_names
    return False


def get_kn(node, init_shapes):
    if node.op_type == "MatMulNBits":
        a = {x.name: x for x in node.attribute}
        return a["K"].i, a["N"].i
    shp = init_shapes.get(node.input[1])
    if shp and len(shp) == 2:
        return shp[0], shp[1]
    return None, None


def vi_shape(graph_value_infos, name):
    """value_info/입출력에서 텐서 shape를 dim 문자열로."""
    vi = graph_value_infos.get(name)
    if vi is None:
        return None
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
    dt = TensorProto.DataType.Name(vi.type.tensor_type.elem_type)
    return f"{dims} {dt}"


def fmt_node(n, init_names, value_infos, init_shapes):
    """노드 1줄 요약: op_type(name) 입력→출력, 가중치matmul이면 (K,N)."""
    extra = ""
    if is_weight_matmul(n, init_names):
        K, N = get_kn(n, init_shapes)
        extra = f"  [proj K={K} N={N}]"
    elif n.op_type in ("MatMul", "Gemm"):
        extra = "  [act×act matmul]"
    elif n.op_type == "Cast":
        to = next((a.i for a in n.attribute if a.name == "to"), None)
        extra = f"  [to={TensorProto.DataType.Name(to) if to else '?'}]"
    elif n.op_type in ("Transpose",):
        perm = next((list(a.ints) for a in n.attribute if a.name == "perm"), None)
        extra = f"  [perm={perm}]"
    elif n.op_type in ("Reshape",):
        extra = "  [reshape]"
    # 출력 shape (value_info 있으면)
    oshape = vi_shape(value_infos, n.output[0]) if n.output else None
    sh = f"  out={oshape}" if oshape else ""
    nm = f"  «{n.name}»" if n.name else ""
    ot = f"  →{n.output[0]}" if n.output else ""
    return f"{n.op_type:14s}{extra}{sh}{nm}{ot}"


def trace_back(start_tensor, producer, init_names, stop_ops, max_depth=40):
    """start_tensor에서 후방으로, stop_ops/프로젝션/그래프입력 만날 때까지 노드열.
    단일 경로 가정(분기 시 input[0] 우선) — attention 코어는 대체로 선형."""
    chain = []
    t = start_tensor
    for _ in range(max_depth):
        p = producer.get(t)
        if p is None:
            chain.append(("<INPUT>", t))
            break
        chain.append((p, t))
        if p.op_type in stop_ops or is_weight_matmul(p, init_names):
            break
        if not p.input:
            break
        t = p.input[0]
    return list(reversed(chain))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--context-input", default="context")
    ap.add_argument("--infer-shapes", action="store_true",
                    help="onnx.shape_inference로 중간 텐서 shape 채우기 "
                         "(value_info 비어있을 때 — Q2/Q4 레이아웃 판정에 필요)")
    args = ap.parse_args()

    print(f"[load] {args.src} (그래프만, 가중치 미적재)")
    model = onnx.load(args.src, load_external_data=False)
    if args.infer_shapes:
        print("[infer] onnx.shape_inference 실행 중...")
        try:
            from onnx import shape_inference
            model = shape_inference.infer_shapes(model, strict_mode=False,
                                                 data_prop=True)
            print("[infer] 완료")
        except Exception as e:
            print(f"[infer] 실패 ({e}) — shape 없이 진행")
    g = model.graph
    init_names = {i.name for i in g.initializer}
    init_names |= {n.output[0] for n in g.node if n.op_type == "Constant"}
    init_shapes = {i.name: tuple(i.dims) for i in g.initializer}
    init_map = {i.name: i for i in g.initializer}
    producer, consumers = build_maps(g)
    value_infos = {vi.name: vi for vi in g.value_info}
    for vi in list(g.input) + list(g.output):
        value_infos[vi.name] = vi

    # context 도달 집합 (cross 판정용) — 프로젝션 투과 안 함
    ctx_reach, frontier = set(), [args.context_input]
    while frontier:
        t = frontier.pop()
        if t in ctx_reach:
            continue
        ctx_reach.add(t)
        for c in consumers.get(t, []):
            if is_weight_matmul(c, init_names):
                continue
            frontier.extend(c.output)

    softmaxes = [n for n in g.node if n.op_type == "Softmax"]
    print(f"[graph] 노드 {len(g.node)} · Softmax {len(softmaxes)} · "
          f"value_info {len(g.value_info)}")
    print(f"[graph] 입력: {[i.name for i in g.input]}")
    print(f"[graph] value_info 유무: "
          f"{'있음 (shape 추적 가능)' if g.value_info else '없음 (shape 미표시 — onnx.shape_inference 필요)'}")

    # self/cross 각각 첫 그룹 1개씩 진단
    done = {"self": False, "cross": False}
    for sm in softmaxes:
        if all(done.values()):
            break
        # QK: softmax 입력 후방 첫 act×act matmul
        qk = None
        fr, seen = list(sm.input), set()
        while fr and qk is None:
            t = fr.pop()
            p = producer.get(t)
            if p is None or id(p) in seen:
                continue
            seen.add(id(p))
            if p.op_type in ("MatMul", "Gemm") and not is_weight_matmul(p, init_names):
                qk = p
                continue
            if not is_weight_matmul(p, init_names):
                fr.extend(p.input)
        if qk is None:
            continue
        # PV: softmax 출력 전방 첫 act×act matmul
        pv = None
        fr, seen = list(sm.output), set()
        while fr and pv is None:
            t = fr.pop()
            for c in consumers.get(t, []):
                if id(c) in seen:
                    continue
                seen.add(id(c))
                if c.op_type in ("MatMul", "Gemm") and not is_weight_matmul(c, init_names):
                    pv = c
                    break
                if not is_weight_matmul(c, init_names):
                    fr.extend(c.output)
        if pv is None:
            continue

        # q/k 프로젝션까지 후방 추적 (qk의 두 입력 각각)
        def back_to_proj(tensor):
            return trace_back(tensor, producer, init_names,
                              stop_ops={"Softmax"})

        qk_in0, qk_in1 = qk.input[0], qk.input[1]
        chain0 = back_to_proj(qk_in0)
        chain1 = back_to_proj(qk_in1)

        # cross 판정: qk 입력 체인 중 프로젝션의 입력이 context 도달인지
        def chain_proj_is_cross(chain):
            for node, _ in chain:
                if isinstance(node, str):
                    continue
                if is_weight_matmul(node, init_names):
                    return node.input[0] in ctx_reach
            return False
        is_cross = chain_proj_is_cross(chain0) or chain_proj_is_cross(chain1)
        kind = "cross" if is_cross else "self"
        if done[kind]:
            continue
        done[kind] = True

        print("\n" + "=" * 72)
        print(f"  [{kind.upper()} ATTENTION]  Softmax={sm.name}")
        print("=" * 72)

        print(f"\n--- (Q1/Q2/Q3) QK 입력측 체인 0 (→ {qk_in0}) ---")
        for node, t in chain0:
            if isinstance(node, str):
                print(f"    {node}  {t}  {vi_shape(value_infos, t) or ''}")
            else:
                print(f"    {fmt_node(node, init_names, value_infos, init_shapes)}")
        print(f"\n--- QK 입력측 체인 1 (→ {qk_in1}) ---")
        for node, t in chain1:
            if isinstance(node, str):
                print(f"    {node}  {t}  {vi_shape(value_infos, t) or ''}")
            else:
                print(f"    {fmt_node(node, init_names, value_infos, init_shapes)}")

        print(f"\n--- (Q4) QK matmul ---")
        print(f"    {fmt_node(qk, init_names, value_infos, init_shapes)}")
        print(f"    in0={qk_in0} {vi_shape(value_infos, qk_in0) or ''}")
        print(f"    in1={qk_in1} {vi_shape(value_infos, qk_in1) or ''}")

        # QK → Softmax 사이 (scale Mul / bias Add 탐색) — QK 출력부터 Softmax 입력까지만
        print(f"\n--- (Q3/Q7) QK출력 ~ Softmax 사이 ---")
        mid = trace_back(sm.input[0], producer, init_names, stop_ops=set())
        # QK 노드 이후(=출력측)만 남긴다: 체인에서 qk를 찾아 그 다음부터 출력
        qk_pos = next((i for i, (nd, _) in enumerate(mid)
                       if (not isinstance(nd, str)) and nd is qk), None)
        span = mid[qk_pos + 1:] if qk_pos is not None else mid
        if not span:
            print("    (없음 — QK 출력이 곧바로 Softmax 입력)")
        for node, t in span:
            if isinstance(node, str):
                print(f"    {node}  {t}")
            else:
                tag = ""
                if node.op_type == "Mul":
                    tag = "  <<< scale 후보"
                if node.op_type == "Add":
                    tag = "  <<< bias/mask 후보"
                print(f"    {fmt_node(node, init_names, value_infos, init_shapes)}{tag}")

        print(f"\n--- (Q5) Softmax ---")
        print(f"    {fmt_node(sm, init_names, value_infos, init_shapes)}")
        print(f"    in={sm.input[0]} {vi_shape(value_infos, sm.input[0]) or ''}")

        # PV value측 체인
        # PV 두 입력 중 softmax(P)에서 안 온 쪽이 V
        v_side = None
        for i in pv.input:
            # softmax 출력에서 전방 도달하는 입력이 P측
            ch = trace_back(i, producer, init_names, stop_ops={"Softmax"})
            from_sm = any((not isinstance(nd, str)) and nd.op_type == "Softmax"
                          for nd, _ in ch)
            if not from_sm:
                v_side = i
        print(f"\n--- (Q6 일부) PV matmul ---")
        print(f"    {fmt_node(pv, init_names, value_infos, init_shapes)}")
        print(f"    V측 입력={v_side} {vi_shape(value_infos, v_side) or ''}")
        if v_side:
            print(f"    --- V측 체인 (→ v_proj) ---")
            for node, t in back_to_proj(v_side):
                if isinstance(node, str):
                    print(f"      {node}  {t}  {vi_shape(value_infos, t) or ''}")
                else:
                    print(f"      {fmt_node(node, init_names, value_infos, init_shapes)}")

        # PV 출력 ~ output_proj
        print(f"\n--- (Q6) PV출력 ~ output_proj 사이 ---")
        pv_to_oproj = []          # PV 출력과 o_proj 사이 노드들 (fuse 시 MHA에 흡수)
        o_proj = None
        fr, seen, steps = list(pv.output), set(), 0
        while fr and steps < 30:
            t = fr.pop(0)
            for c in consumers.get(t, []):
                if id(c) in seen:
                    continue
                seen.add(id(c))
                steps += 1
                print(f"    {fmt_node(c, init_names, value_infos, init_shapes)}")
                if is_weight_matmul(c, init_names):
                    print(f"      ^^^ output_proj 도달 (중단)")
                    o_proj = c
                    fr = []
                    break
                pv_to_oproj.append(c)
                fr.extend(c.output)

        # ── fuse 경계 요약 (fuse_attention.py가 잡을 핵심) ──
        print(f"\n========== [{kind.upper()}] FUSE 경계 요약 ==========")

        # (A) scale 판정: QK 두 입력의 직전 Mul 상수
        print("\n(A) scale 판정 — QK 입력 직전 Mul 상수값:")
        for label, qk_in in (("q측", qk_in0), ("k측", qk_in1)):
            p = producer.get(qk_in)
            found = False
            # 직전 몇 노드 내 Mul 찾아 상수 입력 읽기
            cur = qk_in
            for _ in range(6):
                pp = producer.get(cur)
                if pp is None:
                    break
                if pp.op_type == "Mul":
                    # Mul의 비활성(상수) 입력 탐색
                    for mi in pp.input:
                        cv = const_value(mi, producer, init_map)
                        if cv is not None:
                            print(f"    {label}: Mul «{pp.name}» 상수 {fmt_const(cv)}")
                            found = True
                            break
                    if found:
                        break
                cur = pp.input[0] if pp.input else None
                if cur is None:
                    break
            if not found:
                print(f"    {label}: 직전 Mul 상수 못 찾음 (scale이 다른 형태이거나 흡수)")

        # (B) MHA 입력으로 줄 텐서 (RoPE 끝난 q/k, v) — 현재 (1,16,S,128) 기대
        print("\n(B) MHA 입력 후보 텐서 (현재 레이아웃):")
        print(f"    query <- {qk_in0}  {vi_shape(value_infos, qk_in0) or ''}")
        print(f"    key   <- {qk_in1}  {vi_shape(value_infos, qk_in1) or ''}")
        print(f"            (k는 전치된 (..,128,S)일 수 있음 — 전치 전 텐서를 써야 MHA 호환)")
        print(f"    value <- {v_side}  {vi_shape(value_infos, v_side) or ''}")
        print(f"    MHA out -> {pv.output[0]} (이후 {len(pv_to_oproj)}개 노드 거쳐 o_proj «{o_proj.name if o_proj else '?'}»)")

        # (C) k 전치 분기점: k측 체인에서 (..,S,128) → (..,128,S) 전치 시작 노드
        print("\n(C) k측 전치 체인 (MHA는 (..,S,128)를 받으므로 전치 전 텐서가 경계):")
        kchain = trace_back(qk_in1, producer, init_names, stop_ops={"Softmax"})
        for node, t in kchain:
            if isinstance(node, str):
                continue
            sh = vi_shape(value_infos, node.output[0]) if node.output else ""
            mark = ""
            if node.op_type == "Transpose":
                perm = next((list(a.ints) for a in node.attribute if a.name == "perm"), None)
                # 마지막 두 축을 바꾸는 전치 = (..,S,128)→(..,128,S) 패턴
                if perm and len(perm) >= 2 and perm[-1] == len(perm) - 2 and perm[-2] == len(perm) - 1:
                    mark = "  <<< k 전치(이 노드부터 제거 후보)"
            print(f"    {node.op_type:12s} «{node.name}»  out={sh}{mark}")

        # (D) 삭제/흡수 대상 노드의 소비자 수 (이 attention 외 사용처 없어야 안전)
        print("\n(D) 흡수 대상 노드 출력의 소비자 수 (1이어야 안전 — 분기 시 주의):")
        for nd in [qk, sm, pv] + pv_to_oproj:
            if nd.output:
                nc = n_consumers(nd.output[0], consumers)
                warn = "" if nc <= 1 else f"  <<< 주의! {nc}곳에서 소비 (분기)"
                print(f"    {nd.op_type:12s} «{nd.name}» 소비자={nc}{warn}")

        # (E) dtype 경계
        qshape = vi_shape(value_infos, qk_in0) or ""
        print(f"\n(E) dtype: QK입력={qshape.split()[-1] if qshape else '?'} "
              f"(MHA 커널 dtype 요구와 대조 — fp32-act면 FLOAT 기대)")

    print("\n[done] 진단 완료. value_info가 '없음'이면 shape_inference 돌린 모델로 재실행 권장.")


if __name__ == "__main__":
    main()
