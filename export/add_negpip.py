"""
양자화된 Anima DiT ONNX에 NegPip 입력을 삽입하는 그래프 수술.

NegPip (참조: pamparamm/ComfyUI-ppm anima_negpip.py):
  cross-attention의 v_proj에 들어가는 context에만 토큰별 부호(±1)를 곱한다.
  k_proj는 원본 context를 그대로 사용 → 음수 토큰의 v가 부호 반전되어
  attention 출력에서 해당 개념이 감산됨. CFG=1(터보)에서 부정 프롬프트 효과.

그래프 변경 (전부 합쳐 노드 2개 + 입력 1개):
  1. 새 입력 negpip_mask: (1, CTX_LEN, 1) fp16. 전부 1이면 원본과 완전 동일.
  2. Cast(negpip_mask → context 활성 dtype) + Mul(context_act, mask) 삽입
  3. cross-attn v_proj로 식별된 MatMulNBits/MatMul 노드의 A 입력만 재배선

식별 (익명 그래프 — 인수인계 검증 사실 5):
  앵커: context 입력에서 전방 도달 가능한 가중치-matmul 중 K=context_dim.
        Anima에선 cross k/v만 K=1024 (self/q/mlp는 2048/8192) → 유일.
  k vs v 구분 (전방 도달 분석):
    후보 출력에서 elementwise/shape 노드를 투과해 처음 만나는 attention matmul:
      k_proj → MatMul(QK^T) → … → Softmax        (반대편 입력에 Softmax 없음,
                                                    matmul 출력이 Softmax로 흐름)
      v_proj → MatMul(P·V)   (반대편 입력이 Softmax 출력에서 옴)

파이프라인 위치: quantize_dit.py 다음, shard_onnx_data.py 이전.
  export → quantize(--accuracy-level 4) → add_negpip → shard

사용 예 (한 줄):
  python export/add_negpip.py --src out/dit/anima_dit_dyn32_q8a4.onnx --out out/dit/anima_dit_dyn32_q8a4n.onnx
"""
import argparse
import os

import onnx
from onnx import TensorProto, helper

SCRIPT_VERSION = "v2 (fuse-meta)"

# 식별 시 "투과"하는 노드 (값을 변형해도 attention 구조를 바꾸지 않는 것들)
WEIGHT_MATMULS = {"MatMulNBits", "MatMul", "Gemm"}


def build_maps(graph):
    producer = {}          # tensor name -> node
    consumers = {}         # tensor name -> [node]
    for n in graph.node:
        for o in n.output:
            producer[o] = n
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    return producer, consumers


def is_weight_matmul(node, init_names):
    """가중치(initializer/Constant)를 B 입력으로 갖는 matmul = 프로젝션 레이어."""
    if node.op_type == "MatMulNBits":
        return True
    if node.op_type in ("MatMul", "Gemm") and len(node.input) >= 2:
        return node.input[1] in init_names
    return False


def get_kn(node, init_shapes):
    """프로젝션 matmul의 (K, N) 추출."""
    if node.op_type == "MatMulNBits":
        a = {x.name: x for x in node.attribute}
        return a["K"].i, a["N"].i
    shp = init_shapes.get(node.input[1])
    if shp and len(shp) == 2:
        return shp[0], shp[1]
    return None, None


def forward_first_matmuls(start_node, consumers, init_names, limit=4000):
    """start_node 출력에서 전방 탐색, 처음 만나는 '비-프로젝션' MatMul들을 수집.

    프로젝션(가중치 matmul)을 만나면 그 가지는 중단 (다음 레이어로 넘어간 것).
    activation×activation MatMul = attention matmul 후보.
    반환: [(matmul_node, 도달한 입력 인덱스)]
    """
    hits, seen, frontier = [], set(), list(start_node.output)
    steps = 0
    while frontier and steps < limit:
        t = frontier.pop()
        for c in consumers.get(t, []):
            steps += 1
            key = (id(c), t)
            if key in seen:
                continue
            seen.add(key)
            if c.op_type in ("MatMul", "Gemm") and not is_weight_matmul(c, init_names):
                idx = [i for i, x in enumerate(c.input) if x == t]
                for i in idx:
                    hits.append((c, i))
                continue                       # attention matmul에서 가지 중단
            if c.op_type == "MatMulNBits" or is_weight_matmul(c, init_names):
                continue                       # 다음 프로젝션 — 가지 중단
            if c.op_type == "Softmax":
                continue                       # k가 softmax에 직접 닿는 경우는 없음
            frontier.extend(c.output)          # 투과
    return hits


def reaches_softmax_forward(node, consumers, init_names, limit=4000):
    """노드 출력이 (프로젝션을 넘지 않고) Softmax에 도달하는가."""
    seen, frontier = set(), list(node.output)
    steps = 0
    while frontier and steps < limit:
        t = frontier.pop()
        for c in consumers.get(t, []):
            steps += 1
            if id(c) in seen:
                continue
            seen.add(id(c))
            if c.op_type == "Softmax":
                return True
            if c.op_type == "MatMulNBits" or is_weight_matmul(c, init_names):
                continue
            frontier.extend(c.output)
    return False


def comes_from_softmax(tensor, producer, init_names, limit=4000):
    """텐서가 (프로젝션을 넘지 않고) Softmax 출력에서 유래하는가 (후방 탐색)."""
    seen, frontier = set(), [tensor]
    steps = 0
    while frontier and steps < limit:
        t = frontier.pop()
        p = producer.get(t)
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        steps += 1
        if p.op_type == "Softmax":
            return True
        if p.op_type == "MatMulNBits" or is_weight_matmul(p, init_names):
            continue
        frontier.extend(p.input)
    return False


def identify_via_meta(graph, meta_path, init_names, init_shapes):
    """fuse_meta.json 기반 cross v/k_proj 식별 (fuse된 그래프용).

    fuse 후엔 Softmax가 MHA로 흡수돼 기존 앵커가 죽으므로, fuse_attention.py가
    남긴 메타에서 cross attention의 v_proj/k_proj 노드명을 직접 읽는다.
    반환: (v_nodes, k_nodes) — 둘 다 그래프 노드 객체 리스트. 없으면 (None, None).
    """
    import json
    import os
    if not os.path.exists(meta_path):
        return None, None
    try:
        meta = json.load(open(meta_path))
    except Exception as e:
        print(f"[meta] 로드 실패 ({e}) — Softmax 앵커 폴백")
        return None, None

    byname = {n.name: n for n in graph.node}
    cross = [a for a in meta.get("attentions", []) if a.get("kind") == "cross"]
    if not cross:
        print("[meta] cross attention 항목 없음 — 폴백")
        return None, None

    v_nodes, k_nodes, missing = [], [], []
    for a in cross:
        vn, kn = a.get("v_proj"), a.get("k_proj")
        if vn in byname and kn in byname:
            v_nodes.append(byname[vn])
            k_nodes.append(byname[kn])
        else:
            missing.append((vn, kn))
    if missing:
        print(f"[meta] proj 노드 {len(missing)}개가 그래프에 없음 — 메타/그래프 불일치, 폴백")
        return None, None

    # 검증: v_proj가 K=ctx_dim 프로젝션인지(있으면). 메타 신뢰가 우선이라 경고만.
    bad = 0
    for n in v_nodes:
        K, _ = get_kn(n, init_shapes)
        if K is not None and K not in (1024,):   # Anima cross k/v는 K=1024
            bad += 1
    if bad:
        print(f"[meta] (경고) v_proj 중 K≠1024 {bad}개 — 메타 확인 권장 (계속 진행)")
    print(f"[meta] fuse_meta 기반 식별: cross v_proj {len(v_nodes)} / k_proj {len(k_nodes)}")
    return v_nodes, k_nodes


def identify_via_softmax(graph, init_names, init_shapes, ctx_input, ctx_dim):
    """기존 Softmax 앵커 식별 (비-fuse 모델 폴백). 원본 로직 그대로.
    반환: (v_nodes, k_nodes)."""
    producer, consumers = build_maps(graph)

    ctx_reach, frontier = set(), [ctx_input]
    while frontier:
        t = frontier.pop()
        if t in ctx_reach:
            continue
        ctx_reach.add(t)
        for c in consumers.get(t, []):
            if c.op_type == "MatMulNBits" or is_weight_matmul(c, init_names):
                continue
            frontier.extend(c.output)

    cands = []
    for n in graph.node:
        if not is_weight_matmul(n, init_names):
            continue
        if n.input[0] not in ctx_reach:
            continue
        K, N = get_kn(n, init_shapes)
        if K == ctx_dim:
            cands.append(n)
    print(f"[anchor] context 소비 K={ctx_dim} 프로젝션: {len(cands)}개")
    assert cands and len(cands) % 2 == 0, \
        "cross k/v 후보 수가 짝수가 아님 — 그래프 구조 확인 필요"

    k_nodes, v_nodes, unknown = [], [], []
    for n in cands:
        hits = forward_first_matmuls(n, consumers, init_names)
        is_v = any(comes_from_softmax(mm.input[1 - idx], producer, init_names)
                   for mm, idx in hits)
        is_k = (not is_v) and any(reaches_softmax_forward(mm, consumers, init_names)
                                  for mm, _ in hits)
        (v_nodes if is_v else k_nodes if is_k else unknown).append(n)

    print(f"[classify] v_proj {len(v_nodes)} / k_proj {len(k_nodes)} / 불명 {len(unknown)}")
    assert not unknown, f"분류 불가 노드 {len(unknown)}개 — 수동 확인 필요"
    assert len(v_nodes) == len(k_nodes), "k/v 개수 불일치 — 분류 오류 의심"
    return v_nodes, k_nodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="양자화된 DiT .onnx")
    p.add_argument("--out", required=True, help="출력 .onnx (새 파일명 — immutable 캐시)")
    p.add_argument("--context-input", default="context")
    p.add_argument("--mask-name", default="negpip_mask")
    p.add_argument("--fuse-meta", default=None, dest="fuse_meta",
                   help="fuse_attention.py가 남긴 .fuse_meta.json 경로. "
                        "기본: <src>.fuse_meta.json. 있으면 메타 기반 식별, "
                        "없으면 Softmax 앵커 폴백(비-fuse 모델).")
    args = p.parse_args()

    print(f"[add_negpip {SCRIPT_VERSION}]")
    print(f"[load] {args.src}")
    model = onnx.load(args.src)
    graph = model.graph
    init_names = {i.name for i in graph.initializer}
    init_names |= {n.output[0] for n in graph.node if n.op_type == "Constant"}
    init_shapes = {i.name: tuple(i.dims) for i in graph.initializer}
    producer, consumers = build_maps(graph)

    # context 입력 확인 + 시퀀스 길이/차원
    ctx_in = next((i for i in graph.input if i.name == args.context_input), None)
    assert ctx_in is not None, f"입력 '{args.context_input}' 없음"
    dims = [d.dim_value for d in ctx_in.type.tensor_type.shape.dim]
    ctx_len, ctx_dim = dims[1], dims[2]
    ctx_dtype = ctx_in.type.tensor_type.elem_type
    print(f"[ctx] {args.context_input}: len={ctx_len}, dim={ctx_dim}, "
          f"dtype={TensorProto.DataType.Name(ctx_dtype)}")

    # context에서 전방 도달 가능한 텐서 집합 (프로젝션 투과 안 함)
    # ── 식별: 메타 우선, 실패 시 Softmax 앵커 폴백 ──
    meta_path = args.fuse_meta or (args.src + ".fuse_meta.json")
    v_nodes, k_nodes = identify_via_meta(graph, meta_path, init_names, init_shapes)
    if v_nodes is None:
        print("[identify] 메타 없음/불일치 → Softmax 앵커 폴백 (비-fuse 모델 경로)")
        v_nodes, k_nodes = identify_via_softmax(
            graph, init_names, init_shapes, args.context_input, ctx_dim)
    else:
        print("[identify] 메타 기반 식별 사용 (fuse된 모델)")
    assert len(v_nodes) == len(k_nodes), "k/v 개수 불일치"

    # ── 삽입 ──
    # (producer 맵은 위 build_maps(graph)에서 이미 생성됨 — act_dtype이 사용)

    # 새 입력 negpip_mask (context와 동일 dtype)
    mask_in = helper.make_tensor_value_info(
        args.mask_name, ctx_dtype, [1, ctx_len, 1])
    graph.input.append(mask_in)

    # v_proj들이 소비하는 활성 텐서별로 Mul 1개씩 (보통 1종)
    v_inputs = sorted({n.input[0] for n in v_nodes})
    print(f"[rewire] v_proj 활성 텐서 종류: {len(v_inputs)}")

    new_nodes = []
    # 활성 dtype 판정: 생산자 체인을 거슬러 올라가 Cast의 to 속성을 찾는다.
    # (v_proj 입력의 직접 생산자가 Reshape 등일 수 있음 — fp32-act 그래프에선
    #  context fp16 → Cast fp32 → … 형태라 체인 어딘가에 Cast가 있음)
    def act_dtype(tname):
        t = tname
        for _ in range(64):
            p_ = producer.get(t)
            if p_ is None:
                break                      # 그래프 입력 도달 → context dtype
            if p_.op_type == "Cast":
                return next(a.i for a in p_.attribute if a.name == "to")
            if not p_.input:
                break
            t = p_.input[0]
        return ctx_dtype

    cast_cache = {}
    for ti, tname in enumerate(v_inputs):
        dt = act_dtype(tname)
        if dt not in cast_cache:
            if dt == ctx_dtype:
                cast_cache[dt] = args.mask_name
            else:
                cn = f"{args.mask_name}_cast{dt}"
                new_nodes.append(helper.make_node(
                    "Cast", [args.mask_name], [cn],
                    name=f"negpip_cast_{dt}", to=dt))
                cast_cache[dt] = cn
        mul_out = f"{tname}_negpip"
        new_nodes.append(helper.make_node(
            "Mul", [tname, cast_cache[dt]], [mul_out], name=f"negpip_mul_{ti}"))
        for n in v_nodes:
            if n.input[0] == tname:
                n.input[0] = mul_out

    # 위상 순서 유지: 새 노드들을 첫 v_proj 노드 앞에 삽입
    first_v_idx = min(i for i, n in enumerate(graph.node) if n in v_nodes)
    for j, nn in enumerate(new_nodes):
        graph.node.insert(first_v_idx + j, nn)

    print(f"[insert] 노드 {len(new_nodes)}개 추가, v_proj {len(v_nodes)}개 재배선")

    # ── 저장 (quantize_dit와 동일한 크기 기준) ──
    approx = sum(len(i.raw_data) for i in graph.initializer if i.HasField("raw_data"))
    print(f"[save] 예상 크기 {approx / 2**30:.2f} GiB")
    if approx < 1.8 * 2**30:
        onnx.save_model(model, args.out)
        print(f"[save] 단일 파일 저장: {args.out}")
    else:
        data_name = os.path.basename(args.out) + ".data"
        onnx.save_model(model, args.out, save_as_external_data=True,
                        all_tensors_to_one_file=True, location=data_name,
                        size_threshold=1024)
        print(f"[save] external data 저장: {args.out} + {data_name}")
    print("[done] 다음 단계: shard_onnx_data.py로 샤딩 (새 파일명 필수)")


if __name__ == "__main__":
    main()
