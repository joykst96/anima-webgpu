"""
VAE 디코더의 mid-block self-attention(분해 SDPA) → com.microsoft.MultiHeadAttention 융합.

목적: VAE attention score (1,1,S,S) 단일버퍼가 1280²(S=25600)서 2.44GB>2GB → OOM.
DiT FA와 동일하게 MHA로 fuse → FA2가 score materialize 안 함(O(S)) → OOM 해소 기대.
(단 head_dim 384에서 WebGPU FA2 발동 여부는 브라우저 실측 필요 — 안 되면 무의미.)

VAE attention 구조 (inspect_vae_attn.py 실측):
  to_qkv conv [1152,384,1,1] → Split 3 → 각 Reshape→Transpose → (1,1,S,384)
  Q,K 각각 scale Mul ×0.2259005 (=（1/√384)^0.5)
  QK MatMul → Softmax(axis=-1) → PV MatMul(V=transpose된 split2) → scaled_dot_product_attention
  → Transpose→Reshape→proj conv [384,384,1,1] → residual
  H(head)=1, head_dim=384, S=토큰수.

치환:
  Q/K/V 각각의 'scale/전치 벗긴 (1,1,S,384) 텐서'를 Reshape (1,S,384)로 → MHA(num_heads=1,
  scale=1/√384) → 출력 (1,S,384). 기존 PV 출력 소비자(Transpose#68)를 MHA 출력을
  같은 레이아웃으로 만들어 재배선.

주의: DiT fuse와 달리 attention 1개뿐 → 식별 분기 불필요. graph-only(원본 .data 공유).

사용 (한 줄):
  python_embeded\python.exe export\fuse_vae_attention.py --src out\vae\qwen_image_vae_decoder_dyn32_2d.onnx --out out\vae\qwen_image_vae_decoder_dyn32_2df.onnx
"""
import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper

SCRIPT_VERSION = "v1"
HEAD_DIM = 384
SCALE_SCALAR = 0.2259005      # Q·K 각각 곱하는 분할 scale
SCALE_TOL = 1e-3


def build_maps(graph):
    producer, consumers = {}, {}
    for n in graph.node:
        for o in n.output:
            producer[o] = n
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    return producer, consumers


def const_val(name, init_map):
    if name not in init_map:
        return None
    return numpy_helper.to_array(init_map[name])


def strip_scale(tname, producer, init_map):
    """tname 생산자가 스칼라 scale Mul이면 활성입력으로 한 단계 벗김. (벗긴텐서, Mul노드)."""
    p = producer.get(tname)
    if p is None or p.op_type != "Mul":
        return tname, None
    for i in p.input:
        cv = const_val(i, init_map)
        if cv is not None and cv.size == 1 \
                and abs(float(cv.ravel()[0]) - SCALE_SCALAR) < SCALE_TOL:
            act = [x for x in p.input if x != i][0]
            return act, p
    return tname, None


def get_shape(name, vimap):
    vi = vimap.get(name)
    if vi is None:
        return None
    return [d.dim_value if d.HasField("dim_value") else -1
            for d in vi.type.tensor_type.shape.dim]


def strip_to_rank4_src(tname, producer, init_names, want_last, vimap):
    """tname에서 후방으로 Reshape/Transpose(활성 단일입력)만 투과해
    shape의 마지막 축이 want_last(=384)인 (1,1,S,384) 텐서까지 내려간다.
    shape 정보로 정지 판단. 반환: (도달텐서, 투과한 노드 리스트)."""
    chain = []
    t = tname
    for _ in range(12):
        s = get_shape(t, vimap)
        # 이미 (1,1,S,384) 도달 (rank4, last=want_last, axis1=1)이면 멈춤
        if s is not None and len(s) == 4 and s[-1] == want_last and s[1] in (1, -1):
            break
        p = producer.get(t)
        if p is None or p.op_type not in ("Reshape", "Transpose"):
            break
        chain.append(p)
        act = [i for i in p.input if i not in init_names]
        if not act:
            break
        t = act[0]
    return t, chain


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"[fuse_vae_attention {SCRIPT_VERSION}] head_dim={HEAD_DIM} "
          f"scale_attr={1.0/np.sqrt(HEAD_DIM):.7f} scale_scalar={SCALE_SCALAR}")
    model = onnx.load(args.src, load_external_data=False)
    graph = model.graph
    # shape 추론 (last축 head_dim 도달 판정용). external data 없이 위상만.
    from onnx import shape_inference
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
        vimap = {vi.name: vi for vi in list(inferred.graph.value_info)
                 + list(inferred.graph.input) + list(inferred.graph.output)}
    except Exception as e:
        print(f"[warn] shape inference 실패({e}) — value_info만 사용")
        vimap = {vi.name: vi for vi in list(graph.value_info)
                 + list(graph.input) + list(graph.output)}
    producer, consumers = build_maps(graph)
    init_names = {i.name for i in graph.initializer}
    init_map = {i.name: i for i in graph.initializer}

    sm = [n for n in graph.node if n.op_type == "Softmax"]
    assert len(sm) == 1, f"VAE attention Softmax가 1개가 아님: {len(sm)}"
    sm = sm[0]
    print(f"[load] {args.src}  Softmax={sm.name}")

    # QK = softmax 입력 생산자, PV = softmax 출력 소비 MatMul
    qk = producer.get(sm.input[0])
    assert qk is not None and qk.op_type == "MatMul", "QK MatMul 못 찾음"
    pv = next((c for c in consumers.get(sm.output[0], []) if c.op_type == "MatMul"), None)
    assert pv is not None, "PV MatMul 못 찾음"

    # Q측 / K측: scale Mul 벗기고, rank-4 src까지
    q_after, q_scale = strip_scale(qk.input[0], producer, init_map)
    k_after, k_scale = strip_scale(qk.input[1], producer, init_map)
    assert q_scale is not None and k_scale is not None, \
        f"scale Mul 식별 실패 (q={q_scale}, k={k_scale}) — scale_scalar 확인"

    q_src, q_chain = strip_to_rank4_src(q_after, producer, init_names, HEAD_DIM, vimap)
    k_src, k_chain = strip_to_rank4_src(k_after, producer, init_names, HEAD_DIM, vimap)

    # V측: PV 입력 중 softmax 아닌 쪽
    v_in = [i for i in pv.input if i != sm.output[0]][0]
    v_src, v_chain = strip_to_rank4_src(v_in, producer, init_names, HEAD_DIM, vimap)

    print(f"[trace] q_src={q_src} k_src={k_src} v_src={v_src}")
    print(f"[trace] 투과체인 q{len(q_chain)} k{len(k_chain)} v{len(v_chain)}")

    # ── MHA 입력: (1,1,S,384) → Reshape (1,S,384) (head=1이라 axis1 제거) ──
    new_nodes = []

    def to_3d(src, role):
        rs = f"mha_vae_{role}"
        shp = f"mha_vae_{role}_shape"
        graph.initializer.append(numpy_helper.from_array(
            np.array([1, -1, HEAD_DIM], dtype=np.int64), shp))
        new_nodes.append(helper.make_node(
            "Reshape", [src, shp], [rs], name=f"mha_vae_{role}_reshape"))
        return rs

    q_in = to_3d(q_src, "q")
    k_in = to_3d(k_src, "k")
    v_in3 = to_3d(v_src, "v")

    mha_out = "mha_vae_out"
    new_nodes.append(helper.make_node(
        "MultiHeadAttention", [q_in, k_in, v_in3], [mha_out],
        name="mha_vae", domain="com.microsoft",
        num_heads=1, scale=float(1.0 / np.sqrt(HEAD_DIM))))

    # ── MHA 출력 (1,S,384)을 기존 PV 출력 소비자가 받게 재배선 ──
    # 기존: PV 출력(scaled_dot_product_attention, (1,1,S,384)) → Transpose#68 → Reshape → proj
    # MHA 출력은 (1,S,384). PV 출력 소비자가 기대하는 레이아웃에 맞춰 Unsqueeze로 (1,1,S,384) 복원.
    pv_out = pv.output[0]
    restore = "mha_vae_out_4d"
    axes = "mha_vae_unsq_axes"
    graph.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), axes))
    new_nodes.append(helper.make_node(
        "Unsqueeze", [mha_out, axes], [restore], name="mha_vae_restore"))
    # pv_out을 쓰던 모든 소비자의 입력을 restore로 교체
    rewired = 0
    for c in consumers.get(pv_out, []):
        for ii, inp in enumerate(c.input):
            if inp == pv_out:
                c.input[ii] = restore
                rewired += 1
    print(f"[rewire] PV 출력 소비자 {rewired}개 → MHA 출력(복원)")

    # ── 죽는 노드 제거: QK, Softmax, PV, q/k scale Mul, q/k/v 투과체인 ──
    dead = {id(qk), id(sm), id(pv), id(q_scale), id(k_scale)}
    for ch in (q_chain, k_chain, v_chain):
        for n in ch:
            dead.add(id(n))
    # 단, 투과체인 노드가 다른 곳에서도 쓰이면 살려야 함 — 소비자 단일성 체크
    safe_dead = set()
    for nid in dead:
        node = next(n for n in graph.node if id(n) == nid)
        # 출력이 다른 살아있는 노드에서 안 쓰이면 제거 가능 (보수적으로 QK/SM/PV/scale은 항상 제거)
        if node in (qk, sm, pv, q_scale, k_scale):
            safe_dead.add(nid)
            continue
        outs = set(node.output)
        other = [c for t in outs for c in consumers.get(t, [])
                 if id(c) not in dead]
        if not other:
            safe_dead.add(nid)

    keep = [n for n in graph.node if id(n) not in safe_dead]
    removed = len(graph.node) - len(keep)
    # 삽입: QK 위치 근처(앞쪽)에 new_nodes 추가
    del graph.node[:]
    graph.node.extend(keep)
    graph.node.extend(new_nodes)
    # 위상정렬 (MHA 입력 의존성 — onnx util)
    try:
        from onnx import version_converter  # noqa
    except Exception:
        pass
    print(f"[fuse] MHA 1개 삽입, 죽은 노드 {removed}개 제거, 새 노드 {len(new_nodes)}개")

    # com.microsoft opset 보장
    if not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))

    # 위상정렬 (onnx.helper 없으면 수동 — 여기선 onnxruntime이 받아주도록 sort)
    sort_topological(graph)

    # external data location을 출력 파일명(.data)으로 통일 — 원본 참조 잔존 방지.
    # graph-only라 데이터 내용/offset/length는 원본과 동일. location 문자열만 교체.
    # (set_external_data는 raw_data 가진 텐서용이라 external 텐서엔 못 씀 → 직접 수정.)
    import os
    new_base = os.path.basename(args.out) + ".data"
    n_ext = 0
    for init in graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            for kv in init.external_data:
                if kv.key == "location":
                    kv.value = new_base
            n_ext += 1
    print(f"[data] external initializer {n_ext}개 location → {new_base}")

    onnx.save(model, args.out)
    print(f"[done] {args.out}  (external data: {new_base})")
    # 원본 .data를 출력.data로 복사 (graph-only라 내용 동일)
    src_data = args.src + ".data"
    dst_data = args.out + ".data"
    if os.path.exists(src_data) and os.path.abspath(src_data) != os.path.abspath(dst_data):
        import shutil
        shutil.copyfile(src_data, dst_data)
        print(f"[data] 원본 .data 복사: {src_data} → {dst_data}")
    print("[next] 서버에 출력.onnx + 출력.onnx.data 업로드 → 브라우저 1280² 디코드 실측")


def sort_topological(graph):
    """간단 위상정렬: 입력이 모두 준비된 노드부터 배치."""
    produced = {i.name for i in graph.initializer}
    produced |= {i.name for i in graph.input}
    nodes = list(graph.node)
    ordered, pending = [], nodes
    while pending:
        progress = False
        still = []
        for n in pending:
            if all((inp in produced or inp == "") for inp in n.input):
                ordered.append(n)
                produced.update(n.output)
                progress = True
            else:
                still.append(n)
        pending = still
        if not progress:
            # 사이클/미해결 — 남은 것 그대로 붙임 (ORT가 검증)
            ordered.extend(pending)
            break
    del graph.node[:]
    graph.node.extend(ordered)


if __name__ == "__main__":
    main()
