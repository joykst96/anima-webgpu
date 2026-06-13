"""
분해된 SDPA(QK→Softmax→PV)를 com.microsoft.MultiHeadAttention 1개로 융합.

목적: ORT WebGPU 네이티브 EP(jspi 빌드)의 FlashAttention 2 커널 발동.
  FA2는 MultiHeadAttention/PackedMHA/GQA 에만 붙음 (표준 Attention opset23엔 아직 X).
  → com.microsoft.MultiHeadAttention(3D 입력) 으로 치환. 발동은 브라우저 profiling으로 확인.

식별 (익명 그래프 — 검증 사실 5. add_negpip/add_lora_slots의 Softmax 앵커 재사용):
  Softmax 앵커 → QK(후방 act×act matmul) / PV(전방 act×act matmul)
  MHA query = QK.in0에서 scale Mul(스칼라 ≈0.297302) 한 단계 벗긴 (1,16,S,128)
  MHA key   = QK.in1에서 scale Mul + 전치체인(Reshape/Transpose/Reshape) 벗긴 (1,16,S,128)
  MHA value = PV의 V측 (1,16,S,128)
  cross 판정 = key측 후방 proj 입력이 context 도달
  MHA out   = PV 출력 자리 → 이후 transpose+reshape 건너뛰고 o_proj 입력에 직결

self/cross 비대칭 주의:
  self  q/k: proj→reshape→transpose→RoPE→scale
  cross q/k: proj→reshape→q_norm/k_norm→(k는 전치)→scale  (RoPE 없음)
  → q_norm/RoPE는 MHA 입력 경계 '안'(앞)에 남는다. 식별은 scale Mul 기준이라 공통.

scale: q,k에 (1/128)^0.25=0.297302 쪼개 곱해짐 → 곱하면 1/√128.
  fuse 시 scale Mul 2개 제거하고 MHA scale 속성 = 1/√head_dim 통째로.

merge (A안): MHA는 3D (1,S,hidden) 입력. (1,16,S,128) q/k/v 각각
  Transpose[0,2,1,3]→Reshape(1,S,hidden) 삽입. out(1,S,hidden)→o_proj.

파이프라인 위치: quantize_dit.py 다음, **add_negpip.py 이전** (negpip이 v_proj
  재배선하면 식별이 꼬임). negpip/lora_slots는 이후 MHA 앵커 기반으로 갱신 필요(별도).

사용 (한 줄):
  python export/fuse_attention.py --src out/dit/anima_dit_dyn32_q8a4.onnx --out out/dit/anima_dit_dyn32_q8a4f.onnx
"""
import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SCRIPT_VERSION = "v1"
WEIGHT_MATMULS = {"MatMulNBits", "MatMul", "Gemm"}
SCALE_TOL = 1e-3          # scale Mul 상수 식별 허용오차
MERGE_PERMS = ([0, 2, 1, 3],)   # head split/merge transpose perm


# ───────────────────────── 그래프 유틸 (add_negpip 재사용) ─────────────────────────

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


def const_value(name, producer, init_map):
    if name in init_map:
        try:
            return numpy_helper.to_array(init_map[name])
        except Exception:
            return None
    p = producer.get(name)
    if p is not None and p.op_type == "Constant":
        for a in p.attribute:
            if a.name == "value":
                try:
                    return numpy_helper.to_array(a.t)
                except Exception:
                    return None
    return None


def is_scale_mul(node, producer, init_map, scale_scalar):
    """node가 스칼라 상수(≈scale_scalar)를 곱하는 Mul인지. 맞으면 활성입력 텐서 반환."""
    if node.op_type != "Mul":
        return None
    act_in, const_ok = None, False
    for i in node.input:
        cv = const_value(i, producer, init_map)
        if cv is not None and np.asarray(cv).size == 1:
            if abs(float(np.asarray(cv).ravel()[0]) - scale_scalar) < SCALE_TOL:
                const_ok = True
        else:
            act_in = i
    return act_in if (const_ok and act_in is not None) else None


def strip_transpose_chain(tensor, producer, init_names, value_infos, head_dim, limit=12):
    """k측 전치 체인(Reshape/Transpose/Mul 등 형상/스칼라 연산)을 후방으로 투과해
    (1, H, S, D) 레이아웃 텐서에 도달하면 멈춘다. 가중치 matmul/Softmax 만나도 중단.
    반환: (도달 텐서, 통과한 노드 리스트)."""
    passed = []
    t = tensor
    for _ in range(limit):
        # 이미 (1,H,S,head_dim)면 여기가 경계 — 더 벗기지 않음
        sh = shape_of(value_infos, t)
        if is_bhsd(sh, head_dim):
            break
        p = producer.get(t)
        if p is None:
            break
        if is_weight_matmul(p, init_names) or p.op_type == "Softmax":
            break
        if p.op_type in ("Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity"):
            passed.append(p)
            t = p.input[0]
            continue
        break
    return t, passed


def shape_of(value_infos, name):
    vi = value_infos.get(name)
    if vi is None:
        return None
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
    return dims


def is_bhsd(shape, head_dim):
    """(1, H, S, D) 레이아웃인가 — rank 4, 마지막 차원 == head_dim."""
    return shape is not None and len(shape) == 4 and shape[-1] == head_dim


# ───────────────────────── attention 그룹 식별 ─────────────────────────

def find_groups(graph, producer, consumers, init_names, init_map,
                value_infos, ctx_input, scale_scalar, head_dim):
    groups = []
    ctx_reach = reachable_fwd([ctx_input], consumers, init_names)

    for sm in graph.node:
        if sm.op_type != "Softmax":
            continue
        qk = back_act_matmul(sm.input[0], producer, init_names)
        pv = fwd_act_matmul(sm.output, consumers, init_names)
        if qk is None or pv is None:
            continue

        # q측 / k측: scale Mul 벗기기
        def strip_scale(tname):
            p = producer.get(tname)
            if p is not None:
                act = is_scale_mul(p, producer, init_map, scale_scalar)
                if act is not None:
                    return act, p
            return tname, None

        q_src, q_scale = strip_scale(qk.input[0])
        k_after_scale, k_scale = strip_scale(qk.input[1])
        # k측 전치 체인 벗기기 → (1,H,S,D)
        k_src, k_chain = strip_transpose_chain(k_after_scale, producer, init_names,
                                               value_infos, head_dim)

        # v측 (PV의 softmax 아닌 입력)
        v_src = None
        for i in pv.input:
            if not comes_from_softmax(i, producer, init_names):
                v_src = i
        if v_src is None:
            continue

        # cross 판정: k_src 후방 proj 입력이 context 도달?
        k_proj = back_first_projection(k_src, producer, init_names)
        is_cross = (k_proj is not None and k_proj.input[0] in ctx_reach)

        # q/v proj도 역추적 (메타용 — negpip은 v_proj, lora는 q/k/v/o 전부 필요)
        q_proj = back_first_projection(q_src, producer, init_names)
        v_proj = back_first_projection(v_src, producer, init_names)

        # MHA out 경계: PV 출력 → transpose/reshape 건너뛰고 o_proj
        o_proj, post_nodes = fwd_to_oproj(pv, consumers, init_names)
        if o_proj is None:
            continue

        groups.append(dict(
            kind="cross" if is_cross else "self",
            softmax=sm, qk=qk, pv=pv,
            q_src=q_src, k_src=k_src, v_src=v_src,
            q_scale=q_scale, k_scale=k_scale, k_chain=k_chain,
            o_proj=o_proj, post_nodes=post_nodes,
            q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
            q_shape=shape_of(value_infos, q_src),
            k_shape=shape_of(value_infos, k_src),
            v_shape=shape_of(value_infos, v_src),
        ))
    return groups


def reachable_fwd(start, consumers, init_names):
    reach, frontier = set(), list(start)
    while frontier:
        t = frontier.pop()
        if t in reach:
            continue
        reach.add(t)
        for c in consumers.get(t, []):
            if is_weight_matmul(c, init_names):
                continue
            frontier.extend(c.output)
    return reach


def back_act_matmul(tensor, producer, init_names, limit=4000):
    seen, frontier, steps = set(), [tensor], 0
    while frontier and steps < limit:
        t = frontier.pop()
        p = producer.get(t)
        if p is None or id(p) in seen:
            continue
        seen.add(id(p)); steps += 1
        if p.op_type in ("MatMul", "Gemm") and not is_weight_matmul(p, init_names):
            return p
        if not is_weight_matmul(p, init_names):
            frontier.extend(p.input)
    return None


def fwd_act_matmul(start_outputs, consumers, init_names, limit=4000):
    seen, frontier, steps = set(), list(start_outputs), 0
    while frontier and steps < limit:
        t = frontier.pop()
        for c in consumers.get(t, []):
            if id(c) in seen:
                continue
            seen.add(id(c)); steps += 1
            if c.op_type in ("MatMul", "Gemm") and not is_weight_matmul(c, init_names):
                return c
            if not is_weight_matmul(c, init_names):
                frontier.extend(c.output)
    return None


def comes_from_softmax(tensor, producer, init_names, limit=4000):
    seen, frontier, steps = set(), [tensor], 0
    while frontier and steps < limit:
        t = frontier.pop()
        p = producer.get(t)
        if p is None or id(p) in seen:
            continue
        seen.add(id(p)); steps += 1
        if p.op_type == "Softmax":
            return True
        if is_weight_matmul(p, init_names):
            continue
        frontier.extend(p.input)
    return False


def back_first_projection(tensor, producer, init_names, limit=512):
    frontier, seen = [tensor], set()
    while frontier and len(seen) < limit:
        t = frontier.pop()
        p = producer.get(t)
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        if is_weight_matmul(p, init_names):
            return p
        if p.op_type == "Softmax":
            return None
        frontier.extend(p.input)
    return None


def fwd_to_oproj(pv, consumers, init_names, limit=30):
    """PV 출력 전방으로 형상 노드 투과, 첫 가중치 matmul(o_proj) 도달.
    반환: (o_proj, [중간 노드들])."""
    post, frontier, seen, steps = [], list(pv.output), set(), 0
    while frontier and steps < limit:
        t = frontier.pop(0)
        for c in consumers.get(t, []):
            if id(c) in seen:
                continue
            seen.add(id(c)); steps += 1
            if is_weight_matmul(c, init_names):
                return c, post
            post.append(c)
            frontier.extend(c.output)
    return None, post


# ───────────────────────── 융합 ─────────────────────────

def fuse_group(graph, g, hidden, num_heads, scale_attr, idx, consumers):
    """그룹 1개를 MHA로 치환. 새 노드 리스트 반환 (삽입은 호출측)."""
    kind = g["kind"]
    tag = f"{kind}_{idx}"
    new_nodes = []

    # (A) q/k/v (1,H,S,D) → Transpose[0,2,1,3] → Reshape(1,S,hidden)
    def to_bsh(src, role):
        tr = f"mha_{tag}_{role}_tr"
        rs = f"mha_{tag}_{role}_rs"
        shape_init = f"mha_{tag}_{role}_shape"
        new_nodes.append(helper.make_node(
            "Transpose", [src], [tr], perm=[0, 2, 1, 3],
            name=f"mha_{tag}_{role}_transpose"))
        # Reshape 목표 (1, -1, hidden) — S 동적이라 -1
        graph.initializer.append(numpy_helper.from_array(
            np.array([1, -1, hidden], dtype=np.int64), shape_init))
        new_nodes.append(helper.make_node(
            "Reshape", [tr, shape_init], [rs],
            name=f"mha_{tag}_{role}_reshape"))
        return rs

    q_in = to_bsh(g["q_src"], "q")
    k_in = to_bsh(g["k_src"], "k")
    v_in = to_bsh(g["v_src"], "v")

    # (B) MHA 노드
    mha_out = f"mha_{tag}_out"
    mha = helper.make_node(
        "MultiHeadAttention", [q_in, k_in, v_in], [mha_out],
        name=f"mha_{tag}", domain="com.microsoft",
        num_heads=num_heads, scale=scale_attr)
    new_nodes.append(mha)

    # (C) MHA out(1,S,hidden) → o_proj 입력에 직결.
    #     기존엔 PV→transpose→reshape→o_proj 였음. o_proj의 입력을 mha_out으로 교체.
    o_proj = g["o_proj"]
    old_oin = o_proj.input[0]
    for ii, inp in enumerate(o_proj.input):
        if inp == old_oin:
            o_proj.input[ii] = mha_out
            break

    return new_nodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--context-input", default="context")
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--scale-scalar", type=float, default=0.297302,
                   help="QK 직전 q/k에 곱해진 스칼라 (제곱이 1/head_dim). 식별용.")
    p.add_argument("--dump", action="store_true")
    args = p.parse_args()

    head_dim = args.head_dim
    num_heads = args.num_heads
    scale_attr = float(1.0 / np.sqrt(head_dim))   # MHA scale 속성 (통째)

    print(f"[fuse_attention {SCRIPT_VERSION}] heads={num_heads} head_dim={head_dim} "
          f"scale={scale_attr:.7f}")
    print(f"[load] {args.src}")
    # shape 식별 필요 → external data 없이 로드 후 shape_inference
    model = onnx.load(args.src, load_external_data=False)
    try:
        from onnx import shape_inference
        model = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception as e:
        print(f"[warn] shape_inference 실패 ({e}) — shape 검증 제한됨")
    g = model.graph
    init_names = {i.name for i in g.initializer}
    init_names |= {n.output[0] for n in g.node if n.op_type == "Constant"}
    init_map = {i.name: i for i in g.initializer}
    producer, consumers = build_maps(g)
    value_infos = {vi.name: vi for vi in g.value_info}
    for vi in list(g.input) + list(g.output):
        value_infos[vi.name] = vi

    groups = find_groups(g, producer, consumers, init_names, init_map,
                         value_infos, args.context_input, args.scale_scalar, head_dim)
    self_g = [x for x in groups if x["kind"] == "self"]
    cross_g = [x for x in groups if x["kind"] == "cross"]
    print(f"[identify] attention 그룹: self {len(self_g)} / cross {len(cross_g)}")
    assert self_g and len(self_g) == len(cross_g), \
        f"self/cross 개수 불일치 ({len(self_g)}/{len(cross_g)}) — 식별 확인 필요"

    # ── 노드 정합성 단위 체크 (fuse 전) ──
    print("[check] 노드 정합성:")
    bad = 0
    for x in groups:
        for role, sh in (("q", x["q_shape"]), ("k", x["k_shape"]), ("v", x["v_shape"])):
            if sh is not None and not is_bhsd(sh, head_dim):
                print(f"  [!] {x['kind']} {role}_src shape {sh} — (1,H,S,{head_dim}) 아님")
                bad += 1
        # 흡수 대상 소비자 수 == 1 확인
        for nd in [x["qk"], x["softmax"], x["pv"]] + x["post_nodes"]:
            if nd.output and len(consumers.get(nd.output[0], [])) > 1:
                print(f"  [!] {nd.op_type} «{nd.name}» 소비자>1 — 분기, 흡수 시 주의")
                bad += 1
    if bad:
        print(f"[check] 경고 {bad}건 — shape 미표시(동적)면 무시 가능, 레이아웃 불일치면 중단 권장")
    else:
        print("[check] 통과 (q/k/v 전부 (1,H,S,head_dim), 흡수 노드 소비자=1)")

    # ── 융합 실행 (+ 메타 수집) ──
    def nm(node):
        return node.name if node is not None else None

    all_new = []
    meta_attns = []
    for i, x in enumerate(self_g):
        tag = f"self_self{i}"
        all_new += [(x, fuse_group(g, x, head_dim * num_heads, num_heads, scale_attr,
                                   f"self{i}", consumers))]
        meta_attns.append(dict(
            kind="self", mha=f"mha_{tag}",
            q_proj=nm(x.get("q_proj")), k_proj=nm(x.get("k_proj")),
            v_proj=nm(x.get("v_proj")), o_proj=nm(x.get("o_proj")),
        ))
    for i, x in enumerate(cross_g):
        tag = f"cross_cross{i}"
        all_new += [(x, fuse_group(g, x, head_dim * num_heads, num_heads, scale_attr,
                                   f"cross{i}", consumers))]
        meta_attns.append(dict(
            kind="cross", mha=f"mha_{tag}",
            q_proj=nm(x.get("q_proj")), k_proj=nm(x.get("k_proj")),
            v_proj=nm(x.get("v_proj")), o_proj=nm(x.get("o_proj")),
        ))

    # 새 노드를 o_proj 앞에 삽입 (위상 순서) + 죽은 노드 제거
    node_idx = {id(n): k for k, n in enumerate(g.node)}
    insert_plan = []
    dead = set()
    for x, nns in all_new:
        at = node_idx[id(x["o_proj"])]
        insert_plan.append((at, nns))
        # 죽는 노드: QK, Softmax, PV, post_nodes, scale Mul 2개, k 전치 체인
        for nd in [x["qk"], x["softmax"], x["pv"]] + x["post_nodes"] + x["k_chain"]:
            dead.add(id(nd))
        for s in (x["q_scale"], x["k_scale"]):
            if s is not None:
                dead.add(id(s))

    # 삽입 (역순으로 인덱스 안정화)
    for at, nns in sorted(insert_plan, key=lambda t: -t[0]):
        for j, nn in enumerate(nns):
            g.node.insert(at + j, nn)

    # 죽은 노드 제거 (다른 곳에서 출력이 안 쓰이는 것만 — 안전)
    rebuilt_producer, rebuilt_consumers = build_maps(g)
    removed = 0
    keep = []
    for n in g.node:
        if id(n) in dead:
            # 출력이 살아있는 노드에 의해 소비되면 제거 보류 (안전)
            still_used = any(len(rebuilt_consumers.get(o, [])) > 0
                             and any(id(c) not in dead for c in rebuilt_consumers.get(o, []))
                             for o in n.output)
            if still_used:
                keep.append(n)
            else:
                removed += 1
            continue
        keep.append(n)
    del g.node[:]
    g.node.extend(keep)

    # com.microsoft opset import 보장
    has_ms = any(o.domain == "com.microsoft" for o in model.opset_import)
    if not has_ms:
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))

    print(f"[fuse] MHA {len(self_g)+len(cross_g)}개 삽입, 죽은 노드 {removed}개 제거")

    # ── 저장 (graph-only 편집) ──
    # 원본 initializer는 data_location=EXTERNAL로 원본 .data를 가리킴.
    # MHA 삽입은 그래프 위상만 바꾸고 가중치를 안 건드리므로(새 initializer는
    # reshape용 작은 int64 shape 상수뿐), save_model만 하면 external 참조가 보존된다.
    # 단 전제: 원본 .data 파일이 출력과 같은 디렉토리에 같은 이름으로 존재해야 함
    # (external_data location은 상대경로). add_lora_slots.py와 동일 방식.
    src_data = args.src + ".data"
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    src_data_base = os.path.basename(src_data)
    if os.path.exists(src_data) and not os.path.exists(os.path.join(out_dir, src_data_base)):
        print(f"[warn] 원본 .data가 출력 디렉토리에 없음. external 참조가 깨질 수 있음.")
        print(f"       원본: {src_data}")
        print(f"       출력 dir: {out_dir} (여기에 {src_data_base} 필요)")
        print(f"       → 같은 디렉토리에 출력하거나 .data를 복사할 것.")

    # 새로 추가한 작은 shape 상수만 raw_data로 인라인됨 — 크기 미미
    n_inline = sum(1 for i in g.initializer if i.HasField("raw_data") and i.raw_data)
    n_external = sum(1 for i in g.initializer
                     if i.data_location == TensorProto.EXTERNAL)
    print(f"[save] initializer: external(원본 .data 참조) {n_external}개, "
          f"인라인(신규 shape 상수) {n_inline}개")
    onnx.save_model(model, args.out)
    # 메타 사이드카: negpip/lora가 Softmax 앵커 없이 q/k/v/o proj·MHA를 식별하게
    import json
    meta = dict(version=SCRIPT_VERSION, num_heads=num_heads, head_dim=head_dim,
                scale=scale_attr, attentions=meta_attns)
    meta_path = args.out + ".fuse_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[done] {args.out}  (원본 {src_data_base} 공유)")
    print(f"[meta] {meta_path}  (attention {len(meta_attns)}개: "
          f"self {sum(1 for a in meta_attns if a['kind']=='self')} / "
          f"cross {sum(1 for a in meta_attns if a['kind']=='cross')})")
    print("[next] shard_onnx_data.py → add_negpip.py(식별 MHA앵커로 갱신 필요) → add_lora_slots.py")


if __name__ == "__main__":
    main()
