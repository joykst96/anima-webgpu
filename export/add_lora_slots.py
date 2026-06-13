"""
샤딩된 Anima DiT ONNX에 LoRA 슬롯(저랭크 사이드브랜치)을 삽입하는 그래프 수술.

수학: (W + s·(α/r)·up·down)x = Wx + s·B(Ax)
  → q8 base는 건드리지 않고, 각 타깃 프로젝션 y=proj(x) 옆에
     y' = y + lora_scale × MatMul(MatMul(x, A), B) 브랜치를 추가.
  A/B 가중치는 별도 external data 파일 lora.bin 참조 → 페이지가 임의 LoRA를
  같은 레이아웃으로 변환해 주입. 미장착 시 제로 버퍼(브라우저 생성, 다운로드 0).
  lora_scale은 그래프 스칼라 입력 → 슬라이더로 세션 리빌드 없이 조절.

슬롯 구성 (블록당 16모듈 — inspect_lora 조사 결과 기준):
  self_attn  q/k/v/output_proj   (2048→2048) ×4
  cross_attn q/output_proj       (2048→2048) ×2,  k/v (1024→2048) ×2
  mlp        layer1 (2048→8192), layer2 (8192→2048)
  adaln_modulation_{self_attn,cross_attn,mlp}_{1,2}
             _1 (2048→256), _2 (256→6144)

식별 (익명 그래프 — 검증 사실 5. 전부 구조/형상 기반 + 소스 확인된 코드 순서):
  attention 그룹: Softmax 앵커 → QK·PV matmul → q/k/v/o 역추적.
    cross 판정 = k/v 입력이 context 입력에서 도달. v = PV측, q vs k = 노드 순서
    (predict2.py: q=q_proj(x); k=k_proj(ctx); v=v_proj(ctx) — 선언·실행 순서 확인).
  adaln_2 = (R, 3h) 형상 (h=hidden). adaln_1 = (h, R) 중 출력이 adaln_2로 직결.
    (final_layer의 (h,R)은 (R,2h)로 가서 자동 배제.) 브랜치 할당 = 블록 내 순서
    self→cross→mlp (소스 확인).
  mlp = 잔여 시그니처 중 블록 수만큼 등장하는 (h,X)/(X,h) 쌍.
  블록 인덱스 = 각 유형의 그래프 등장 순서 (블록은 순차 트레이스됨).

파이프라인 위치: shard_onnx_data.py **이후** (graph-only 편집 — 가중치 IO 없음).
  external data는 건드리지 않으므로 기존 샤드 그대로 유효. 출력은 .onnx 골격
  + lora.bin(제로) + <out>.lora_manifest.json. 단, .onnx 파일명이 바뀌므로
  manifest.json(샤드 목록)을 복사해 이름 맞춰줄 것 (스크립트가 자동 처리).

사용 (한 줄):
  python export/add_lora_slots.py --src out/dit/anima_dit_turbo32_q8a4ns.onnx --out out/dit/anima_dit_turbo32_q8a4nLs.onnx --rank 48
"""
import argparse
import json
import os
import shutil

import onnx
from onnx import TensorProto, helper

SCRIPT_VERSION = "v1"
WEIGHT_MATMULS = {"MatMulNBits", "MatMul", "Gemm"}

# 블록당 모듈의 정규 순서 (manifest/lora.bin 레이아웃 기준)
MODULE_ORDER = [
    "self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj", "self_attn_output_proj",
    "cross_attn_q_proj", "cross_attn_k_proj", "cross_attn_v_proj", "cross_attn_output_proj",
    "mlp_layer1", "mlp_layer2",
    "adaln_modulation_self_attn_1", "adaln_modulation_self_attn_2",
    "adaln_modulation_cross_attn_1", "adaln_modulation_cross_attn_2",
    "adaln_modulation_mlp_1", "adaln_modulation_mlp_2",
]


# ───────────────────────── 그래프 유틸 ─────────────────────────

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


def back_first_projection(tensor, producer, init_names, limit=512):
    """텐서에서 후방으로 투과 탐색, 처음 만나는 가중치 matmul 노드."""
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
            return None                       # softmax 너머는 다른 영역
        frontier.extend(p.input)
    return None


def fwd_first_weight_matmul(node, consumers, init_names, limit=2048):
    frontier, seen = list(node.output), set()
    while frontier and len(seen) < limit:
        t = frontier.pop()
        for c in consumers.get(t, []):
            if id(c) in seen:
                continue
            seen.add(id(c))
            if is_weight_matmul(c, init_names):
                return c
            if c.op_type in ("Softmax", "MatMul", "Gemm"):
                continue
            frontier.extend(c.output)
    return None


def reachable_fwd(start_tensors, consumers, init_names):
    """프로젝션을 투과하지 않는 전방 도달 텐서 집합."""
    reach, frontier = set(), list(start_tensors)
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


# ───────────────────────── 모듈 식별 ─────────────────────────

def identify_modules(graph, ctx_input="context"):
    init_names = {i.name for i in graph.initializer}
    init_names |= {n.output[0] for n in graph.node if n.op_type == "Constant"}
    init_shapes = {i.name: tuple(i.dims) for i in graph.initializer}
    producer, consumers = build_maps(graph)
    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    ctx_reach = reachable_fwd([ctx_input], consumers, init_names)

    # ── attention 그룹 (Softmax 앵커) ──
    groups = []
    for sm in graph.node:
        if sm.op_type != "Softmax":
            continue
        # PV: softmax 출력을 (투과 경유) 소비하는 활성×활성 MatMul
        pv = None
        frontier, seen = list(sm.output), set()
        while frontier and pv is None:
            t = frontier.pop()
            for c in consumers.get(t, []):
                if id(c) in seen:
                    continue
                seen.add(id(c))
                if c.op_type in ("MatMul", "Gemm") and not is_weight_matmul(c, init_names):
                    pv = (c, t)
                    break
                if not is_weight_matmul(c, init_names):
                    frontier.extend(c.output)
        if pv is None:
            continue
        pv_node, sm_side = pv
        v_side = [i for i in pv_node.input if i != sm_side]
        v_node = back_first_projection(v_side[0], producer, init_names) if v_side else None

        # QK: softmax 입력 후방의 활성×활성 MatMul
        qk = None
        frontier, seen = list(sm.input), set()
        while frontier and qk is None:
            t = frontier.pop()
            p = producer.get(t)
            if p is None or id(p) in seen:
                continue
            seen.add(id(p))
            if p.op_type in ("MatMul", "Gemm") and not is_weight_matmul(p, init_names):
                qk = p
                continue
            if not is_weight_matmul(p, init_names):
                frontier.extend(p.input)
        if qk is None or v_node is None:
            continue
        pa = back_first_projection(qk.input[0], producer, init_names)
        pb = back_first_projection(qk.input[1], producer, init_names)
        if pa is None or pb is None:
            continue
        o_node = fwd_first_weight_matmul(pv_node, consumers, init_names)
        is_cross = v_node.input[0] in ctx_reach
        if is_cross:
            # q = context 미도달 측, k = 도달 측
            q_node = pa if pa.input[0] not in ctx_reach else pb
            k_node = pb if q_node is pa else pa
        else:
            # self: 소스 코드 순서 q → k (predict2.py 확인)
            q_node, k_node = sorted([pa, pb], key=lambda n: node_idx[id(n)])
        groups.append(dict(cross=is_cross, q=q_node, k=k_node, v=v_node, o=o_node,
                           pos=node_idx[id(q_node)]))

    self_g = sorted([g for g in groups if not g["cross"]], key=lambda g: g["pos"])
    cross_g = sorted([g for g in groups if g["cross"]], key=lambda g: g["pos"])
    nb = len(cross_g)
    assert nb > 0 and len(self_g) == nb, \
        f"attention 그룹 불일치: self {len(self_g)} vs cross {nb}"

    hidden = get_kn(self_g[0]["q"], init_shapes)[0]

    # ── 모든 가중치 matmul과 형상 ──
    wnodes = [n for n in graph.node if is_weight_matmul(n, init_names)]
    attn_ids = {id(g[r]) for g in groups for r in ("q", "k", "v", "o") if g[r] is not None}

    # adaln_2: (R, 3h). 단 t_embedder의 adaln_lora 선형도 N=3h (K=h, 1개)이므로
    # K 그룹별로 묶어 "정확히 3*nb개 등장하는 그룹"을 adaln_2로 선별.
    cand3h = [n for n in wnodes if id(n) not in attn_ids
              and get_kn(n, init_shapes)[1] == 3 * hidden]
    byK = {}
    for n in cand3h:
        byK.setdefault(get_kn(n, init_shapes)[0], []).append(n)
    sel = [(k, ns) for k, ns in byK.items() if len(ns) == 3 * nb]
    counts = {k: len(v) for k, v in byK.items()}
    assert len(sel) == 1, f"adaln_2 그룹 모호: K별 개수 {counts} (기대: {3*nb}개짜리 1그룹)"
    R, a2 = sel[0]
    a2_ids = {id(n) for n in a2}
    # adaln_1: (h, R) 중 출력이 adaln_2로 직결 (final_layer의 (h,R)은 배제됨)
    a1 = [n for n in wnodes if id(n) not in attn_ids
          and get_kn(n, init_shapes) == (hidden, R)
          and id(fwd_first_weight_matmul(n, consumers, init_names) or 0) in a2_ids]
    assert len(a1) == 3 * nb and len(a2) == 3 * nb, \
        f"adaln 수 불일치: _1 {len(a1)}, _2 {len(a2)} (기대 {3*nb})"
    a1.sort(key=lambda n: node_idx[id(n)])
    a2.sort(key=lambda n: node_idx[id(n)])

    # mlp: 잔여 중 (h, X)/(X, h)가 각각 nb개인 쌍
    used = attn_ids | a2_ids | {id(n) for n in a1}
    from collections import Counter
    rest = [n for n in wnodes if id(n) not in used]
    sigc = Counter(get_kn(n, init_shapes) for n in rest)
    m1sig = [(k, v) for (k, v) in sigc.items()
             if v == nb and k[0] == hidden and sigc.get((k[1], hidden), 0) == nb]
    assert len(m1sig) == 1, f"mlp 시그니처 모호: {m1sig}"
    (mh,) = [k[1] for k, _ in m1sig]
    mlp1 = sorted([n for n in rest if get_kn(n, init_shapes) == (hidden, mh)],
                  key=lambda n: node_idx[id(n)])
    mlp2 = sorted([n for n in rest if get_kn(n, init_shapes) == (mh, hidden)],
                  key=lambda n: node_idx[id(n)])

    # ── (블록, 모듈) → 노드 매핑 ──
    mapping = {}
    for b in range(nb):
        sg, cg = self_g[b], cross_g[b]
        mapping[(b, "self_attn_q_proj")] = sg["q"]
        mapping[(b, "self_attn_k_proj")] = sg["k"]
        mapping[(b, "self_attn_v_proj")] = sg["v"]
        mapping[(b, "self_attn_output_proj")] = sg["o"]
        mapping[(b, "cross_attn_q_proj")] = cg["q"]
        mapping[(b, "cross_attn_k_proj")] = cg["k"]
        mapping[(b, "cross_attn_v_proj")] = cg["v"]
        mapping[(b, "cross_attn_output_proj")] = cg["o"]
        mapping[(b, "mlp_layer1")] = mlp1[b]
        mapping[(b, "mlp_layer2")] = mlp2[b]
        for j, br in enumerate(("self_attn", "cross_attn", "mlp")):
            mapping[(b, f"adaln_modulation_{br}_1")] = a1[b * 3 + j]
            mapping[(b, f"adaln_modulation_{br}_2")] = a2[b * 3 + j]

    # 중복 배정 방지 검증
    assert len({id(n) for n in mapping.values()}) == len(mapping), "노드 중복 배정!"
    print(f"[identify] 블록 {nb} · hidden {hidden} · adaln R {R} · mlp hidden {mh}"
          f" · 슬롯 대상 {len(mapping)}개")
    return mapping, nb, init_shapes


# ───────────────────────── 슬롯 삽입 ─────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="샤딩 완료된 DiT .onnx")
    p.add_argument("--out", required=True, help="출력 .onnx (새 파일명)")
    p.add_argument("--rank", type=int, default=48)
    p.add_argument("--lora-bin", default=None,
                   help="lora.bin 파일명 (기본: <out 베이스>.lora.bin)")
    p.add_argument("--dump", action="store_true", help="식별 결과 전체 출력")
    args = p.parse_args()

    lora_bin = args.lora_bin or (os.path.splitext(os.path.basename(args.out))[0] + ".lora.bin")
    print(f"[add_lora_slots {SCRIPT_VERSION}] rank={args.rank}, bin={lora_bin}")

    # graph-only 편집: external data를 메모리에 올리지 않음
    model = onnx.load(args.src, load_external_data=False)
    graph = model.graph
    mapping, nb, init_shapes = identify_modules(graph)

    producer, consumers = build_maps(graph)
    init_names = {i.name for i in graph.initializer}
    r = args.rank
    F32 = TensorProto.FLOAT

    # lora_scale 입력 (스칼라, fp32)
    graph.input.append(helper.make_tensor_value_info("lora_scale", F32, [1]))

    manifest = {"rank": r, "dtype": "float32", "bin": lora_bin,
                "blocks": nb, "entries": []}
    offset = 0
    new_nodes_at = []                          # (삽입 위치 노드, [새 노드들])

    for b in range(nb):
        for mod in MODULE_ORDER:
            node = mapping[(b, mod)]
            if node.op_type == "MatMulNBits":
                a = {x.name: x for x in node.attribute}
                K, N = a["K"].i, a["N"].i
            else:
                K, N = init_shapes[node.input[1]]
            key = f"b{b}_{mod}"
            tA, tB = f"lora_{key}_A", f"lora_{key}_B"
            for tname, dims in ((tA, [K, r]), (tB, [r, N])):
                t = TensorProto()
                t.name = tname
                t.data_type = F32
                t.dims.extend(dims)
                nbytes = dims[0] * dims[1] * 4
                t.data_location = TensorProto.EXTERNAL
                for kk, vv in (("location", lora_bin),
                               ("offset", str(offset)), ("length", str(nbytes))):
                    e = t.external_data.add()
                    e.key, e.value = kk, vv
                graph.initializer.append(t)
                offset_entry = offset
                offset += nbytes
                if tname == tA:
                    entA = {"name": tname, "offset": offset_entry, "K": K, "r": r}
                else:
                    entB = {"name": tname, "offset": offset_entry, "r": r, "N": N}
            manifest["entries"].append({"key": key, "K": K, "N": N,
                                        "A": entA, "B": entB})

            x_in = node.input[0]
            y_out = node.output[0]
            n_a = helper.make_node("MatMul", [x_in, tA], [f"lora_{key}_xa"],
                                   name=f"lora_{key}_ma")
            n_b = helper.make_node("MatMul", [f"lora_{key}_xa", tB],
                                   [f"lora_{key}_xb"], name=f"lora_{key}_mb")
            n_s = helper.make_node("Mul", [f"lora_{key}_xb", "lora_scale"],
                                   [f"lora_{key}_xs"], name=f"lora_{key}_ms")
            y_new = f"{y_out}_lora"
            n_add = helper.make_node("Add", [y_out, f"lora_{key}_xs"], [y_new],
                                     name=f"lora_{key}_add")
            # 기존 소비자 재배선 (그래프 출력 포함)
            for c in consumers.get(y_out, []):
                for ii, inp in enumerate(c.input):
                    if inp == y_out:
                        c.input[ii] = y_new
            for go in graph.output:
                if go.name == y_out:
                    go.name = y_new
            new_nodes_at.append((node, [n_a, n_b, n_s, n_add]))
            if args.dump:
                print(f"  b{b:02d} {mod:34s} <- {node.name} ({K}x{N})")

    # 위상 순서: 각 타깃 노드 바로 뒤에 삽입 (역순으로 처리해 인덱스 안정화)
    idx_of = {id(n): i for i, n in enumerate(graph.node)}
    for node, nns in sorted(new_nodes_at, key=lambda t: -idx_of[id(t[0])]):
        at = idx_of[id(node)] + 1
        for j, nn in enumerate(nns):
            graph.node.insert(at + j, nn)

    total = offset
    print(f"[insert] 슬롯 {len(manifest['entries'])}개 · lora.bin {total/2**20:.1f} MiB (fp32)")

    onnx.save_model(model, args.out)

    out_dir = os.path.dirname(args.out) or "."
    with open(os.path.join(out_dir, lora_bin), "wb") as f:
        f.truncate(total)                       # 제로 파일 (sparse)
    mpath = args.out + ".lora_manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f)
    # 샤드 목록 manifest 복사 (파일명 변경 대응)
    src_man = args.src + ".manifest.json"
    if os.path.exists(src_man):
        shutil.copy(src_man, args.out + ".manifest.json")
        print(f"[copy] 샤드 manifest → {os.path.basename(args.out)}.manifest.json")
    print(f"[save] {args.out}\n[save] {lora_bin} (zeros)\n[save] {os.path.basename(mpath)}")
    print("[done] 페이지의 DiT 경로를 새 파일명으로 변경. lora.bin은 페이지가 주입.")


if __name__ == "__main__":
    main()
