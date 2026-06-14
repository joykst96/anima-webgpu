"""
VAE mid-block attention의 score 단일버퍼를 Q 토큰축 N분할로 제한 (chunked SDPA).

배경: VAE attention score (1,1,S,S)가 1280²(S=25600)서 2.44GB > 2GB → OOM.
MHA fuse는 head_dim 384가 WebGPU FA2 workgroup 한계(32KB) 초과로 컴파일 실패.
→ FA 원리를 그래프 수준에서 수동 구현: Q를 토큰축 N등분, 각 chunk만 score 생성.

수치 동치: softmax가 행(Q토큰) 단위라 Q 행을 쪼개도 결과 불변. score 단일버퍼 1/N.
N=8: 1536²(S=36864)서 chunk score = (S/8)*S*4 = 0.68GB < 2GB. K^T·V는 공유.

원본 SDPA 구조 (inspect_vae_attn.py):
  QK(Q=val_113, K^T=val_116) → val_117 → Softmax → val_118 → PV(val_118, V=transpose_2) → sdpa
  Q,K는 이미 scale 적용됨(val_113/val_116). V=transpose_2.
교체: 위 QK→Softmax→PV 3노드를, Q를 N등분해 chunk별 (QK_i→softmax_i→PV_i) 후 Concat.
  K^T(val_116), V(transpose_2)는 그대로 공유 입력.

균등분할 안 되는 S는 Split을 런타임 shape 기반 균등+나머지로 처리(SplitToSequence 대신
정적 N개 Split + 동적 sizes). 단 1280²~1536²만 사용 예정이라 대부분 N=8로 나눠떨어짐.

graph-only(.data 공유, location 자기이름으로 통일 + 원본 .data 복사).

사용 (한 줄):
  python_embeded\python.exe export\chunk_vae_attention.py --src out\vae\qwen_image_vae_decoder_dyn32_2d.onnx --out out\vae\qwen_image_vae_decoder_dyn32_2dc.onnx --chunks 8
"""
import argparse
import os
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper


SCRIPT_VERSION = "v1"


def build_maps(graph):
    producer, consumers = {}, {}
    for n in graph.node:
        for o in n.output:
            producer[o] = n
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    return producer, consumers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chunks", type=int, default=8)
    args = p.parse_args()
    N = args.chunks

    print(f"[chunk_vae_attention {SCRIPT_VERSION}] N={N}")
    model = onnx.load(args.src, load_external_data=False)
    graph = model.graph
    producer, consumers = build_maps(graph)

    sm = [n for n in graph.node if n.op_type == "Softmax"]
    assert len(sm) == 1, f"VAE Softmax가 1개가 아님: {len(sm)}"
    sm = sm[0]
    qk = producer.get(sm.input[0])
    assert qk is not None and qk.op_type == "MatMul"
    pv = next((c for c in consumers.get(sm.output[0], []) if c.op_type == "MatMul"), None)
    assert pv is not None

    Q = qk.input[0]                                   # val_113 (scale 적용 Q, (1,1,S,384))
    Kt = qk.input[1]                                  # val_116 (scale 적용 K^T, (1,1,384,S))
    V = [i for i in pv.input if i != sm.output[0]][0]  # transpose_2 (V, (1,1,S,384))
    sdpa_out = pv.output[0]                           # scaled_dot_product_attention
    sm_axis = next((a.i for a in sm.attribute if a.name == "axis"), -1)
    print(f"[trace] Q={Q} K^T={Kt} V={V} out={sdpa_out} softmax_axis={sm_axis}")

    new_nodes = []

    # ── Q를 토큰축(axis=2, (1,1,S,384)의 S)으로 N등분 ──
    # 균등분할: Split은 기본 균등. S가 N으로 안 나눠떨어지면 ONNX Split(opset13+)은
    # split 입력 없으면 균등 시도하다 실패 → 동적 sizes 계산이 안전.
    # 여기선 Shape→Gather(dim2)→나눗셈으로 chunk 크기 산출, 마지막에 나머지.
    # 단순화: split 속성 없이 num_outputs로 균등분할(opset18 Split의 num_outputs).
    #   opset18 Split: num_outputs 지정 시 S//N씩 + 마지막이 나머지 흡수. ← 이게 정확히 필요.
    q_chunks = [f"qchunk_{i}" for i in range(N)]
    new_nodes.append(helper.make_node(
        "Split", [Q], q_chunks, name="vae_q_split", axis=2, num_outputs=N))

    # ── 각 chunk: QK_i (q_i @ K^T) → softmax → PV_i (prob_i @ V) ──
    out_chunks = []
    for i in range(N):
        sc = f"vae_score_{i}"
        pr = f"vae_prob_{i}"
        oc = f"vae_out_{i}"
        new_nodes.append(helper.make_node("MatMul", [q_chunks[i], Kt], [sc],
                                          name=f"vae_qk_{i}"))
        new_nodes.append(helper.make_node("Softmax", [sc], [pr],
                                          name=f"vae_softmax_{i}", axis=sm_axis))
        new_nodes.append(helper.make_node("MatMul", [pr, V], [oc],
                                          name=f"vae_pv_{i}"))
        out_chunks.append(oc)

    # ── Concat 출력 chunks (토큰축 axis=2) → sdpa_out ──
    new_nodes.append(helper.make_node(
        "Concat", out_chunks, [sdpa_out], name="vae_out_concat", axis=2))

    # ── 기존 QK/Softmax/PV 제거 ──
    dead = {id(qk), id(sm), id(pv)}
    keep = [n for n in graph.node if id(n) not in dead]
    removed = len(graph.node) - len(keep)
    del graph.node[:]
    graph.node.extend(keep)
    graph.node.extend(new_nodes)
    sort_topological(graph)
    print(f"[chunk] QK/SM/PV 3노드 → {len(new_nodes)}노드 "
          f"(Split1 + chunk{N}*(QK,SM,PV)=3N + Concat1). 제거 {removed}, 추가 {len(new_nodes)}")

    # external data location 통일 + 원본 .data 복사
    new_base = os.path.basename(args.out) + ".data"
    n_ext = 0
    for init in graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            for kv in init.external_data:
                if kv.key == "location":
                    kv.value = new_base
            n_ext += 1
    onnx.save(model, args.out)
    src_data, dst_data = args.src + ".data", args.out + ".data"
    if os.path.exists(src_data) and os.path.abspath(src_data) != os.path.abspath(dst_data):
        shutil.copyfile(src_data, dst_data)
        print(f"[data] external {n_ext}개 → {new_base}, 원본 .data 복사 완료")
    print(f"[done] {args.out}")
    print("[next] 브라우저서 1280²/1536² 디코드 → score 1/N로 OOM 해소 실측")


def sort_topological(graph):
    produced = {i.name for i in graph.initializer} | {i.name for i in graph.input}
    pending, ordered = list(graph.node), []
    while pending:
        still, progress = [], False
        for n in pending:
            if all((inp in produced or inp == "") for inp in n.input):
                ordered.append(n)
                produced.update(n.output)
                progress = True
            else:
                still.append(n)
        pending = still
        if not progress:
            ordered.extend(pending)
            break
    del graph.node[:]
    graph.node.extend(ordered)


if __name__ == "__main__":
    main()
