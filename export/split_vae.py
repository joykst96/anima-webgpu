"""
VAE 디코더 ONNX → head / tail 두 그래프로 분할 (재export 불필요, 그래프 후처리).

왜: ORT WebGPU는 한 번의 run() 안에서 해제된 버퍼를 재사용하지 않아, VAE 정점 메모리가
중간 텐서 합만큼 커진다(1024²에서 +7~8GB, 1280² 이상은 VRAM 상한 → 페이징 5~17s).
- head: latent → mid-block attention + residual 출력 (latent 해상도, 전역 연산 포함)
- tail: 나머지 업샘플·conv (국소 연산만, 수용영역 반경 117.5px = 14.7 latent 실측)
페이지는 head를 한 번 돌리고 tail을 가로 띠로 나눠(halo 16 latent) 돌린다 — 단일 호출과 비트 동일
(1280²·1536²에서 실측, web/js/vae.js).

경계: 마지막 Softmax(어텐션) 뒤 첫 Add = 어텐션 residual 합. tail은 latent를 Shape로만 쓴다
(업샘플 GatherND 인덱스 계산) → 페이지는 띠 크기의 0 텐서를 넘긴다.
가중치는 원본 .data를 그대로 참조한다(head·tail 모두 같은 파일, 같은 오프셋).

사용:
  python export/split_vae.py --src out/vae/qwen_image_vae_decoder_dyn32_2dc.onnx
  → out/vae/qwen_image_vae_decoder_dyn32_2dc_head.onnx, ..._tail.onnx (그래프만, 수백 KB)
"""
import argparse
import os

import onnx
from onnx import TensorProto, helper, utils


def find_boundary(model):
    nodes = list(model.graph.node)
    last_sm = max(i for i, n in enumerate(nodes) if n.op_type == "Softmax")
    for n in nodes[last_sm + 1:]:
        if n.op_type == "Add":
            return n.output[0]
    raise SystemExit("경계 Add를 찾지 못함")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--boundary", default=None, help="경계 텐서 이름 (기본: 자동 탐지)")
    args = p.parse_args()

    m = onnx.load(args.src, load_external_data=False)
    inp = m.graph.input[0].name
    out = m.graph.output[0].name
    b = args.boundary or find_boundary(m)
    print(f"[split] 입력 {inp}, 경계 {b}, 출력 {out}")

    e = utils.Extractor(m)
    head = e.extract_model([inp], [b])
    tail = e.extract_model([b, inp], [out])

    # tail이 latent를 Shape로만 쓰는지 확인 (값을 쓰면 띠 분할 시 0 텐서를 넘길 수 없다)
    users = {n.op_type for n in tail.graph.node if inp in n.input}
    assert users <= {"Shape"}, f"tail이 {inp} 값을 사용함: {users}"
    # 전역 연산이 tail에 없어야 띠 분할이 단일 호출과 동치
    glob = {n.op_type for n in tail.graph.node} & {"Softmax", "GlobalAveragePool", "ReduceMean"}
    assert not glob, f"tail에 전역 연산: {glob}"

    stem = os.path.splitext(args.src)[0]
    outs = [("head", head), ("tail", tail)]

    # head의 mid-block 어텐션도 쿼리 청크별 run으로 나눌 수 있게 3조각을 추가로 만든다 (1536² 이상에서 사용).
    # 한 run 안에선 8청크의 score·prob(1536²: 청크당 680MB×2)이 전부 살아 있어 정점이 커지기 때문.
    #   head_a: latent → (scaled q, scaled kᵀ, v, residual)   attn: (q 청크, kᵀ, v) → 출력 청크   head_b: (청크 합친 출력, residual) → 경계
    split = next(n for n in m.graph.node if n.op_type == "Split" and n.output[0].startswith("qchunk"))
    users = {n.output[0]: n for n in m.graph.node}
    cons = {}
    for n in m.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    mm0 = next(n for n in cons[split.output[0]] if n.op_type == "MatMul")      # q·kᵀ
    sm0 = next(n for n in cons[mm0.output[0]] if n.op_type == "Softmax")
    pv0 = next(n for n in cons[sm0.output[0]] if n.op_type == "MatMul")        # p·v
    cat = next(n for n in cons[pv0.output[0]] if n.op_type == "Concat")
    q_src, kT, v = split.input[0], mm0.input[1], pv0.input[1]
    nh = cat.output[0]
    resid = next(i for i in users[b].input if i in users and users[i].op_type != "Transpose")  # 경계 Add의 residual 쪽
    head_a = e.extract_model([inp], [q_src, kT, v, resid])
    head_b = e.extract_model([nh, resid, inp], [b])
    assert {n.op_type for n in head_b.graph.node if inp in n.input} <= {"Shape"}, "head_b가 latent 값을 사용"
    sm_axis = next(a.i for a in sm0.attribute if a.name == "axis")
    F = TensorProto.FLOAT
    attn = helper.make_model(
        helper.make_graph(
            [helper.make_node("MatMul", ["q", "kT"], ["score"]),
             helper.make_node("Softmax", ["score"], ["prob"], axis=sm_axis),
             helper.make_node("MatMul", ["prob", "v"], ["out"])],
            "vae_attn_chunk",
            [helper.make_tensor_value_info("q", F, [1, 1, "n", "c"]),
             helper.make_tensor_value_info("kT", F, [1, 1, "c", "N"]),
             helper.make_tensor_value_info("v", F, [1, 1, "N", "c"])],
            [helper.make_tensor_value_info("out", F, [1, 1, "n", "c"])]),
        opset_imports=m.opset_import, ir_version=m.ir_version)
    onnx.checker.check_model(attn)
    print(f"[split] 어텐션 분할: q={q_src} kT={kT} v={v} residual={resid} concat={nh}")
    outs += [("head_a", head_a), ("attn", attn), ("head_b", head_b)]

    for name, mm in outs:
        dst = f"{stem}_{name}.onnx"
        onnx.save(mm, dst)
        print(f"[split] {name}: {len(mm.graph.node)} 노드 → {dst}")


if __name__ == "__main__":
    main()
