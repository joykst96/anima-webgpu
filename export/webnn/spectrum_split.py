# Spectrum용 그래프 분할 (그래프만, 가중치 샤드는 그대로): WebNN용 DiT 그래프에서
#   본체  <src>_sp.onnx        : 원래 그래프 + 출력 spectrum_feat (final_layer에 들어가는 특징, fp32 [1,1,H/2,W/2,D]).
#                              샤드 목록(.manifest.json)·슬롯 목록(.lora_manifest.json)은 <src>의 것을 그대로 쓴다
#   머리  <head>.onnx(+.data)  : spectrum_feat, timesteps → v_pred  (t_embedder → final_layer → unpatchify, 가중치 약 35MB)
#         <head>k.onnx         : feat0..15(fp16), coef[16], timesteps → v_pred — 특징의 가중합까지 그래프 안에서(.data는 머리 것 공유)
# 를 만든다.
#   python -I export/webnn/spectrum_split.py <src.onnx> [--head <head.onnx>]
#   --head를 주면 머리도 쓴다(머리는 LoRA 슬롯과 무관하므로 비슬롯 그래프에서 한 번만 만든다).
#   동적 그래프(export_dit.py --dynamic)도 받는다: 특징의 H/2·W/2는 이름 있는 차원(feat_h, feat_w)이 되고, 페이지가 세션을 만들 때
#   freeDimensionOverrides로 고정한다. 머리 안에서 latent의 H·W를 읽던 Shape(latent)는 2 × 특징 H·W로 바꿔 latent 입력을 없앤다.
#
# Spectrum(Han et al., arXiv:2603.01623)은 건너뛰는 스텝에서 블록 전체를 생략하고, 저장해 둔 이 특징들로 그 스텝의 특징을 예측해
# 머리만 돌린다(web/js/spectrum.js). 머리는 그 스텝의 timestep으로 돌려야 한다 — 같은 특징이라도 σ가 0.5 → 0.1이면 출력이 10% 달라진다.
#
# 분할 지점: final_layer의 LayerNorm 입력. webnn_fix.py가 그래프에 하나뿐인 fp16 LayerNorm(= final_layer)을 Cast(fp32) → LN → Cast(fp16)로
# 감싸 두었으므로, 그 Cast(fp32)의 출력을 특징으로 쓴다. fp32라 예측한 특징(가중합)이 fp16 한계(65,504)를 넘을 걱정이 없다
# (특징 |x| 최대 약 4만 실측).
# 검증(2026-11-13, 네이티브 DirectML, Turbo 1024²): 본체의 v_pred는 원본과 비트 동일, 머리(본체가 낸 특징, σ)의 출력도 본체 v_pred와
# 비트 동일. 본체 388ms(추가 출력 비용 없음), 머리 2ms.
import os
import sys

import onnx
from onnx import TensorProto, helper, shape_inference, utils
from onnx.external_data_helper import load_external_data_for_model

FEAT = "spectrum_feat"
FEAT16 = "spectrum_feat16"   # 같은 특징의 fp16판 — GPU에 둔 채 다음 그래프로 넘길 때 쓴다(메모리 절반)
SLOTS = 16                   # 가중합 머리가 받는 특징 수 = 예측에 쓰는 이력 길이 상한 (web/js/spectrum.js SPECTRUM.slots와 같아야 한다)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
src = args[0]
head_out = sys.argv[sys.argv.index("--head") + 1] if "--head" in sys.argv else None
if head_out in args:
    args.remove(head_out)

m = onnx.load(src, load_external_data=False)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
wrapped = [n for n in g.node if n.op_type == "LayerNormalization"
           and n.input[0] in prod and prod[n.input[0]].op_type == "Cast" and prod[n.input[0]].name.endswith("_wf_c0")]
if len(wrapped) != 1:
    sys.exit(f"webnn_fix가 감싼 LayerNorm이 1개가 아님({len(wrapped)}) — webnn_fix.py를 거친 fp16 DiT 그래프여야 한다")
feat32 = wrapped[0].input[0]                      # Cast(fp32)의 출력 = LayerNorm 입력
out_name = g.output[0].name
inf = shape_inference.infer_shapes(m)
vi = {v.name: v for v in list(inf.graph.value_info) + list(inf.graph.input)}
fshape = [d.dim_value if d.HasField("dim_value") else ("feat_h", "feat_w")[k - 2] for k, d in enumerate(vi[prod[feat32].input[0]].type.tensor_type.shape.dim)]  # 동적판: H/2·W/2를 이름 있는 차원으로

# ── 머리: (특징, timesteps) → v_pred ──
if head_out:
    # Extractor는 입력 이름을 value_info에서 찾는다 — webnn_fix가 만든 Cast 출력은 거기 없으므로 사본에 적어 넣고 자른다
    m_head = onnx.ModelProto()
    m_head.CopyFrom(m)
    m_head.graph.value_info.append(helper.make_tensor_value_info(feat32, TensorProto.FLOAT, fshape))
    head = utils.Extractor(m_head).extract_model([feat32, "timesteps", "latent"], [out_name])
    names = {i.name for i in head.graph.input} | {t.name for t in head.graph.initializer} | {o for n in head.graph.node for o in n.output}
    dangling = sorted({i for n in head.graph.node for i in n.input if i} - names)
    if dangling:
        sys.exit(f"머리가 특징·timesteps 밖의 값에 의존한다: {dangling[:5]}")
    for n in head.graph.node:                      # 입력 이름을 고정 이름으로
        for k, i in enumerate(n.input):
            if i == feat32:
                n.input[k] = FEAT
    for i in head.graph.input:
        if i.name == feat32:
            i.CopyFrom(helper.make_tensor_value_info(FEAT, TensorProto.FLOAT, fshape))
    # 동적판: 머리 안의 Shape(latent, start=3|4)(= latent H·W, unpatchify reshape·출력 자르기에 쓰임)를 2 × 특징 H·W로 바꾸고 latent 입력을 뺀다.
    # 짝수 latent 전용(64px 배수 해상도는 모두 해당) — 홀수면 원래 그래프는 패딩 후 자르지만 여기선 2·ceil(H/2)가 된다.
    from onnx import numpy_helper
    import numpy as np
    head.graph.initializer.append(numpy_helper.from_array(np.array([2], np.int64), "sp_two"))
    new, n_sh = [], 0
    for n in head.graph.node:
        if n.op_type == "Shape" and n.input[0] == "latent":
            st = {x.name: x.i for x in n.attribute}["start"]          # 3 → H, 4 → W
            new.append(helper.make_node("Shape", [FEAT], [n.output[0] + "_fh"], start=st - 1, end=st, name=n.name + "_feat"))
            new.append(helper.make_node("Mul", [n.output[0] + "_fh", "sp_two"], [n.output[0]], name=n.name + "_x2"))
            n_sh += 1
        else:
            new.append(n)
    del head.graph.node[:]
    head.graph.node.extend(new)
    keep = [i for i in head.graph.input if i.name != "latent"]
    del head.graph.input[:]
    head.graph.input.extend(keep)
    left = [i for n in head.graph.node for i in n.input if i == "latent"]
    if left:
        sys.exit(f"머리에 latent 사용이 남음: {len(left)}")
    if n_sh:
        print(f"[spectrum_split] 동적 머리: Shape(latent) {n_sh}개 -> 2 × Shape({FEAT})")
    load_external_data_for_model(head, os.path.dirname(os.path.abspath(src)))   # 머리가 쓰는 초기값만 샤드에서 읽는다
    nbytes = sum(len(t.raw_data) for t in head.graph.initializer)
    data = os.path.basename(head_out) + ".data"
    dpath = os.path.join(os.path.dirname(os.path.abspath(head_out)), data)
    if os.path.exists(dpath):
        os.remove(dpath)
    onnx.save(head, head_out, save_as_external_data=True, all_tensors_to_one_file=True, location=data, size_threshold=1024)
    print(f"[spectrum_split] head: {len(head.graph.node)} nodes, {nbytes / 2**20:.1f} MiB -> {head_out}")

    # ── 가중합 머리: feat0..feat{K-1}(fp16), coef[K], timesteps → v_pred.  특징 = Σ coef_i · feat_i (fp32로 합산) 뒤는 머리와 같다.
    #    특징을 GPU 텐서로 둔 채 예측할 때 쓴다(본체의 spectrum_feat16 출력을 그대로 입력으로). 가중치는 머리의 .data를 같이 쓴다.
    hk = onnx.load(head_out, load_external_data=False)
    pre = [helper.make_node("Split", ["coef"], [f"sp_c{i}" for i in range(SLOTS)], axis=0, num_outputs=SLOTS, name="sp_split")]
    acc = None
    for i in range(SLOTS):
        pre.append(helper.make_node("Cast", [f"feat{i}"], [f"sp_f{i}"], to=TensorProto.FLOAT, name=f"sp_cast{i}"))
        pre.append(helper.make_node("Mul", [f"sp_f{i}", f"sp_c{i}"], [f"sp_m{i}"], name=f"sp_mul{i}"))
        if acc is None:
            acc = f"sp_m{i}"
        else:
            out = FEAT if i == SLOTS - 1 else f"sp_a{i}"
            pre.append(helper.make_node("Add", [acc, f"sp_m{i}"], [out], name=f"sp_add{i}"))
            acc = out
    nodes = pre + list(hk.graph.node)
    del hk.graph.node[:]
    hk.graph.node.extend(nodes)
    keep = [i for i in hk.graph.input if i.name != FEAT]
    del hk.graph.input[:]
    hk.graph.input.extend([helper.make_tensor_value_info(f"feat{i}", TensorProto.FLOAT16, fshape) for i in range(SLOTS)]
                          + [helper.make_tensor_value_info("coef", TensorProto.FLOAT, [SLOTS])] + keep)
    hk_out = head_out[:-len(".onnx")] + "k.onnx"
    onnx.save(hk, hk_out)
    print(f"[spectrum_split] head with weighted sum ({SLOTS} slots) -> {hk_out}")

# ── 본체: 원래 그래프 + 특징 출력 ──
g.node.append(helper.make_node("Identity", [feat32], [FEAT], name="spectrum_feat_out"))
g.output.append(helper.make_tensor_value_info(FEAT, TensorProto.FLOAT, fshape))
# fp16판은 fp32 특징을 새 Cast로 내려 만든다. 그래프 안의 원래 fp16 텐서(Cast fp32→fp16의 출력)를 그대로 출력으로 내면 안 된다 —
# ORT Web이 그 Cast와 webnn_fix의 Cast(fp16→fp32)를 한 쌍으로 정리하던 것이 막혀 v_pred가 0.08% 달라진다(2026-11-13 실측).
g.node.append(helper.make_node("Cast", [feat32], [FEAT16], to=TensorProto.FLOAT16, name="spectrum_feat16_out"))
g.output.append(helper.make_tensor_value_info(FEAT16, TensorProto.FLOAT16, fshape))
dst = src[:-len(".onnx")] + "_sp.onnx"
onnx.save(m, dst)
print(f"[spectrum_split] main: + outputs {FEAT} (fp32), {FEAT16} (fp16) {fshape} -> {dst}")
