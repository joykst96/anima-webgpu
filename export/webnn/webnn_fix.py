# WebNN(Chrome 157 = ONNX Runtime DirectML EP)이 받는 형태로 정적 DiT 그래프를 고친다. 가중치는 건드리지 않는다.
#   python -I export/webnn/webnn_fix.py <src.onnx> <dst.onnx>
#
# 고치는 것:
#   1) 랭크 5 이상 입력의 MatMul(A[..., K] × W[K, N]) — DML EP가 세션 생성에서 E_INVALIDARG(AbiCustomRegistry.cpp(519))로 거부한다.
#      Cosmos 블록은 x를 [B,T,H,W,D] 5차원으로 들고 다녀 선형층 대부분이 여기 걸린다.
#      → Reshape([1, -1, K]) → MatMul → Reshape(원래 앞 차원 + [N]). 수치는 같다.
#   2)·3)은 아래 코드의 주석 참고(allowzero Reshape, fp16 LayerNormalization). 동적 그래프(--dynamic)도 받는다(1').
# 필요: 정적 shape 그래프(shape inference로 모든 MatMul 입력 shape이 정해져야 함).
#   입력 그래프 = export/export_dit.py --size 1024 (--dynamic·--fp32-act 없이, fp16 로드). 이후 export/shard_onnx_data.py로 샤딩.
#   세 단계를 묶은 진입점: export/webnn/build_webnn.bat
#
# 결과 (2026-11-13, RTX 5070 Ti, Chrome 157 플래그, Turbo 28블록 1024², 같은 입력, 기준 = PyTorch fp32):
#   현재 운영 그래프(WebGPU, DP4A+FA)  1,431ms/스텝, 오차 2.44/1.74/1.55% (σ 1.0/0.5/0.1)
#   이 스크립트로 고친 fp16 그래프(WebNN)  383ms/스텝, 오차 1.09/0.48/0.50%, 4,783노드 중 4,782개 WebNN 배정
#   고치지 않으면 브라우저에서 세션 생성이 오류 없이 영원히 멈춘다(ORT Web 1.26이 build() 거부를 전달하지 않음).
import sys, os, onnx, numpy as np
from onnx import helper, numpy_helper, shape_inference

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
inf = shape_inference.infer_shapes(m)
shapes = {}
for v in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output):
    shapes[v.name] = [d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim]
dtypes = {v.name: v.type.tensor_type.elem_type for v in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output)}
for t in m.graph.initializer:
    shapes[t.name] = list(t.dims)
    dtypes[t.name] = t.data_type

g = m.graph
new_nodes, new_inits, n_fix, n_rs, n_ln = [], [], 0, 0, 0
for n in g.node:
    a = shapes.get(n.input[0]) if n.op_type == "MatMul" else None
    b = shapes.get(n.input[1]) if n.op_type == "MatMul" else None
    if a and b and len(a) >= 5 and len(b) == 2 and None not in a:
        k, nn = a[-1], b[1]
        s_in, s_out = f"{n.name}_wf_s3", f"{n.name}_wf_sN"
        new_inits.append(numpy_helper.from_array(np.array([1, -1, k], np.int64), s_in))
        new_inits.append(numpy_helper.from_array(np.array(a[:-1] + [nn], np.int64), s_out))
        new_nodes.append(helper.make_node("Reshape", [n.input[0], s_in], [n.name + "_wf_a3"], name=n.name + "_wf_r1"))
        new_nodes.append(helper.make_node("MatMul", [n.name + "_wf_a3", n.input[1]], [n.name + "_wf_y3"], name=n.name))
        new_nodes.append(helper.make_node("Reshape", [n.name + "_wf_y3", s_out], [n.output[0]], name=n.name + "_wf_r2"))
        n_fix += 1
    elif a and b and len(a) >= 5 and len(b) == 2:
        # 1') 동적 그래프(H/W가 dim_param): 앞 차원을 Shape로 읽어 되돌린다. 브라우저에서 freeDimensionOverrides로 차원을 고정하면
        #     ORT가 이 Shape/Slice/Concat을 상수로 접는다(2026-11-13 실측: 접힌 뒤 전 노드 WebNN 배정).
        k, nn = b[0], b[1]
        p = n.name + "_wf_"
        new_inits += [numpy_helper.from_array(np.array([1, -1, k], np.int64), p + "s3"), numpy_helper.from_array(np.array([nn], np.int64), p + "n"),
                      numpy_helper.from_array(np.array([0], np.int64), p + "z"), numpy_helper.from_array(np.array([-1], np.int64), p + "m1")]
        new_nodes += [helper.make_node("Shape", [n.input[0]], [p + "shape"], name=p + "shape_n"),
                      helper.make_node("Slice", [p + "shape", p + "z", p + "m1", p + "z"], [p + "lead"], name=p + "slice_n"),
                      helper.make_node("Concat", [p + "lead", p + "n"], [p + "sN"], axis=0, name=p + "concat_n"),
                      helper.make_node("Reshape", [n.input[0], p + "s3"], [p + "a3"], name=p + "r1"),
                      helper.make_node("MatMul", [p + "a3", n.input[1]], [p + "y3"], name=n.name),
                      helper.make_node("Reshape", [p + "y3", p + "sN"], [n.output[0]], name=p + "r2")]
        n_fix += 1
    elif n.op_type == "Reshape" and any(x.name == "allowzero" and x.i == 1 for x in n.attribute) \
            and shapes.get(n.output[0]) and None not in shapes[n.output[0]]:
        # 2) allowzero=1 이면서 shape에 -1이 있는 Reshape — 네이티브 DML EP가 거부한다(브라우저에서는 ORT Web의 WebNN EP가
        #    정적 shape으로 바꿔 넘기므로 무관). 정적 그래프라 추론된 출력 shape을 그대로 적고 allowzero를 뺀다.
        s_new = f"{n.name}_wf_shape"
        new_inits.append(numpy_helper.from_array(np.array(shapes[n.output[0]], np.int64), s_new))
        new_nodes.append(helper.make_node("Reshape", [n.input[0], s_new], list(n.output), name=n.name))
        n_rs += 1
    elif n.op_type == "LayerNormalization" and dtypes.get(n.input[0]) == onnx.TensorProto.FLOAT16:
        # 3) fp16 LayerNormalization — DML은 분산을 fp16으로 계산해 입력이 큰 곳(마지막 층: 잔차 |x|~7000, x²>65504)에서 넘친다.
        #    (CPU EP는 stash_type=1대로 fp32로 계산해 멀쩡해서 CPU 검증으로는 안 보인다.) 입력·계수를 fp32로 올려 계산하고 fp16으로 내린다.
        ins32 = []
        for k, x in enumerate(n.input):
            y = f"{n.name}_wf_in{k}_f32"
            new_nodes.append(helper.make_node("Cast", [x], [y], to=onnx.TensorProto.FLOAT, name=f"{n.name}_wf_c{k}"))
            ins32.append(y)
        ln = helper.make_node("LayerNormalization", ins32, [n.name + "_wf_y32"], name=n.name)
        ln.attribute.extend(n.attribute)
        new_nodes.append(ln)
        new_nodes.append(helper.make_node("Cast", [n.name + "_wf_y32"], [n.output[0]], to=onnx.TensorProto.FLOAT16, name=n.name + "_wf_co"))
        n_ln += 1
    else:
        if n.op_type == "MatMul" and a and len(a) >= 5:
            print("  [skip] 5D MatMul인데 형태가 달라 그대로 둠:", n.name, a, b)
        new_nodes.append(n)
del g.node[:]
g.node.extend(new_nodes)
g.initializer.extend(new_inits)
onnx.save(m, dst)   # 외부 데이터 참조(location)는 그대로 — src와 같은 폴더에 두거나 .data를 같이 둔다
print(f"[webnn_fix] 5D MatMul {n_fix}개를 3D로 접음, allowzero Reshape {n_rs}개를 정적 shape으로, fp16 LayerNorm {n_ln}개를 fp32로 → {dst}")
