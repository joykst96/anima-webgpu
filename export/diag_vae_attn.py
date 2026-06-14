"""
VAE 디코더의 attention(Softmax 주변) 구조/해상도 진단.

목적: attention이 어느 해상도에서 도는지, score 행렬 크기가 얼마인지 파악.
1280² latent(H=W=160) VAE OOM의 원인이 이 attention score (S²) 행렬인지 확정.
attention이 latent 해상도(다운샘플 없음)에서 돌면 S=H*W라 1280²서 S=25600,
score = S² fp32 = 2.6GB > 2GB 한계 → OOM 원인.

사용 (한 줄, shape 추론 위해 입력 크기 지정):
  python_embeded\python.exe export\diag_vae_attn.py --vae out\vae\..._2d.onnx --lat 128
"""
import argparse

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", required=True)
    p.add_argument("--lat", type=int, default=128, help="latent H=W (shape 추론용)")
    p.add_argument("--c", type=int, default=16)
    args = p.parse_args()

    m = onnx.load(args.vae, load_external_data=False)
    g = m.graph

    # Softmax/MatMul 노드와 그 입출력 이름
    sm = [n for n in g.node if n.op_type == "Softmax"]
    mms = [n for n in g.node if n.op_type == "MatMul"]
    print(f"[attn] Softmax {len(sm)}개, MatMul {len(mms)}개")
    for n in sm:
        ax = next((a.i for a in n.attribute if a.name == "axis"), "?")
        print(f"  Softmax {n.name}: in={list(n.input)} out={list(n.output)} axis={ax}")
    for n in mms:
        print(f"  MatMul {n.name}: in={list(n.input)} out={list(n.output)}")

    # 실측 대신 정적 shape inference (external .data 불필요).
    # latent 입력 shape을 고정해서 중간 텐서 shape을 전파시킨다.
    print(f"\n[shape] latent {args.lat}({args.lat*8}px)로 attention 텐서 shape 추론...")
    want = set()
    for n in sm:
        want.update(n.input)
        want.update(n.output)
    for n in mms:
        want.update(n.input)
        want.update(n.output)

    # 입력 shape 고정
    inp_vi = g.input[0]
    dims = inp_vi.type.tensor_type.shape.dim
    fixed = [1, args.c, 1, args.lat, args.lat]
    for d, val in zip(dims, fixed):
        d.ClearField("dim_param")
        d.dim_value = val

    inferred = shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
    vimap = {vi.name: vi for vi in list(inferred.graph.value_info)
             + list(inferred.graph.output) + list(inferred.graph.input)}

    def shp(name):
        vi = vimap.get(name)
        if vi is None:
            return None
        return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                for d in vi.type.tensor_type.shape.dim]

    for name in sorted(want):
        s = shp(name)
        if s is None:
            print(f"  {name}: (shape 미추론)")
            continue
        print(f"  {name}: {s}")
        if len(s) >= 2 and isinstance(s[-1], int) and isinstance(s[-2], int) \
                and s[-1] == s[-2]:
            S = s[-1]
            gb = S * S * 4 / 2**30
            print(f"    → score 정사각 S={S}, S² fp32 = {gb:.3f} GiB "
                  f"({'초과!' if gb > 2 else 'OK'})")
    return


if __name__ == "__main__":
    main()
