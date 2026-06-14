"""
실모델 VAE fuse 전후 수치 검증 (CPU EP).

fuse 전(_2d) vs 후(_2df)를 같은 latent로 디코드해 출력 비교.
CPU EP는 MHA를 UnfusedAttention으로 폴백하나 결과는 원본 SDPA와 일치해야 함.
일치하면 fuse가 수학적으로 정확 → 브라우저(FA2)서도 동일 결과 기대.

작은 latent(예 32)로 빠르게. 출력 (1,3,256,256) 비교.

사용 (한 줄):
  python_embeded\python.exe export\verify_vae_fuse.py --a out\vae\qwen_image_vae_decoder_dyn32_2d.onnx --b out\vae\qwen_image_vae_decoder_dyn32_2df.onnx --lat 32
"""
import argparse

import numpy as np
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="fuse 전")
    p.add_argument("--b", required=True, help="fuse 후")
    p.add_argument("--lat", type=int, default=32)
    p.add_argument("--c", type=int, default=16)
    args = p.parse_args()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    sa = ort.InferenceSession(args.a, so, providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(args.b, so, providers=["CPUExecutionProvider"])

    inp = sa.get_inputs()[0].name
    fp16 = sa.get_inputs()[0].type == "tensor(float16)"
    dt = np.float16 if fp16 else np.float32
    np.random.seed(0)
    lat = (np.random.randn(1, args.c, 1, args.lat, args.lat) * 0.5).astype(dt)

    oa = np.asarray(sa.run(None, {inp: lat})[0], np.float32)
    ob = np.asarray(sb.run(None, {inp: lat})[0], np.float32)
    print(f"[a] {oa.shape}  [b] {ob.shape}")
    if oa.shape != ob.shape:
        print("[!] shape 불일치 — fuse 출력 레이아웃 문제")
        return
    d = np.abs(oa - ob)
    rng = float(oa.max() - oa.min())
    print(f"[diff] 최대 절대차 {d.max():.6e} / 평균 {d.mean():.6e}")
    print(f"[diff] 출력 범위 {rng:.4f}, 상대 최대차 {d.max()/rng:.6e}")
    if d.max() < 2e-3:
        print("==> 수치 동일. fuse 정확. 브라우저 FA2 실측 진행 가능.")
    else:
        print("==> 차이 큼! K/V src 또는 scale/레이아웃 점검 필요.")
        # 어디서 깨지는지 힌트: 채널별 최대차
        per_c = d[0].reshape(3, -1).max(axis=1)
        print(f"    채널별 최대차(R,G,B): {per_c}")


if __name__ == "__main__":
    main()
