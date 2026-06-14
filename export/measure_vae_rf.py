"""
VAE 디코더의 receptive field(RF) 실측 — tiled VAE의 crop 폭 산정 근거.

방법: 정규화 latent 중앙 1픽셀(전 채널)에만 임펄스를 주고 나머지는 0(=정규화 평균 상태)로
디코드 → 출력에서 baseline(전 0 입력) 대비 달라진 픽셀의 최대 반경을 잰다.
그 반경이 "한 출력 픽셀이 보는 입력 범위"의 절반 = RF/2. crop overlap은 이 값(픽셀)을
8로 나눈 latent 단위로 잡으면 seam 없이 단일호출과 동치.

conv-only VAE라 RF는 유한·위치불변. attention 있으면 이 방법 무의미(전역) — 우리는 conv-only라 OK.

사용 (한 줄):
  python_embeded\python.exe export\measure_vae_rf.py --vae out\vae\qwen_image_vae_decoder_dyn32_2d.onnx --lat 32
"""
import argparse
import numpy as np
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", required=True)
    p.add_argument("--lat", type=int, default=32, help="latent H=W (픽셀 = lat*8)")
    p.add_argument("--c", type=int, default=16, help="latent 채널")
    p.add_argument("--thr", type=float, default=1e-3,
                   help="baseline 대비 |차이| 임계 (이 이상이면 '영향 받음')")
    args = p.parse_args()

    L = args.lat
    sess = ort.InferenceSession(args.vae, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    # dtype 자동 (fp16 입력 모델이면 fp16)
    want_fp16 = sess.get_inputs()[0].type == "tensor(float16)"
    dt = np.float16 if want_fp16 else np.float32

    def decode(latent):
        out = sess.run(None, {inp: latent.astype(dt)})[0]
        return np.asarray(out, np.float32)  # (1,3,L*8,L*8)

    base = decode(np.zeros((1, args.c, 1, L, L)))           # baseline
    imp = np.zeros((1, args.c, 1, L, L))
    cx = cy = L // 2
    imp[0, :, 0, cy, cx] = 10.0                              # 중앙 1 latent에 큰 임펄스
    resp = decode(imp)

    diff = np.abs(resp - base).max(axis=1)[0]               # (L*8, L*8) 채널 최대
    px = L * 8
    pcx, pcy = cx * 8 + 4, cy * 8 + 4                        # 임펄스의 픽셀 중심(8배+중앙)
    ys, xs = np.where(diff > args.thr)
    if len(xs) == 0:
        print(f"[rf] 임계 {args.thr} 초과 픽셀 없음 — thr 낮추거나 임펄스 키우기")
        return
    rad = max(np.abs(xs - pcx).max(), np.abs(ys - pcy).max())
    print(f"[rf] latent {L}({px}px) · 임펄스 중심 픽셀=({pcx},{pcy})")
    print(f"[rf] 영향 반경(픽셀) = {rad}  → RF ≈ {2*rad+1}px")
    print(f"[rf] latent 단위 overlap = ceil(rad/8) = {-(-int(rad)//8)} (한쪽)")
    print(f"[rf] 권장 crop overlap(양쪽 여유 포함) = {-(-int(rad)//8)+1} latent")
    # 경계 도달 여부 경고 (latent가 작아 RF가 잘렸을 수)
    if rad >= px // 2 - 8:
        print(f"[rf] (경고) 반경이 출력 경계에 근접 — --lat 더 키워 재측정 권장")


if __name__ == "__main__":
    main()
