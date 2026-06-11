"""
Qwen-Image VAE (= Wan 2.1 VAE) 디코더 → ONNX export.

Wan21 latent 역정규화(process_out: z * std + mean)를 그래프 안에 굽는다.
→ 웹 쪽에서는 DiT가 뱉는 latent를 그대로 넣으면 픽셀이 나옴.

입력: (1, 16, 1, H/8, W/8) fp16  — 샘플링 끝난 latent (정규화 상태 그대로)
출력: (1, 3, H, W) fp32, 값 범위 [-1, 1] → 픽셀 = (x + 1) / 2

사용 예:
  python export_vae_decoder.py \
    --comfyui ~/ComfyUI \
    --vae ~/ComfyUI/models/vae/qwen_image_vae.safetensors \
    --out out/vae/qwen_image_vae_decoder_1024.onnx --size 1024 --verify
"""
import torch

import os
import sys

# ComfyUI portable의 embedded python은 ._pth 파일 때문에 스크립트 폴더를
# sys.path에 넣지 않으므로 (PYTHONPATH도 무시됨) 직접 등록한다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import base_argparser, setup_comfyui, verify_onnx

# comfy/latent_formats.py Wan21 상수 (확인됨)
WAN21_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517,
              1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497,
              0.2503, -0.2921]
WAN21_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
             3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae = vae_model
        self.register_buffer(
            "mean", torch.tensor(WAN21_MEAN).view(1, 16, 1, 1, 1))
        self.register_buffer(
            "std", torch.tensor(WAN21_STD).view(1, 16, 1, 1, 1))

    def forward(self, latent):
        # process_out (scale_factor=1.0): z * std + mean — 이후 전부 fp32 유지
        z = latent.float() * self.std + self.mean
        img = self.vae.decode(z)              # (1, 3, T, H, W) fp32
        return img.float().clamp(-1, 1)[:, :, 0]  # (1, 3, H, W)


def main():
    p = base_argparser("Qwen-Image VAE decoder → ONNX")
    p.add_argument("--vae", required=True,
                   help="qwen_image_vae.safetensors 경로")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--dynamic", action="store_true",
                   help="해상도 동적 export (H/W dynamic axes)")
    args = p.parse_args()

    setup_comfyui(args.comfyui)

    import comfy.sd
    import comfy.utils

    print(f"[load] {args.vae}")
    sd = comfy.utils.load_torch_file(args.vae)
    vae = comfy.sd.VAE(sd=sd)
    # fp32 필수: Wan VAE는 내부 활성값이 fp16 한계를 넘어 NaN(검은 이미지) 발생.
    # WebGPU EP는 진짜 fp16으로 연산하므로 CPU/DML과 달리 오버플로가 실제로 터진다.
    model = vae.first_stage_model.eval().to(args.device).float()

    wrapper = VAEDecoderWrapper(model).eval().to(args.device)

    h = w = args.size // 8
    latent = torch.randn(1, 16, 1, h, w,
                         dtype=torch.float16, device=args.device)

    with torch.inference_mode():
        out = wrapper(latent)
        print(f"[sanity] output: {tuple(out.shape)}, "
              f"range [{out.min():.3f}, {out.max():.3f}]")
        assert not torch.isnan(out).any(), "NaN 발생 — fp32 로드로 재시도 권장"

        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        dyn = ({"latent": {3: "lat_h", 4: "lat_w"},
                "image": {2: "img_h", 3: "img_w"}} if args.dynamic else None)
        torch.onnx.export(
            wrapper, (latent,), args.out,
            input_names=["latent"], output_names=["image"],
            dynamic_axes=dyn, opset_version=18,
        )
        print(f"[export] {args.out} 저장 완료")

    if args.verify:
        verify_onnx(args.out, wrapper, (latent,), ["latent"])


if __name__ == "__main__":
    main()
