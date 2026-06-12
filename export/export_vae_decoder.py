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



def patch_causal_conv_t1_to_2d(model):
    """CausalConv3d를 T=1 전용 2D conv로 치환 (모델 로드 후, export 전 호출).

    가중치는 마지막 시간 탭만 남겨 4D로 실체화 → ONNX에 정적 4D initializer.

    Wan VAE 디코더의 모든 conv는 CausalConv3d(시간축 causal zero-pad, 커널 보통 3x3x3).
    T=1 입력에서는 causal 패딩이 [0, 0, x]라 마지막 시간 탭(weight[:, :, -1])만
    실제 데이터를 보므로, 그 탭만 쓰는 2D conv와 수학적으로 완전 동치.

    효과:
    1) 구버전 ComfyUI의 zero-pad 경로였다면 시간축 3탭 헛연산 제거 (FLOP 1/3)
    2) ONNX에 rank-4 Conv로 박혀 ORT WebGPU의 최적화된 Conv2d 커널을 탐
       (rank-5 Conv는 비최적 경로 — JSEP 커널 파일명이 conv3d_naive일 정도)

    전제: T=1 decode에서는 feat_cache=None이라 Resample.time_conv가 호출되지
    않음 (wan/vae.py decode: iter_==1이면 feat_map=None). 비디오 디코드 금지.
    """
    import torch.nn.functional as F
    import comfy.ldm.wan.vae as wv

    # 가중치를 export 전에 4D로 실체화한다. forward 안에서 weight[:, :, -1]을
    # 슬라이스하면 그래프에 Slice/Gather가 남는데, exporter constant folding이
    # 이를 못 접는 경우 ONNX가 3탭 원본 가중치를 통째로 싣고 다니게 됨
    # (파일 ~3배, ORT prepack 저해). 미리 잘라두면 4D initializer로 직행.
    n = 0
    for mod in model.modules():
        if isinstance(mod, wv.CausalConv3d):
            with torch.no_grad():
                mod.weight = torch.nn.Parameter(
                    mod.weight[:, :, -1].contiguous(), requires_grad=False)
            n += 1

    def _t1_forward(self, x, cache_x=None, cache_list=None, cache_idx=None):
        assert x.shape[2] == 1, "T=1 전용 패치 — 비디오 디코드에 사용 불가"
        y = F.conv2d(x.squeeze(2), self.weight, self.bias,
                     stride=self.stride[1:], padding=self.padding[1:],
                     dilation=self.dilation[1:], groups=self.groups)
        return y.unsqueeze(2)

    wv.CausalConv3d.forward = _t1_forward
    print(f"[patch] CausalConv3d {n}개 → T=1 등가 2D conv 치환 "
          f"(가중치 4D 실체화) 완료")


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
    p.add_argument("--keep-3d", action="store_true", dest="keep_3d",
                   help="CausalConv3d→2D 치환을 끄고 기존 3D conv 그대로 export "
                        "(회귀 비교용)")
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

    # 동치 검증은 수식 일치 확인이 목적 — CUDA 기본값인 TF32(가수 10비트)는
    # conv3d/conv2d가 다른 알고리즘을 탈 때 1e-3급 차이를 만들어 검증을
    # 오염시키므로 끈다. (export 결과물엔 영향 없음, 검증이 약간 느려질 뿐)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    with torch.inference_mode():
        ref = wrapper(latent)          # 패치 전 원본 3D 경로 (레퍼런스)
        print(f"[sanity] output: {tuple(ref.shape)}, "
              f"range [{ref.min():.3f}, {ref.max():.3f}]")
        assert not torch.isnan(ref).any(), "NaN 발생 — fp32 로드로 재시도 권장"

    if not args.keep_3d:
        patch_causal_conv_t1_to_2d(model)
        with torch.inference_mode():
            out = wrapper(latent)
            diff = (out - ref).abs().max().item()
            print(f"[equiv] 3D vs 2D 최대 오차: {diff:.2e} (기대: fp32 커널 차이 수준)")
            assert diff < 1e-3, "2D 치환 동치 검증 실패 — --keep-3d로 우회 후 보고"

    with torch.inference_mode():
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
