"""
Anima DiT (MiniTrainDIT 2B) → ONNX export.

핵심 포인트:
- 반드시 fp16으로 로드한다. MiniTrainDIT._forward에는
  `if x.dtype == torch.float16: x = x.float()` 분기가 있어서,
  fp16으로 trace하면 "residual stream은 fp32, attention/MLP는 fp16" 캐스팅
  패턴이 그래프에 그대로 박힌다. (bf16은 WebGPU에 없으므로 이 패턴이 필수)
- WrapperExecutor(ComfyUI 런타임 배관)를 우회하기 위해 _forward를 직접 호출.
- 해상도 고정 export: RoPE 임베딩이 trace 시 상수로 구워진다.
  여러 해상도가 필요하면 --size를 바꿔 여러 번 export.
- LLMAdapter는 여기 포함하지 않음 (export_adapter.py에서 별도 export.
  어댑터는 프롬프트당 1회만 돌면 되므로 스텝 루프 그래프에서 분리하는 게 맞음).

사용 예:
  python export_dit.py \
    --comfyui ~/ComfyUI \
    --ckpt ~/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors \
    --out out/dit/anima_dit_1024.onnx \
    --size 1024 --verify

Turbo LoRA 머지본을 쓰려면 ComfyUI에서 LoRA 머지 후 저장한 safetensors를
--ckpt로 주면 된다 (구조 동일).
"""
import torch

import os
import sys

# ComfyUI portable의 embedded python은 ._pth 파일 때문에 스크립트 폴더를
# sys.path에 넣지 않으므로 (PYTHONPATH도 무시됨) 직접 등록한다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (base_argparser, patch_comfy_for_export, save_onnx_external,
                    setup_comfyui, verify_onnx)

LATENT_CHANNELS = 16     # Wan21 latent format
VAE_DOWNSCALE = 8
CONTEXT_LEN = 512        # LLMAdapter 출력은 항상 512 토큰으로 패딩됨
CONTEXT_DIM = 1024       # LLMAdapter target_dim


class DitWrapper(torch.nn.Module):
    """WrapperExecutor를 건너뛰고 _forward를 직접 노출.
    fp32_act 모드: 입출력은 fp16, 내부 연산은 fp32 (캐스팅을 그래프에 굽는다)."""

    def __init__(self, dit, fp32_act=False):
        super().__init__()
        self.dit = dit
        self.fp32_act = fp32_act

    def forward(self, latent, timesteps, context):
        # latent:    (1, 16, 1, H/8, W/8) fp16
        # timesteps: (1,) fp32  — sigma 그 자체, 0~1 범위 (multiplier=1.0)
        # context:   (1, 512, 1024) fp16 — LLMAdapter 출력
        if self.fp32_act:
            out = self.dit._forward(latent.float(), timesteps, context.float())
            return out.half()
        return self.dit._forward(latent, timesteps, context)


def main():
    p = base_argparser("Anima DiT → ONNX")
    p.add_argument("--ckpt", required=True,
                   help="anima-base-v1.0.safetensors (또는 Turbo 머지본) 경로")
    p.add_argument("--size", type=int, default=1024,
                   help="픽셀 해상도 (trace용 기준 해상도, 기본 1024)")
    p.add_argument("--dynamic", action="store_true",
                   help="해상도 동적 export: H/W를 dynamic axes로 열어 "
                        "한 그래프로 모든 해상도/종횡비 지원 (RoPE가 셰이프 "
                        "의존 연산으로 그래프에 포함됨)")
    p.add_argument("--lora", default=None,
                   help="LoRA safetensors 경로 (예: Turbo LoRA). 지정 시 "
                        "export 전에 가중치에 머지됨 — 별도 머지 단계 불필요")
    p.add_argument("--lora-strength", type=float, default=1.0,
                   dest="lora_strength")
    p.add_argument("--fp32-act", action="store_true", dest="fp32_act",
                   help="활성(연산)을 fp32로 유지. WebGPU 진짜-fp16 환경에서 "
                        "활성값 오버플로(NaN) 방지. 입출력은 fp16 유지라 "
                        "웹페이지 수정 불필요. 가중치는 이후 quantize_dit로 "
                        "int4 양자화하면 됨.")
    args = p.parse_args()

    setup_comfyui(args.comfyui)
    patch_comfy_for_export()

    import comfy.sd

    load_dtype = torch.float32 if args.fp32_act else torch.float16
    print(f"[load] {args.ckpt} (dtype={load_dtype})")
    patcher = comfy.sd.load_diffusion_model(
        args.ckpt, model_options={"dtype": load_dtype})
    dit = patcher.model.diffusion_model
    dit.eval().to(args.device)

    if args.lora:
        import comfy.utils
        import comfy.model_management as mm
        print(f"[lora] {args.lora} (strength {args.lora_strength}) 머지 중...")
        lora_sd = comfy.utils.load_torch_file(args.lora, safe_load=True)
        patcher, _ = comfy.sd.load_lora_for_models(
            patcher, None, lora_sd, args.lora_strength, 0)
        try:
            mm.load_models_gpu([patcher], force_full_load=True)
        except TypeError:
            try:
                mm.load_models_gpu([patcher])
            except Exception:
                patcher.patch_model()
        dit = patcher.model.diffusion_model
        dit.eval().to(args.device)
        print("[lora] 머지 완료 (가중치 실체화됨)")

    n_params = sum(t.numel() for t in dit.parameters())
    print(f"[load] 파라미터 수: {n_params / 1e9:.2f}B")

    wrapper = DitWrapper(dit, fp32_act=args.fp32_act).eval()

    h = w = args.size // VAE_DOWNSCALE
    latent = torch.randn(1, LATENT_CHANNELS, 1, h, w,
                         dtype=torch.float16, device=args.device)
    timesteps = torch.tensor([1.0], dtype=torch.float32, device=args.device)
    context = torch.randn(1, CONTEXT_LEN, CONTEXT_DIM,
                          dtype=torch.float16, device=args.device)

    dyn = None
    if args.dynamic:
        dyn = {"latent": {3: "lat_h", 4: "lat_w"},
               "v_pred": {3: "lat_h", 4: "lat_w"}}
        print("[dynamic] H/W dynamic axes 활성화")

    with torch.inference_mode():
        # trace 전 한 번 실행해서 forward가 도는지부터 확인
        out = wrapper(latent, timesteps, context)
        print(f"[sanity] output shape: {tuple(out.shape)}, dtype: {out.dtype}")
        assert not torch.isnan(out).any(), "NaN 발생 — fp16 수치 문제. 먼저 해결 필요."

        save_onnx_external(
            None, wrapper, (latent, timesteps, context), args.out,
            input_names=["latent", "timesteps", "context"],
            output_names=["v_pred"],
            dynamic_axes=dyn,
            use_dynamo=args.dynamo,
        )

    if args.verify:
        verify_onnx(args.out, wrapper, (latent, timesteps, context),
                    ["latent", "timesteps", "context"])
        if args.dynamic:
            # 다른 해상도/종횡비에서도 그래프가 올바른지 확인 (832x1216 세로 버킷)
            h2, w2 = 1216 // VAE_DOWNSCALE, 832 // VAE_DOWNSCALE
            print(f"[verify] 동적 해상도 검증: latent {h2}x{w2} (832x1216px)")
            latent2 = torch.randn(1, LATENT_CHANNELS, 1, h2, w2,
                                  dtype=torch.float16, device=args.device)
            verify_onnx(args.out, wrapper, (latent2, timesteps, context),
                        ["latent", "timesteps", "context"])


if __name__ == "__main__":
    main()
