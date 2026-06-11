"""
Anima LLMAdapter → ONNX export.

어댑터는 DiT 체크포인트 안에 llm_adapter.* 키로 들어있는 6층 미니 트랜스포머.
입력:
  - source_hidden_states: Qwen3 0.6B 인코더의 hidden states (1, S_qwen, 1024)
  - target_input_ids:     같은 프롬프트의 T5 토큰 id (1, S_t5) — vocab 32128
출력:
  - (1, S_t5, 1024)  ※ 512 토큰 zero-pad는 그래프 밖(JS)에서 수행

시퀀스 길이는 dynamic axes로 열어둠 (모델이 작아서 부담 없음).
프롬프트당 1회만 실행되므로 DiT 스텝 루프와 분리.

사용 예:
  python export_adapter.py \
    --comfyui ~/ComfyUI \
    --ckpt ~/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors \
    --out out/adapter/anima_llm_adapter.onnx --verify
"""
import torch

import os
import sys

# ComfyUI portable의 embedded python은 ._pth 파일 때문에 스크립트 폴더를
# sys.path에 넣지 않으므로 (PYTHONPATH도 무시됨) 직접 등록한다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (base_argparser, patch_comfy_for_export, save_onnx_external,
                    setup_comfyui, verify_onnx)


class AdapterWrapper(torch.nn.Module):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter

    def forward(self, qwen_hidden_states, t5_input_ids):
        # 추론 경로(model_base.Anima.extra_conds)에서는 attention mask를
        # 넘기지 않으므로 동일하게 mask 없이 export.
        return self.adapter(qwen_hidden_states, t5_input_ids)


def main():
    p = base_argparser("Anima LLMAdapter → ONNX")
    p.add_argument("--ckpt", required=True,
                   help="anima-base-v1.0.safetensors 경로 (DiT와 동일 파일)")
    args = p.parse_args()

    setup_comfyui(args.comfyui)
    patch_comfy_for_export()

    import comfy.sd

    print(f"[load] {args.ckpt} (fp16)")
    patcher = comfy.sd.load_diffusion_model(
        args.ckpt, model_options={"dtype": torch.float16})
    adapter = patcher.model.diffusion_model.llm_adapter
    adapter.eval().to(args.device)

    n_params = sum(t.numel() for t in adapter.parameters())
    print(f"[load] 어댑터 파라미터 수: {n_params / 1e6:.1f}M")

    wrapper = AdapterWrapper(adapter).eval()

    s_qwen, s_t5 = 77, 64  # trace용 임의 길이 (dynamic axes로 가변)
    hidden = torch.randn(1, s_qwen, 1024,
                         dtype=torch.float16, device=args.device)
    ids = torch.randint(0, 32128, (1, s_t5),
                        dtype=torch.int64, device=args.device)

    with torch.inference_mode():
        out = wrapper(hidden, ids)
        print(f"[sanity] output shape: {tuple(out.shape)}, dtype: {out.dtype}")
        assert not torch.isnan(out).any(), "NaN 발생"

        save_onnx_external(
            None, wrapper, (hidden, ids), args.out,
            input_names=["qwen_hidden_states", "t5_input_ids"],
            output_names=["context"],
            dynamic_axes={
                "qwen_hidden_states": {1: "qwen_seq"},
                "t5_input_ids": {1: "t5_seq"},
                "context": {1: "t5_seq"},
            },
            use_dynamo=args.dynamo,
        )

    if args.verify:
        verify_onnx(args.out, wrapper, (hidden, ids),
                    ["qwen_hidden_states", "t5_input_ids"])


if __name__ == "__main__":
    main()
