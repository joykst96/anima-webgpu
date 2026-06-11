"""
Qwen3 0.6B Base 텍스트 인코더 → ONNX export.

ComfyUI는 qwen_3_06b_base.safetensors(= Qwen/Qwen3-0.6B-Base)를
sd1_clip.SDClipModel(layer="last")로 감싸서 최종 hidden states를 뽑는다.
여기서는 HF transformers로 동일 모델을 로드해서 last_hidden_state를 export.

※ 주의: ComfyUI 쪽 토크나이저가 프롬프트에 prefix/template을 붙이는지는
  comfy/text_encoders/anima.py의 Qwen3Tokenizer 설정을 따라야 한다.
  웹 파이프라인 연결 전에 동일 프롬프트로 ComfyUI 임베딩과 본 export의
  출력을 비교하는 검증을 한 번 거칠 것 (README 4번 항목).

사용 예:
  python export_text_encoder.py \
    --model Qwen/Qwen3-0.6B-Base \
    --out out/text_encoder/qwen3_06b.onnx --verify
"""
import argparse

import torch


class TEWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


def main():
    p = argparse.ArgumentParser(description="Qwen3 0.6B → ONNX")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base",
                   help="HF 모델 id 또는 로컬 경로")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    from transformers import AutoModel, AutoTokenizer

    print(f"[load] {args.model}")
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.float16).eval().to(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)

    wrapper = TEWrapper(model).eval()

    enc = tok("masterpiece, 1girl, looking at viewer", return_tensors="pt")
    input_ids = enc["input_ids"].to(args.device)
    attn = enc["attention_mask"].to(args.device)

    with torch.inference_mode():
        out = wrapper(input_ids, attn)
        print(f"[sanity] hidden states: {tuple(out.shape)}")  # (1, S, 1024)

        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        torch.onnx.export(
            wrapper, (input_ids, attn), args.out,
            input_names=["input_ids", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids": {1: "seq"},
                "attention_mask": {1: "seq"},
                "hidden_states": {1: "seq"},
            },
            opset_version=18,
        )
        print(f"[export] {args.out} 저장 완료")

    if args.verify:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(
            args.out,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        ort_out = sess.run(None, {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attn.cpu().numpy(),
        })[0]
        ref = out.detach().float().cpu().numpy()
        diff = float(np.max(np.abs(ref - ort_out.astype(np.float32))))
        print(f"[verify] max_abs_diff={diff:.5f}")


if __name__ == "__main__":
    main()
