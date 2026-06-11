# Anima WebGPU — Browser-Local Anime Image Generation

English | [한국어](README.ko.md)

Run [Anima](https://huggingface.co/circlestone-labs/Anima) (CircleStone Labs / Comfy Org, 2B DiT) entirely in the browser. No server-side inference, no installation — weights download once, then every generation runs on the visitor's own GPU via WebGPU.

This repo contains the full pipeline: PyTorch/ComfyUI → ONNX export scripts, weight-only quantization, shard packaging for browsers/CDNs, a single-file web frontend, and deployment configs.

**Highlights**

- 100% client-side inference (ONNX Runtime Web, WebGPU EP). Prompts never leave the device.
- Dynamic resolution & aspect ratio from a single graph (RoPE traced symbolically — no per-resolution builds).
- Faithful port of ComfyUI sampling: `er_sde` (CONST/flow path) and `simple` scheduler, numerically matched.
- int8 weight-only quantization (MatMulNBits): fp16 compute, ~2.3 GB download, runs on any WebGPU GPU with `shader-f16` — no INT8 tensor-core requirement.
- Weight sharding (≤ ~480 MB/file) to bypass the browser's single-ArrayBuffer limit and fit free-tier CDN cache limits.
- OPFS caching: first visit downloads, later visits load from disk in seconds.
- Two model variants in one page: base (er_sde · 30 steps · CFG 5) and Turbo-LoRA-merged (euler · 8 steps · CFG 1).

## How it works

| Stage | Model | Format | Notes |
|---|---|---|---|
| Text encoding | Qwen3-0.6B-Base | ONNX fp16 | last hidden states |
| Context adapter | Anima LLMAdapter | ONNX fp16 | dual tokenization: Qwen3 ids → hidden states, T5 ids → adapter queries; output zero-padded to 512×1024 |
| Denoiser | Anima DiT 2B (Cosmos-Predict2) | ONNX, int8 weights / fp32 activations, fp16 I/O | dynamic H/W axes, RoPE in-graph |
| Decode | Qwen-Image (Wan 2.1) VAE decoder | ONNX fp32 | Wan21 latent de-normalization baked in |

The sampler (Euler / ER-SDE-Solver) and σ schedule run in plain JavaScript; each step calls the DiT session once (CFG 1) or twice (CFG > 1).

## Requirements

**Conversion machine** — a working ComfyUI install that can already run Anima, Python 3.10+, recent PyTorch (the new dynamo ONNX exporter is used), `onnx`, `onnxruntime`, `transformers`, `sentencepiece`. A 16 GB GPU is comfortable; fp32 export of the DiT peaks around 10 GB VRAM. ComfyUI **portable** users: replace `python` below with `python_embeded\python.exe` and run from the portable root.

**Serving** — any static file server. HTTPS is mandatory in production (WebGPU and OPFS require a secure context; `localhost` is exempt for testing).

**Client** — a WebGPU browser (desktop Chrome/Edge), GPU with `shader-f16` support, ~6 GB VRAM recommended. First visit downloads ~2.5 GB (then cached in OPFS).

## 1. Convert the models

Model files needed: `anima-base-v1.0.safetensors`, the official **Anima Turbo LoRA** (optional, for the fast variant), `qwen_image_vae.safetensors`. Qwen3-0.6B-Base is fetched from HF automatically.

```bash
# DiT — base variant (fp32 activations are REQUIRED for WebGPU; see Pitfalls)
python export/export_dit.py --comfyui /path/to/ComfyUI \
  --ckpt /path/to/anima-base-v1.0.safetensors \
  --out out/dit/anima_dit_dyn32.onnx --dynamic --fp32-act

# DiT — Turbo variant (LoRA is merged at export time, no separate merge step)
python export/export_dit.py --comfyui /path/to/ComfyUI \
  --ckpt /path/to/anima-base-v1.0.safetensors \
  --lora /path/to/anima_turbo.safetensors \
  --out out/dit/anima_dit_turbo32.onnx --dynamic --fp32-act

# Adapter (extracted from the same checkpoint) / text encoder / VAE
python export/export_adapter.py --comfyui /path/to/ComfyUI \
  --ckpt /path/to/anima-base-v1.0.safetensors --out out/adapter/anima_llm_adapter.onnx
python export/export_text_encoder.py --out out/text_encoder/qwen3_06b.onnx
python export/export_vae_decoder.py --comfyui /path/to/ComfyUI \
  --vae /path/to/qwen_image_vae.safetensors \
  --out out/vae/qwen_image_vae_decoder_dyn32.onnx --dynamic --size 512
```

Notes: `--size 512` on the VAE only sets the trace size (the graph is dynamic); fp32 VAE traced at 1024 can OOM a 16 GB card. `export_dit.py --verify` additionally runs an ONNX-vs-PyTorch comparison at two resolutions.

## 2. Quantize & shard the DiT

```bash
# int8 weight-only (recommended; near-lossless for this DiT)
python export/quantize_dit.py --src out/dit/anima_dit_dyn32.onnx \
  --out out/dit/anima_dit_dyn32_q8.onnx --bits 8 --no-exclude

# split weights into CDN/browser-friendly shards
python export/shard_onnx_data.py --src out/dit/anima_dit_dyn32_q8.onnx \
  --out out/dit/anima_dit_dyn32_q8s.onnx --shard-mb 400
```

Repeat both for the Turbo export. Also shard anything else whose `.data` exceeds your CDN's cache limit (the fp16 text encoder is ~1.2 GB):

```bash
python export/shard_onnx_data.py --src out/text_encoder/qwen3_06b.onnx \
  --out out/text_encoder/qwen3_06b_s.onnx --shard-mb 400
```

Why these choices: **int4 RTN visibly destroys this DiT** (diffusion transformers are far more weight-quantization-sensitive than LLMs — the reason SVDQuant exists). int8 RTN is effectively lossless. `--algo hqq` provides a higher-quality int4 (~1.2 GB) if download size matters more than fidelity. Quantization is weight-only: shaders dequantize to fp16/fp32 at run time, so there is **no hardware INT8 requirement** and, since batch-1 inference is bandwidth-bound, it is also faster than fp16.

## 3. Verify locally (recommended)

```bash
pip install onnxruntime pillow   # or onnxruntime-directml / onnxruntime-gpu
python export/run_pipeline.py \
  --dit out/dit/anima_dit_dyn32_q8s.onnx \
  --adapter out/adapter/anima_llm_adapter.onnx \
  --te out/text_encoder/qwen3_06b_s.onnx \
  --vae out/vae/qwen_image_vae_decoder_dyn32.onnx \
  --prompt "1girl, silver hair" --steps 30 --cfg 5.0 --size 768 --out check.png
```

This runs the exact pipeline the browser uses (same schedule/sampler math). If `check.png` looks right, any later browser-side problem is frontend code, not the models — that separation saved this project repeatedly.

## 4. Tokenizers (self-hosted)

```bash
python export/prepare_tokenizers.py --out tokenizers
```

Converts Qwen3 and the genuine `t5-v1_1` tokenizers to the fast `tokenizer.json` format transformers.js requires. The page tries `tokenizers/` locally first and falls back to HF Hub if absent.

## 5. Deploy

Web-root layout:

```
index.html
tokenizers/{qwen3,t5}/
out/
├── dit/           *_q8s.onnx + .bin shards + .manifest.json   (per variant)
├── adapter/       anima_llm_adapter.onnx (+ .data)
├── text_encoder/  qwen3_06b_s.onnx + shards + manifest
└── vae/           qwen_image_vae_decoder_dyn32.onnx (+ .data)
```

Edit the `PATHS` / `MODELS` constants at the top of `web/index.html` if your filenames differ. Do **not** leave un-quantized source models in the web root.

A ready-to-run static server is in `deploy/` (`docker compose up -d`; nginx with `immutable` cache headers for weights, `no-cache` for the page). Behind Cloudflare, add a Cache Rule making `.bin/.onnx/.data/.json` cache-eligible — these extensions are not cached by default, and without the rule every request hits your origin. Keep shards ≤ ~500 MB for free-tier per-file cache limits. Because weights are served `immutable`, **never overwrite a model file in place** — upload under a new filename and update `index.html`.

## Implementation notes (for porting elsewhere)

Hard-won facts about Anima's inference contract, verified against ComfyUI source:

- Flow model, `ModelSamplingDiscreteFlow`, **shift = 3.0**, and crucially **multiplier = 1.0**: the DiT's timestep input is **σ itself (0–1), not σ×1000**. Feeding ×1000 yields pure noise.
- σ schedule (`simple`): build the 1000-entry table `σ(t)=3t/(1+2t)` for t = j/1000, sample it uniformly from the top, append 0.
- Euler step: `x ← x + (σ_next − σ)·v`; `denoised = x − σ·v` (CONST). CFG applies identically in v-space and denoised-space.
- ER-SDE (CONST path): λ = σ/(1−σ), α = 1−σ, offset σ₀ to σ(1−10⁻⁴) to avoid logit divergence; see `run_pipeline.py` / `index.html` for the full 3-stage solver.
- Text path is **dual-tokenized**: the same prompt goes through the Qwen3 tokenizer (→ encoder hidden states) *and* the T5 tokenizer (→ adapter query ids, vocab 32128). Adapter output is zero-padded to 512 tokens.
- Initial latent: `randn × σ₀` with σ₀ = 1.0; latent format Wan21 (16ch, /8 spatial), pixel sizes must be multiples of 16.

## Pitfalls encountered (read before reimplementing)

1. **fp16 activations NaN on WebGPU.** CPU/DML execution providers silently upcast fp16 ops, so models that "work locally" still NaN in real fp16 shaders. Export the DiT with `--fp32-act` (fp16 I/O, fp32 compute baked as casts) and the VAE in fp32 — the Wan VAE overflows fp16 → the classic black-image bug.
2. **ORT Web returns `Float16Array`.** On browsers with native `Float16Array`, fp16 tensor `.data` contains *floats*, not uint16 bit patterns. Bit-twiddling it corrupts everything (zeros + NaN). Detect and branch (see `toF32`/`makeF16Tensor` in `index.html`).
3. **Naive int4 RTN ruins diffusion DiTs** even with sensitive layers (adaLN/t_embedder/final_layer) excluded. Use int8, or HQQ for int4.
4. **The dynamo ONNX exporter anonymizes names** (`node_linear`, `val_46`) — name-based layer matching is impossible; `quantize_dit.py` identifies sensitive layers by graph topology (reachable from `timesteps` but not `latent`) and weight shapes instead.
5. **`aten._fused_rms_norm` has no ONNX lowering** in recent PyTorch; `F.rms_norm` is monkeypatched to a manual decomposition before export (`common.py`).
6. **Browser single-ArrayBuffer limit (~2 GB)** blocks loading large `.data` files whole — hence sharding, which ORT Web's `externalData` list supports natively.
7. ComfyUI custom ops (`comfy_kitchen.apply_rope_split_half`) must be replaced with pure-PyTorch equivalents pre-trace.

## Performance (reference)

On an RTX 5070 Ti, 1024×1024: Turbo variant (8 steps, CFG 1) ≈ tens of seconds; base (30 steps, CFG 5 → 60 DiT passes) runs minutes. WebGPU currently lacks fused attention, so expect a sizeable gap vs native CUDA; this narrows as ORT Web kernels improve. The first generation at any new resolution is slower (shader specialization), then cached.

## Licenses

- **Code in this repository**: MIT (see `LICENSE`).
- **Anima model & all converted/quantized/merged derivatives**: [CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md). Non-commercial use only; distribution of derivatives requires the attribution notice, a "modified" statement, and no implication of official endorsement — `web/index.html` ships the required notice. Generated images (Outputs) may be used commercially per the license. Also subject to the NVIDIA Open Model License as a Cosmos derivative ("Built on NVIDIA Cosmos").
- **Qwen3-0.6B-Base**: Apache 2.0.

This repository does **not** include model weights. Convert them yourself from the official releases, and keep all license obligations intact when distributing the results.

## Acknowledgements

CircleStone Labs & Comfy Org (Anima), NVIDIA (Cosmos-Predict2), Alibaba (Qwen3, Qwen-Image VAE), the ComfyUI project (reference sampler/scheduler implementations), ONNX Runtime Web and transformers.js teams.
