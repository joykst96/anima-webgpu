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
- Two model variants in one page: base (er_sde · 30 steps · CFG 5) and Turbo-LoRA-merged (euler · 8 steps · CFG 1), plus a **Custom** option to load your own ONNX DiT directly from disk (mobile-friendly, no OPFS needed).
- **Native WebGPU EP (JSPI build)** with DP4A int8 matmul (`accuracy_level=4`) — roughly 2× faster sampling than the JSEP build, with automatic fallback on older browsers.
- **FlashAttention fusion (DiT)**: decomposed SDPA is fused into `com.microsoft.MultiHeadAttention` to trigger the JSPI EP's FA2 kernel. The attention score matrix is no longer materialized whole (O(S²)→O(S)), so resident VRAM drops by about half and above-1024 resolutions clear the DiT stage that previously OOM'd.
- **VAE chunked attention**: the VAE decoder's mid-block self-attention is split along the query-token axis into 8 chunks, capping the score buffer at 1/8 (numerically identical). This fixes the OOM where the VAE hit the 2 GB single-buffer limit at high resolutions — **generation up to 1536² in the browser**.
- **Runtime LoRA**: drop in any Anima `.safetensors` LoRA (kohya or `diffusion_model.` formats) with per-LoRA strength; multiple LoRAs are merged exactly via ΔW + SVD in a Web Worker (UI never blocks). No re-export.
- **Prompt weighting** `(tag:1.2)` and **NegPip** (negative prompts under CFG=1 / Turbo), both faithful to ComfyUI.
- **SPEED (spectral progressive sampling)**: optionally start denoising at half resolution and expand to full resolution mid-schedule via a DCT spectrum extension (after [Xiao et al.](https://arxiv.org/abs/2605.18736)). On Turbo this cuts generation time by ~40-45% at 1024²/1280² with a minor, seed-dependent detail trade-off. Opt-in toggle; the dynamic-resolution DiT graph handles the mid-run resolution switch with no re-export.

## How it works

| Stage | Model | Format | Notes |
|---|---|---|---|
| Text encoding | Qwen3-0.6B-Base | ONNX fp16 | last hidden states |
| Context adapter | Anima LLMAdapter | ONNX fp16 | dual tokenization: Qwen3 ids → hidden states, T5 ids → adapter queries; output zero-padded to 512×1024 |
| Denoiser | Anima DiT 2B (Cosmos-Predict2) | ONNX, int8 weights (`accuracy_level=4` for DP4A) / fp32 activations (recommended — see Pitfalls), fp16 I/O | dynamic H/W axes, RoPE in-graph; self/cross attention fused into `MultiHeadAttention` (FA2); optional NegPip + LoRA-slot inputs |
| Decode | Qwen-Image (Wan 2.1) VAE decoder | ONNX fp32 | CausalConv3d folded to 2D for T=1 (Conv2d kernel path); Wan21 de-normalization baked in; mid-block attention split into 8 query-axis chunks to cap the score buffer |

The sampler (Euler / ER-SDE-Solver) and σ schedule run in plain JavaScript; each step calls the DiT session once (CFG 1) or twice (CFG > 1).

## Requirements

**Conversion machine** — a working ComfyUI install that can already run Anima, Python 3.10+, recent PyTorch (the new dynamo ONNX exporter is used), `onnx`, `onnxruntime`, `transformers`, `sentencepiece`. A 16 GB GPU is comfortable; fp32 export of the DiT peaks around 10 GB VRAM. ComfyUI **portable** users: replace `python` below with `python_embeded\python.exe` and run from the portable root.

**Serving** — any static file server. HTTPS is mandatory in production (WebGPU and OPFS require a secure context; `localhost` is exempt for testing).

**Client** — a WebGPU browser (desktop Chrome/Edge), GPU with `shader-f16` support, ~8 GB VRAM recommended. First visit downloads ~2.5 GB (then cached in OPFS).

## 1. Convert the models

Model files needed: `anima-base-v1.0.safetensors`, the official **Anima Turbo LoRA** (optional, for the fast variant), `qwen_image_vae.safetensors`. Qwen3-0.6B-Base is fetched from HF automatically.

```bash
# DiT — base variant (--fp32-act recommended: fp16 runs but is ~1-2s slower; see Pitfalls)
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
  --out out/vae/qwen_image_vae_decoder_dyn32_2d.onnx --dynamic --size 1024
# CausalConv3d is folded to an equivalent 2D conv (T=1); --keep-3d disables this. --verify checks equivalence.
```

Notes: `--size 512` on the VAE only sets the trace size (the graph is dynamic); fp32 VAE traced at 1024 can OOM a 16 GB card. `export_dit.py --verify` additionally runs an ONNX-vs-PyTorch comparison at two resolutions.

**VAE attention chunk (high-resolution OOM fix)** — the Wan VAE decoder has a full-latent-resolution self-attention in its mid block, so the score matrix `(1,1,S,S)` reaches ~2.4 GB at 1280² (S=25600), past the browser's 2 GB single-buffer limit → OOM. `chunk_vae_attention.py` unrolls that SDPA into N query-token chunks (default 8) so each chunk builds only a partial score (softmax is row-wise, so this is **numerically identical**). K and V are shared. At 1536² (S=36864) each chunk's score is ~0.68 GB and clears.

```bash
python export/chunk_vae_attention.py \
  --src out/vae/qwen_image_vae_decoder_dyn32_2d.onnx \
  --out out/vae/qwen_image_vae_decoder_dyn32_2dc.onnx --chunks 8
```

> Note: fusing the VAE attention into `MultiHeadAttention` is a dead end — head_dim 384 exceeds the WebGPU FA2 kernel's workgroup-storage limit (32 KB; it needs 48 KB) and the shader fails to compile. Splitting the single head into multiple heads changes the math, so the score chunk is the correct route — plain MatMul+Softmax, no head_dim constraint.

## 2. Quantize, add features & shard the DiT

The full DiT pipeline is: **quantize → FlashAttention fusion → NegPip → shard → LoRA slots**. Fusion runs right after quantization (before NegPip rewires v_proj); NegPip must precede sharding (it edits the graph); LoRA slots are added *after* sharding (graph-only, reuses the same shards). Because fusion absorbs the attention Softmax anchor, NegPip and LoRA slots identify attention from the `.fuse_meta.json` (q/k/v/o proj node names) that fusion emits (`--fuse-meta`), falling back to the old Softmax anchor when no meta is present.

```bash
# int8 weight-only + accuracy_level=4 (enables the DP4A integer kernel → ~2× sampling)
python export/quantize_dit.py --src out/dit/anima_dit_dyn32.onnx \
  --out out/dit/anima_dit_dyn32_q8a4.onnx --bits 8 --no-exclude --accuracy-level 4

# FlashAttention fusion: decomposed SDPA → com.microsoft.MultiHeadAttention. Emits .fuse_meta.json.
python export/fuse_attention.py --src out/dit/anima_dit_dyn32_q8a4.onnx \
  --out out/dit/anima_dit_dyn32_q8a4f.onnx

# NegPip: adds a negpip_mask input, rewires cross-attn v_proj (mask=1 ⇒ identical output).
#   Uses fuse_meta if present beside the source, else falls back to the Softmax anchor.
python export/add_negpip.py --src out/dit/anima_dit_dyn32_q8a4f.onnx \
  --out out/dit/anima_dit_dyn32_q8a4fn.onnx

# shard weights into CDN/browser-friendly files
python export/shard_onnx_data.py --src out/dit/anima_dit_dyn32_q8a4fn.onnx \
  --out out/dit/anima_dit_dyn32_q8a4fns.onnx --shard-mb 400

# LoRA slots: low-rank side-branches (rank 48) for runtime LoRA. Output reuses the shards above.
#   Sharding changes the filename, so pass the fuse_meta path explicitly.
python export/add_lora_slots.py --src out/dit/anima_dit_dyn32_q8a4fns.onnx \
  --out out/dit/anima_dit_dyn32_q8a4fnLs.onnx --rank 48 \
  --fuse-meta out/dit/anima_dit_dyn32_q8a4f.onnx.fuse_meta.json
```

Keep **both** the non-slot (`q8a4fns`) and slot (`q8a4fnLs`) graphs on the server: the page serves the faster non-slot graph when no LoRA is active and only switches to the slot graph (same shards, no extra download) when a LoRA is applied. Repeat the whole chain for the Turbo export.

**Batch automation** — running the six steps × two variants by hand for every checkpoint gets tedious. `export/build_dit.bat` (Windows, ComfyUI portable) runs the full `export_dit → quantize → fuse → negpip → shard → lora-slots` chain for both base and Turbo in one go:

```bat
REM from the ComfyUI portable root:
export\build_dit.bat ComfyUI\models\diffusion_models\my-finetune.safetensors myft
```

This emits `myft_dit_dyn32_q8a4fns/fnLs.onnx` and `myft_dit_turbo32_*` (with shards + manifests). The Turbo variant merges the Turbo LoRA at export time (`TURBO_LORA` path is set at the top of the script). Re-run it per checkpoint whenever you rebuild; register the output in the `MODELS` map in `index.html`.

### Adding models to the page

Once the DiT files are in `out/dit/`, register the model in **two places** in `index.html`. The two must use the **same key**, or the model fails to load.

**1. The `MODELS` map** (search `const MODELS`) — add a base/Turbo pair:

```js
mymodel: {
  label: "My Model",
  dit:     "out/dit/mymodel_dit_dyn32_q8a4fns.onnx",
  ditLora: "out/dit/mymodel_dit_dyn32_q8a4fnLs.onnx",
  steps: 30, cfg: 5.0, sampler: "er_sde", scheduler: "simple",
},
mymodel_turbo: {
  label: "My Model+Turbo",
  dit:     "out/dit/mymodel_dit_turbo32_q8a4fns.onnx",
  ditLora: "out/dit/mymodel_dit_turbo32_q8a4fnLs.onnx",
  steps: 8, cfg: 1.0, sampler: "euler", scheduler: "simple",
},
```

Keys are free-form (`[A-Za-z0-9_]`); `dit`/`ditLora` paths must match the actual filenames. Base uses 30 steps / CFG 5 / er_sde; Turbo uses 8 steps / CFG 1 / euler.

**2. The model dropdown** (search `id="modelSel"`) — add one `<option>` per key:

```html
<option value="mymodel">My Model (base · 30 steps · CFG 5)</option>
<option value="mymodel_turbo">My Model+Turbo (8 steps · CFG 1)</option>
```

The `value` must equal the `MODELS` key exactly.

Quantization alone (legacy / minimal):

```bash
python export/quantize_dit.py --src out/dit/anima_dit_dyn32.onnx \
  --out out/dit/anima_dit_dyn32_q8.onnx --bits 8 --no-exclude
python export/shard_onnx_data.py --src out/dit/anima_dit_dyn32_q8.onnx \
  --out out/dit/anima_dit_dyn32_q8s.onnx --shard-mb 480
``` Also shard anything else whose `.data` exceeds your CDN's cache limit (the fp16 text encoder is ~1.2 GB):

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
├── dit/           per variant: *_q8a4fns.onnx (non-slot) and *_q8a4fnLs.onnx (LoRA slots),
│                 each with .bin shards + .onnx.manifest.json; slot graphs also have
│                 .onnx.lora_manifest.json. Shards are shared between the two graphs.
├── adapter/       anima_llm_adapter.onnx (+ .data)
├── text_encoder/  qwen3_06b_s.onnx + shards + manifest
└── vae/           qwen_image_vae_decoder_dyn32_2dc.onnx (+ .data)
```

Edit the `PATHS` / `MODELS` constants at the top of `web/index.html` if your filenames differ. Do **not** leave un-quantized source models in the web root.

A ready-to-run static server is in `deploy/` (`docker compose up -d`; nginx with `immutable` cache headers for weights, `no-cache` for the page). Behind Cloudflare, add a Cache Rule making `.bin/.onnx/.data/.json` cache-eligible — these extensions are not cached by default, and without the rule every request hits your origin. Keep shards ≤ ~500 MB for free-tier per-file cache limits. Because weights are served `immutable`, **never overwrite a model file in place** — upload under a new filename and update `index.html`.

## Browser features

- **Prompt weighting** — `(tag:1.2)` scales a tag, `(tag)` = ×1.1, nesting multiplies, `\(` `\)` are literal parens (for Danbooru tags). Applied at the adapter output (token-wise), matching ComfyUI.
- **NegPip** — under CFG=1 (Turbo), negative prompts are normally ignored. Check the NegPip box and the negative prompt is folded in as a negative-weight group; the slot model's `negpip_mask` flips the sign of those tokens' cross-attn values so the concept is subtracted. Requires a DiT built with `add_negpip.py`.
- **Runtime LoRA** — add one or more `.safetensors` LoRAs with independent strengths, then press **Apply LoRA** (deferred so N LoRAs compile once, not N times). Conversion (safetensors parse + multi-LoRA ΔW/SVD merge) runs in a Web Worker with a progress bar; generation is disabled until it finishes. With a **single** LoRA the strength is a live graph input (`lora_scale`), so the slider takes effect on the next generation with no recompile; with **multiple** LoRAs the strength is baked into the SVD merge, so changing it re-runs the merge. Either way the graph stays rank-48. Unsupported keys (text-encoder LoRAs, LoKr) and out-of-slot modules are reported, not silently dropped. Requires a slot model from `add_lora_slots.py`.
- **Custom model** — the *Custom* dropdown loads a DiT (`.onnx` + shards + manifests) you pick from disk, straight into memory (no OPFS, works on mobile). TE/adapter/VAE use the default paths.
- **SPEED** — opt-in spectral progressive sampling (arXiv:2605.18736), available on the 1024² and 1280² preset buckets. The latent starts at a lower resolution (default `stages = 0.5, 1.0` = half) and, when the noise level drops to a transition σ (default `0.7`), a DCT spectrum extension grows it to full resolution: the low-frequency block is preserved, the new high-frequency band is filled with σ-scaled noise, and the result is rescaled by κ = r/(1+(r-1)σ) with the schedule re-spaced to the aligned σ. Because the early steps run on a quarter-area latent, Turbo generation drops ~40-45% (e.g. 1024² 12.9s → 6.8s, 1280² ~28s → ~17s). Lower σ is faster but risks fine-detail loss (hands); 0.7 is a reasonable balance, though the best value is seed-dependent — the advanced panel exposes `stages` and `transition σ` for tuning. The DCT/IDCT and spectrum-extension math is a verbatim port of [`ComfyUI-Spectrum-KSampler`](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler)'s `spd.py` (with [`ComfyUI-SPEED`](https://github.com/ruwwww/ComfyUI-SPEED) as a cross-reference), validated against a numpy reference to 1e-15.

## Implementation notes (for porting elsewhere)

Hard-won facts about Anima's inference contract, verified against ComfyUI source:

- Flow model, `ModelSamplingDiscreteFlow`, **shift = 3.0**, and crucially **multiplier = 1.0**: the DiT's timestep input is **σ itself (0–1), not σ×1000**. Feeding ×1000 yields pure noise.
- σ schedule (`simple`): build the 1000-entry table `σ(t)=3t/(1+2t)` for t = j/1000, sample it uniformly from the top, append 0.
- Euler step: `x ← x + (σ_next − σ)·v`; `denoised = x − σ·v` (CONST). CFG applies identically in v-space and denoised-space.
- ER-SDE (CONST path): λ = σ/(1−σ), α = 1−σ, offset σ₀ to σ(1−10⁻⁴) to avoid logit divergence; see `run_pipeline.py` / `index.html` for the full 3-stage solver.
- Text path is **dual-tokenized**: the same prompt goes through the Qwen3 tokenizer (→ encoder hidden states) *and* the T5 tokenizer (→ adapter query ids, vocab 32128). Adapter output is zero-padded to 512 tokens.
- Initial latent: `randn × σ₀` with σ₀ = 1.0; latent format Wan21 (16ch, /8 spatial), pixel sizes must be multiples of 16.

## Pitfalls encountered (read before reimplementing)

1. **DiT activation dtype — `--fp32-act` is a *speed* choice, not a NaN fix (corrected 2026-06).** Earlier this was believed to be "fp16 NaNs on WebGPU, so fp32-act is mandatory." That was wrong: exporting **without** `--fp32-act` (i.e. fp16 activations) runs fine with no NaN. But the model's internal autocast (block forward fp32 / submodules fp16) plus the `if x.dtype==fp16: x=x.float()` branch then bake a flood of fp16↔fp32 round-trip Casts into the graph (measured: 6 → 763 Cast nodes), which makes inference ~1-2s *slower* (12s → 14s) despite identical int8 weights and DP4A path. So `--fp32-act` is the recommended default for speed. Removing those Casts safely would require graph surgery across the whole network (high risk, uncertain memory payoff) — not currently worth it. **Separately**, export the VAE in fp32: the Wan VAE genuinely overflows fp16 → the classic black-image bug. (Also note: `add_lora_slots.py` hardcodes fp32 LoRA branches, so it is incompatible with an fp16-act DiT as-is — see the warning in that file.)
2. **ORT Web returns `Float16Array`.** On browsers with native `Float16Array`, fp16 tensor `.data` contains *floats*, not uint16 bit patterns. Bit-twiddling it corrupts everything (zeros + NaN). Detect and branch (see `toF32`/`makeF16Tensor` in `index.html`).
3. **Naive int4 RTN ruins diffusion DiTs** even with sensitive layers (adaLN/t_embedder/final_layer) excluded. Use int8, or HQQ for int4.
4. **The dynamo ONNX exporter anonymizes names** (`node_linear`, `val_46`) — name-based layer matching is impossible; `quantize_dit.py` identifies sensitive layers by graph topology (reachable from `timesteps` but not `latent`) and weight shapes instead.
5. **`aten._fused_rms_norm` has no ONNX lowering** in recent PyTorch; `F.rms_norm` is monkeypatched to a manual decomposition before export (`common.py`).
6. **Browser single-ArrayBuffer limit (~2 GB)** blocks loading large `.data` files whole — hence sharding, which ORT Web's `externalData` list supports natively.
7. ComfyUI custom ops (`comfy_kitchen.apply_rope_split_half`) must be replaced with pure-PyTorch equivalents pre-trace.
8. **onnxruntime-web ships two WebGPU builds.** The default (`ort.webgpu.*`, JSEP) has **no** DP4A or FlashAttention kernels; the `ort.jspi.*` build is the native C++ WebGPU EP that does. This page loads the JSPI build (pinned `1.26.0`) when `WebAssembly.Suspending` exists (Chrome 137+), else falls back. `subgroup-matrix` paths are compiled out of the wasm build entirely — don't chase them.
9. **DP4A needs `accuracy_level=4`** as a node attribute (set by `quantize_dit.py --accuracy-level 4`), plus `K%128==0 && N%16==0 && block_size%32==0`. It dynamically int8-quantizes activations — a *different* accuracy trade-off from weight quantization, so verify image quality in the browser (the CPU EP used by `run_pipeline.py` never takes this path).
10. **VAE `CausalConv3d` is fp32-only and rank-5** — ORT runs it on a slow naive conv3d kernel. For single images (T=1) it's mathematically identical to a 2D conv on the last time-tap; `export_vae_decoder.py` materializes 4D weights so the Conv2d kernel runs instead (folds the per-step decode to near-instant). Not valid for video (T>1).
11. **RoPE uses `max(H, W, T)`** internally, a Python `int` comparison that the dynamo exporter freezes at trace time — so the traced H/W ordering bakes in and **landscape (W>H) resolutions fail with a Reshape error** while square/portrait pass. A non-square trace does *not* fix it (dynamo still won't split the symbols); `patch_comfy_for_export` in `common.py` monkeypatches the RoPE embedding to use independent per-axis `arange` (numerically identical). `export_dit.py --verify` now checks landscape cases.
12. **The VAE decoder's mid-block attention is the high-resolution OOM culprit.** Its score matrix `(1,1,S,S)` is a single buffer at full latent resolution (S=16384 at 1024², 25600 at 1280²), hitting ~2.4 GB at 1280² — past the 2 GB limit. `MultiHeadAttention` fusion fails to compile because head_dim 384 exceeds the WebGPU FA2 workgroup limit (32 KB), and splitting the single head changes the math. The fix is a query-token chunk (`chunk_vae_attention.py`) — numerically identical since softmax is row-wise, while cutting the score buffer to 1/N. The page only generates up to 1536² and uses N=8.

## Performance (reference)

On an RTX 5070 Ti, 1024×1024, with the JSPI build + DP4A + 2D VAE: **Turbo ≈ 11 s** (8 steps, CFG 1), **base ≈ 70 s** (30 steps, CFG 5 → 60 DiT passes) — roughly 3× slower than local ComfyUI (down from ~10×). DP4A roughly halves sampling time; folding the VAE to 2D made decode effectively instant. Applying a LoRA adds ~9 s from the extra slot dispatches. The first generation at any new resolution is slower (shader specialization), then cached.

**The measured value of DiT FlashAttention fusion was memory, not speed.** Within the current resolution range total time is roughly flat (±a few seconds), but no longer materializing the score matrix halves resident DiT VRAM and lets above-1024 resolutions clear the DiT stage that OOM'd when unfused. With **VAE chunked attention** on top, the last bottleneck — the VAE's single-buffer score OOM at high resolution — is gone too: on an RTX 5070 Ti, **1280² ≈ 29 s and 1536² ≈ 58 s**.

> Note: re-compiling the DiT repeatedly (e.g. toggling LoRAs many times) can leak GPU memory in current ORT Web until a page refresh — a known upstream issue. The page minimizes recompiles (deferred LoRA apply, single-resident sessions) to avoid it in normal use.

## Licenses

- **Code in this repository**: MIT (see `LICENSE`).
- **Anima model & all converted/quantized/merged derivatives**: [CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md). Non-commercial use only; distribution of derivatives requires the attribution notice, a "modified" statement, and no implication of official endorsement — `web/index.html` ships the required notice. Generated images (Outputs) may be used commercially per the license. Also subject to the NVIDIA Open Model License as a Cosmos derivative ("Built on NVIDIA Cosmos").
- **Qwen3-0.6B-Base**: Apache 2.0.

This repository does **not** include model weights. Convert them yourself from the official releases, and keep all license obligations intact when distributing the results.

## Acknowledgements

CircleStone Labs & Comfy Org (Anima), NVIDIA (Cosmos-Predict2), Alibaba (Qwen3, Qwen-Image VAE), the ComfyUI project (reference sampler/scheduler implementations), ONNX Runtime Web and transformers.js teams.
