# Anima WebGPU — anime image generation that runs in your browser

English | [한국어](README.ko.md)

Runs [Anima](https://huggingface.co/circlestone-labs/Anima) (CircleStone Labs / Comfy Org, 2B DiT) entirely inside the browser.
No server-side inference, nothing to install. After the weights are downloaded once, every generation runs on the visitor's own GPU through WebGPU, and prompts never leave the device.

This repo contains the whole chain: ComfyUI/PyTorch → ONNX export, quantization and graph post-processing, shard packaging, the web front end (plain ES modules, no build step), a static-server config, and the verification tools.

## At a glance

RTX 5070 Ti (16 GB), Chrome 152, ONNX Runtime Web 1.26 JSPI build, warm median:

| Task | Time |
|---|---|
| Turbo 1024² (8 steps, CFG 1) | **12.4 s** (DiT 1.43 s/step + VAE 0.75 s) |
| Turbo 1024², advanced option "Fast" | 11.1–11.5 s |
| Turbo 1024², SPEED | 6.7–6.9 s |
| Turbo 1280² / 1536² | 24.4 s / about 45 s |
| Base 1024² (er_sde 30 steps, CFG 5 = 60 DiT calls) | about 88 s |
| Base + EasyCache / cfgpp_ud10_ab 20 steps | 60–66 s / 56–61 s |
| With a LoRA | +1.5 s per image (7 s to apply) |

- The first visit downloads about 3.6 GB (DiT 2.05 GB + text encoder, adapter, VAE). Later visits load from the OPFS cache.
- A 256² warm-up right after loading builds the shaders, so even the first generation finishes in about 13 s.
- For the same input, one DiT call differs from the ComfyUI (PyTorch fp32) reference by **0.8–1.7% for Base and 1.6–3.0% for Turbo** (relative L2, σ 1.0–0.1). Samplers, schedulers and prompt weighting compute the same thing as ComfyUI (see "Accuracy").

## Features

**Generation**
- Model dropdown: Base (er_sde · 30 steps · CFG 5) and Turbo (Turbo LoRA merged, euler · 8 steps · CFG 1). A deployment can add fine-tunes.
- Resolution: multiples of 16, aspect-ratio presets. One graph handles every resolution (dynamic H/W).
- Samplers: `euler`, `er_sde`, `cfgpp_ud10_ab` (for Base, CFG 2 recommended — 20 steps is 31–34% faster than er_sde at 30). Schedulers: `simple`, `normal`.
- Batch and queue: 1–100 images per click. A `-1` seed is redrawn for every image. Queued jobs can be cancelled one by one.
- **SPEED** (opt-in): starts at low resolution and grows to full resolution mid-schedule via DCT spectrum expansion ([Xiao et al.](https://arxiv.org/abs/2605.18736)). About 45% faster for Turbo at 1024²/1280², with some risk of losing fine detail such as hands. 1024² and 1280² presets only.

**Prompt**
- Weighting follows ComfyUI's rules exactly: `(tag:1.2)` (an explicit number overrides the enclosing weight), `(tag)` = enclosing ×1.1, `\(` `\)` are literal parentheses. Tokenization also follows ComfyUI, one weighted span at a time.
- **NegPip**: at CFG 1 (Turbo) the negative prompt is normally ignored. With NegPip checked, the negative prompt is merged into the positive one with negative weights, which subtracts the concept.
- Tag autocomplete: Danbooru tags plus Korean aliases (200k rows), searched entirely in the browser (Web Worker, IndexedDB cache).

**LoRA**
- Attach several `.safetensors` LoRAs (kohya or `diffusion_model.` format), each with its own strength. They are injected into a rank-48 slot graph without re-exporting.
- Multiple LoRAs are merged exactly via ΔW sum → SVD (in a Web Worker). With a single LoRA the strength slider applies on the next image without recompiling.
- Keys that cannot be used (text-encoder LoRA, LoKr, …) are reported, not silently dropped.

**Advanced options (gear icon)** — all off / standard by default; turning them off gives the same result as before.
- Precision "Fast": the inside of each block in fp16 (norms, residuals and int8 inputs stay fp32). −10% per step; only fine details change.
- Cross-attention padding compaction: replaces the 512-token padding with "one pad token + score bias ln(m)" (mathematically equivalent). −5.6% per step.
- EasyCache: skips DiT calls on steps with little change and extrapolates from the previous result (port of the ComfyUI node). −25–29% for Base, 0–11% for Turbo.

**Results**
- Gallery, lightbox, ZIP download. Each entry shows its generation settings (seed, size, time, options in use).
- Auto-save to a folder (Chromium): writes `{prefix}_00001.png` straight into the chosen folder. Only file names are read to continue the numbering; existing file contents are never read.
- Custom model: pick a folder and the page lists the models it finds inside (subfolders included), whether browser-converter output or build output, and loads the chosen one straight into memory (no OPFS). Chromium remembers the folder for the next visit.

## How it works

| Stage | Model | Format | Notes |
|---|---|---|---|
| Text encoding | Qwen3-0.6B-Base | ONNX fp16, sharded | last hidden state |
| Context adapter | Anima LLMAdapter | ONNX fp16 | dual tokenization (Qwen3 → hidden, T5 → queries); output zero-padded to 512×1024 and multiplied by the T5 token weights |
| Denoiser | Anima DiT 2B (Cosmos-Predict2) | ONNX, int8 weights (MatMulNBits) + int8 activations (DP4A), fp16 I/O | dynamic H/W, RoPE in-graph, attention fused into `MultiHeadAttention` (FlashAttention), NegPip / LoRA-slot / padding-compaction inputs |
| Decode | Qwen-Image (Wan 2.1) VAE decoder | ONNX fp32 | CausalConv3d folded to 2D, split into head (global attention) / tail (local convs) |

- Samplers and schedulers run in JavaScript. Each step calls the DiT once (CFG 1) or twice (CFG > 1).
- When `WebAssembly.Suspending` exists (Chrome 137+), ORT uses the **JSPI build** (native C++ WebGPU EP). The DP4A int8 matmul and FlashAttention kernels exist only there; otherwise it falls back to the JSEP build. Pinned to 1.26.0 (`?ort=x.y.z` loads another version for testing).
- The VAE runs the head once and the tail in horizontal strips (16-latent halo) — bit-identical to a single call, with peak memory bounded by the strip size. Above 1024² the head's attention also runs per query chunk.
- When the DiT is swapped (model change, precision change, applying LoRAs), the text encoder, adapter and VAE are unloaded and reloaded with it, because ORT WebGPU's buffer cache is only freed once every session on the GPU context is released (pitfall 14).

## Accuracy (vs ComfyUI)

Checked directly against ComfyUI v0.43 + anima-base-v1.0 (PyTorch fp32).

| Item | Result |
|---|---|
| Schedulers `simple`/`normal`, samplers `euler`/`er_sde`/`cfgpp_ud10_ab` | ≤2.5e-7 vs a line-by-line port of the originals (float32 rounding) |
| SPEED (DCT, spectrum expansion) | 4–6e-8 vs the original `spd_core` |
| Prompt weighting and tokenization | same spans and weights as ComfyUI `token_weights`, same Qwen token sequence |
| One DiT call (same input) | Base 0.8–1.7%, Turbo 1.6–3.0% |
| Where that error comes from | int8 weights (block 32) are under 1%. Most of it is int8 activations (DP4A); the two timestep-embedding layers that contribute the most are kept in fp32 |

The same seed does **not** give the same image as ComfyUI, because the initial-noise RNG differs (mulberry32 vs `torch.randn`). Fed the same initial noise, the images match.

## Not included — supply your own

This repo ships code only. The following are left out on purpose; the page still runs without the optional ones.

| What | Where it goes | How to fill it in |
|---|---|---|
| Model weights and graphs | `out/dit/`, `out/text_encoder/`, `out/adapter/`, `out/vae/` | Build them from the official releases with the steps under "Converting the models" (license terms apply). |
| Tokenizers (optional) | `tokenizers/{qwen3,t5}/` | `export/prepare_tokenizers.py`. Without them the page loads the tokenizers from the Hugging Face Hub. |
| Tag autocomplete data (optional) | `out/danbooru_ko.csv` | Not distributed; bring your own Danbooru tag list. Format: header `name,category,post_count,description`, then one tag per line; Korean aliases go after `키워드:` inside the quoted description, comma-separated. Without the file, autocomplete is just off. |
| Your model list | `web/js/config.js` | The committed file lists only Anima. Generate it from your own list (`export/model-list.yml`, or a separate deployment list) with `gen_models.py`. |
| Converter map | `web/convert/anima2b.map.json` | The shipped map matches the maintainer's template build, and the converter checks template hashes. Regenerate it from your own build with `export/make_convert_map.py` and serve those graphs as templates. |
| Golden baseline | `tools/golden.json` | Pixel hashes depend on the build, ORT version, GPU and driver. Create your own on the first run (see "Development and verification tools"). |

## Converting the models

### Environment

- A ComfyUI that runs Anima (portable recommended). Verified with v0.43.0 + Python 3.13 + PyTorch 2.14 (cu130).
- Extra packages: `onnx`, `onnxruntime`, `onnxscript` (for portable: `python_embeded\python.exe -m pip install ...`).
- Files (Hugging Face): `anima-base-v1.0.safetensors`, `qwen_3_06b_base.safetensors`, `qwen_image_vae.safetensors` from `circlestone-labs/Anima`, and `anima-turbo-lora-v0.2.safetensors` from `circlestone-labs/Anima-Official-LoRAs`.
- A 16 GB GPU is enough (DiT fp32 export peaks around 10 GB). Quantization uses about 10 GB of RAM.
- On Windows, set `PYTHONIOENCODING=utf-8` when redirecting logs to a file (torch.onnx progress output fails under cp949).

### DiT in one go

From the ComfyUI portable root:

```bat
export\build_dit.bat ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors anima
```

For Base and for Turbo (merging `TURBO_LORA` from the top of the script) it runs six steps:

1. `export_dit.py --dynamic --fp32-act` — dynamic-resolution ONNX (fp32 activations, fp16 I/O)
2. `quantize_dit.py --bits 8 --no-exclude --accuracy-level 4 --fp32-t-embed` — int8 weights + DP4A, with only the two t_embedder layers on fp32 activations
3. `fuse_attention.py` — decomposed SDPA → `MultiHeadAttention` (FlashAttention); writes `.fuse_meta.json`
4. `add_negpip.py` — adds the `negpip_mask` input (output unchanged when mask = 1)
5. `shard_onnx_data.py --shard-mb 400` — splits the weights into 400 MB shards
6. `add_lora_slots.py --rank 48` — LoRA slot graph (shares the shards)

Output: `out/dit/{prefix}_dit_dyn32_q8a4fns.onnx` (plain) / `_q8a4fnLs.onnx` (LoRA slots), and `dit_turbo32` for Turbo.

### DiT post-processing for the page

The page first looks for `<graph>_x.onnx` (padding-compaction inputs) and `<graph>_f16.onnx` ("Fast"), and falls back to the original graph if they are missing. For each plain and slot graph:

```bash
python -I tools/graph/xattn_compact.py <graph>.onnx <graph>_x.onnx
python -I tools/graph/fp16_zones.py <graph>_x.onnx <graph>_f16.onnx --fast
```

Only the graph changes; the original shards are referenced as they are. To apply this to every model already on a server at once, use `tools/graph/make_fast_graphs.py`.

### Text encoder, adapter, VAE

```bash
python export/export_adapter.py --comfyui <ComfyUI> --ckpt <anima-base-v1.0.safetensors> --out out/adapter/anima_llm_adapter.onnx
python export/export_text_encoder.py --out out/text_encoder/qwen3_06b.onnx
python export/shard_onnx_data.py --src out/text_encoder/qwen3_06b.onnx --out out/text_encoder/qwen3_06b_s.onnx --shard-mb 400

python export/export_vae_decoder.py --comfyui <ComfyUI> --vae <qwen_image_vae.safetensors> \
  --out out/vae/qwen_image_vae_decoder_dyn32_2d.onnx --dynamic --size 512
python export/chunk_vae_attention.py --src out/vae/qwen_image_vae_decoder_dyn32_2d.onnx \
  --out out/vae/qwen_image_vae_decoder_dyn32_2dc.onnx --chunks 8
python export/split_vae.py --src out/vae/qwen_image_vae_decoder_dyn32_2dc.onnx   # → _head / _tail / _head_a / _attn / _head_b
```

Export the VAE in fp32. `--size` is only the trace size (the graph is dynamic); tracing the fp32 VAE at 1024 can OOM on 16 GB.

### Tokenizers

```bash
python export/prepare_tokenizers.py --out tokenizers
```

Saves the Qwen3 and T5 tokenizers as `tokenizer.json` for transformers.js. The page looks in `tokenizers/` first and falls back to the Hugging Face Hub.

### Converting in the browser (no ComfyUI)

`web/convert/` is a standalone page that turns an Anima 2B (28-block) `.safetensors` checkpoint into a page-ready DiT (6 graphs + six 400 MB shards + manifests) inside the browser. It saves the result to a folder you pick, as `<name>_base/` (plus `<name>_turbo/` when you add the Turbo LoRA); the generation page then loads that folder through "Custom model". Browsers without the folder-save API get per-file download links instead.

- Every Anima 2B checkpoint has the same graph structure, so the 12 graphs built by the pipeline above serve as **templates** and only the weight bytes are rebuilt (bf16 → fp32 → int8 block quantization exactly as the ORT quantizer does it; Turbo merges the LoRA through the same fp16 path as ComfyUI). For the same checkpoint the output is **byte-identical** to the pipeline's (checked on the base model and two fine-tunes).
- Templates: the 12 base/Turbo graphs under `out/dit/` (`<stem>_q8a4fn{s,Ls}{,_x,_f16}.onnx`) and the two `<stem>_q8a4fnLs.onnx.lora_manifest.json`. The page looks in `../out/dit/`; `?templates=<URL>` overrides it. No shards, text encoder or VAE are needed.
- The map `web/convert/anima2b.map.json` is generated by `export/make_convert_map.py` from a build folder and the source checkpoint (`--prefix` = the build's prefix, which also names the converter's output files; regenerate it whenever you rebuild the graphs, since it checks the templates by hash).
- The adapter (`llm_adapter`) is not converted; the page uses one shared adapter for every model.
- It can live on its own subdomain: `convert/` + `js/convert-core.js` + `workers/convert-*.js` + `css/app.css` + the templates. The generation page links to it through `convert_url` in the model list.

### Registering models with the page

The model list lives in `export/model-list.yml`; `gen_models.py` injects it into `web/js/config.js`, the only file that differs between deployments. Keep your real deployment list separately (for example a git-ignored `export/model-list.deploy.yml`) and generate from it when you deploy.

```bash
python export/gen_models.py export/model-list.yml web/js/config.js
```

```yaml
models:
  - key: myft                 # = build_dit.bat prefix → out/dit/myft_dit_{dyn32|turbo32}_q8a4f{ns,Ls}.onnx
    label: My Model
    variants: [base, turbo]
    source: https://...       # optional: footer source link
    overrides:                # optional, per variant: steps/cfg/sampler or graph paths (dit/ditLora)
      turbo: { steps: 6 }
```

Top-level `defaults:` sets steps/CFG/sampler per variant (also used for custom models), `selected:` the initially selected model, and `convert_url:` the converter page address (empty = `convert/`).

## Deployment

```
index.html  css/  js/  workers/      ← web/ as is (js/config.js is the deployment config)
tokenizers/{qwen3,t5}/
out/
├── dit/           per graph: .onnx + _x + _f16, shards .N.bin + .manifest.json; slot graphs also .lora_manifest.json (the page builds .lora.bin itself)
├── adapter/       anima_llm_adapter.onnx (+ .data)
├── text_encoder/  qwen3_06b_s.onnx + shards + manifest
├── vae/           qwen_image_vae_decoder_dyn32_2dc.onnx + _head/_tail/_head_a/_attn/_head_b (+ .data)
└── danbooru_ko.csv  tag-autocomplete data
```

- `deploy/` has a ready nginx config (`docker compose up -d`). Weights and graphs are served `immutable` for a year; the page and `js/`, `css/`, `workers/` are `no-cache`.
- Because of that, **overwriting a model file under the same name never reaches existing users.** Upload under a new name and change only `config.js`. The page cleans up caches the config no longer references in each user's browser.
- HTTPS is required (WebGPU and OPFS need a secure context; `localhost` is the only exception).
- Behind Cloudflare, add a Cache Rule that caches `.bin/.onnx/.data/.json` (not cached by default). Keep shards under 500 MB because of the free plan's per-file limit.
- Never put the unquantized model in the web root.

## Development and verification tools

- `node tools/devserver.mjs --port 5173 --upstream <weights server> --local-out out-local` — serves `web/` and proxies `/out/` to the weights server (GET only; the `ANIMA_UPSTREAM` environment variable works instead of `--upstream`). If the same path exists under `--local-out`, that file is served instead (for testing graphs before deployment; supports HTTP Range).
- `tools/harness.mjs` — golden harness. Compares pixel SHA-256 of 9 generation cases plus 8 autocomplete queries against `tools/golden.json`. Changes that alter numerics are judged by PSNR against reference images. The baseline is per environment and not in the repo: load the harness in the page console (`const h = await import("/__tools/harness.mjs")`), run `h.full()`, and once it finishes `await h.saveGolden()` writes `tools/golden.json` through the dev server; later runs use `await h.compareToGolden()`.
- `tools/profile.mjs` — per-kernel GPU time via session profiling.
- `tools/graph/` — graph analysis and post-processing scripts (`info.py`, `xattn_compact.py`, `fp16_zones.py`, and experiments such as `batch2.py`, `fuse_matmulnbits.py`, `split_context.py`, each with its results in the file header).
- Browser-converter checks: `tools/convert-check.mjs` (runs the converter core in Node and compares bytes with a build), `tools/convert-folder.mjs` (writes the same result folder as the converter page to disk), `tools/convert-compare.py` (byte-compares that folder with a pipeline build made under another name).

## Inference contract (for porting)

- Flow model, `ModelSamplingDiscreteFlow`, **shift 3.0**, **multiplier 1.0** — the DiT's timestep input is σ itself (0–1). Feeding σ×1000 gives pure noise.
- `denoised = x − σ·v` (CONST). CFG is equivalent in v space and denoised space.
- `simple`: build a 1000-entry table `σ(t) = 3t/(1+2t)` for t = j/1000, sample evenly from the top, append 0. `normal`: split σ_max…σ_min linearly, then apply the shift again (exactly as ComfyUI does).
- er_sde (CONST): λ = σ/(1−σ), α = 1−σ; σ₀ = 1 is moved to shift(1−10⁻⁴).
- Dual tokenization on the text path (Qwen3 → hidden, T5 → adapter queries); adapter output × T5 token weights, then zero-padded to 512.
- Initial latent is `randn` (σ₀ = 1); Wan21 latent (16 channels, spatial /8). Pixel sizes must be multiples of 16.

## Pitfalls (read before reimplementing)

1. **`--fp32-act` is a speed choice.** fp16 activations run without NaNs, but the model's internal autocast inserts 763 fp16↔fp32 Casts and makes it 1–2 s slower. Always export the VAE in fp32 (fp16 is known to overflow into black images, and on ORT 1.26 fp16 convs are actually slower).
2. **ORT Web may return fp16 tensors as `Float16Array` (real values).** Assuming uint16 bit patterns breaks everything (see `toF32` in `js/ort.js`).
3. **int4 RTN breaks this DiT,** even with sensitive layers excluded. int8 RTN (block 32) is effectively lossless at under 1%; blocks of 64 or 128 are worse.
4. **DP4A (`accuracy_level=4`) also quantizes the activations to int8.** That is a separate error from weight quantization, and the two timestep-embedding layers (M = 1) produce most of it. Putting just those two at `accuracy_level=1` (`--fp32-t-embed`) cuts the error to a third to a fifth at no time cost. The CPU EP does not take this path, so judge quality in the browser.
5. **The dynamo exporter anonymizes node names** (`node_linear_3`, `val_46`), so layers must be found by graph structure. Example: the t_embedder is "the linears reachable only from the timestep whose K and N are both ≥ 1024" (every adaLN goes through a rank-256 bottleneck).
6. **ComfyUI custom ops must be replaced with plain PyTorch before export:** `comfy_kitchen`'s `rms_rope_split_half` (v0.43, q/k RMSNorm + RoPE), the older `apply_rope_split_half`, and `F.rms_norm` (`aten._fused_rms_norm` has no ONNX lowering). `export/common.py` handles these.
7. **RoPE compares `max(H, W, T)` as Python ints, which freezes at trace time.** Landscape resolutions then break in a Reshape, so the embedding is replaced with an independent `arange` per axis (`common.py`).
8. **A single browser ArrayBuffer tops out around 2 GB,** hence the sharding. ORT Web's `externalData` takes a list of files.
9. **onnxruntime-web ships two WebGPU builds.** The default (JSEP) lacks the DP4A and FlashAttention kernels; the JSPI build has them.
10. **The VAE's `CausalConv3d` (rank 5) runs on a slow naive kernel.** With T = 1 it equals a 2D conv over the last temporal tap, so it is folded to 4D to take the Conv2d path.
11. **The VAE mid-block attention's score matrix causes high-resolution OOM** (2.4 GB at 1280²). head_dim 384 exceeds the FlashAttention kernel's workgroup limit, so the query axis is chunked instead (softmax is row-wise, so numerically equivalent) to shrink the score buffer.
12. **ORT WebGPU does not reuse freed buffers within a single `run()`.** The VAE's peak grows to the sum of its intermediates, so it is split into head/tail and the tail runs in strips.
13. **The GC does not see VRAM pressure.** ORT tensors are small JS objects wrapping big GPU buffers and are collected late. `dispose()` tensors as soon as they are read (guarded, since some builds lack it).
14. **ORT WebGPU's buffer cache is not freed when one session is `release()`d.** The cache belongs to the GPU context and goes away only when every session on that context is released. Swapping just the DiT grows resident VRAM from 5 to 9.7 GB and pushes the peak to the 16 GB limit, doubling VAE time. The cache mode (`storageBufferCacheMode`) cannot be changed, because ORT JS adds session config entries *after* attaching the execution provider.
15. **WebGPU `Split` uses a storage buffer per output.** With many outputs (e.g. 85) it exceeds the shader limit and fails; use `Slice` instead.
16. **Do not assume a newer ORT is faster.** 1.30 made full generation 5–6% slower for this DiT (its fp16 kernels are better, though). Measure full generations in a fresh page.
17. **Check that it is the same checkpoint.** Different checkpoints of the same family have identical graph structure and sizes. Comparing against the original PyTorch output is the only real check.

## License

- **Code in this repo**: MIT (`LICENSE`).
- **The Anima model and every converted, quantized or merged derivative**: [CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md). Non-commercial use only; distributing derivatives requires the specified notice, a "modified" statement, and not implying official endorsement (`web/index.html` includes the required notice). Generated images may be used commercially under the license. As a Cosmos derivative, the NVIDIA Open Model License also applies ("Built on NVIDIA Cosmos").
- **Qwen3-0.6B-Base**: Apache 2.0.

This repo **does not include** model weights. Convert them yourself from the official releases and keep the license obligations when you distribute the results.

## Acknowledgements

CircleStone Labs & Comfy Org (Anima), NVIDIA (Cosmos-Predict2), Alibaba (Qwen3, Qwen-Image VAE), ComfyUI (sampler, scheduler and EasyCache reference implementations), sorryhyun/ComfyUI-Spectrum-KSampler (SPEED), and the ONNX Runtime Web and transformers.js teams.
