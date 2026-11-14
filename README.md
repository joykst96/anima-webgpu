# Anima WebGPU — anime image generation that runs in your browser

English | [한국어](README.ko.md)

Runs [Anima](https://huggingface.co/circlestone-labs/Anima) (CircleStone Labs / Comfy Org, 2B DiT) entirely inside the browser.
No server-side inference, nothing to install. After the weights are downloaded once, every generation runs on the visitor's own GPU through WebGPU, and prompts never leave the device.

This repo contains the whole chain: ComfyUI/PyTorch → ONNX export, quantization and graph post-processing, shard packaging, the web front end (plain ES modules, no build step), a static-server config, and the verification tools.

## At a glance

On an RTX 5070 Ti (16 GB):

| Task | Time |
|---|---|
| Turbo 1024² (8 steps, CFG 1) | **about 12.5 s** |
| Turbo 1024², advanced option "Fast" | 11.1–11.5 s |
| Turbo 1024², SPEED | 6.7–6.9 s |
| Turbo 1280² / 1536² | 24.4 s / about 45 s |
| Base 1024² (er_sde 30 steps, CFG 5 = 60 DiT calls) | about 88 s |
| Base + EasyCache / cfgpp_ud10_ab 20 steps | 60–66 s / 56–61 s |
| With a LoRA | +1.5 s per image (7 s to apply) |
| (experimental) WebNN, Turbo 1024² | about 3.4 s |
| (experimental) WebNN, Base 1024² 30 steps / + Spectrum | about 24 s / about 15 s |

- Needs a browser with WebGPU (Chrome 137 or later recommended).
- The first visit downloads about 3.6 GB. Later visits load from the browser cache (OPFS).
- Samplers, schedulers and prompt weighting compute the same thing as ComfyUI. For the same input, one DiT call differs from the ComfyUI (PyTorch fp32) reference by 0.8–1.7% for Base and 1.6–3.0% for Turbo ([Accuracy](MAINTENANCE.md#accuracy-vs-comfyui)).

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
- Tag autocomplete: Danbooru tags plus Korean aliases (about 150k tags, from [mengod-hf/danbooru-tags-kr](https://huggingface.co/datasets/mengod-hf/danbooru-tags-kr)), searched entirely in the browser (Web Worker, IndexedDB cache). Search by English prefix (`long_h`, `school u`) or by Korean (`긴 머리`, `교복`).

**LoRA**
- Attach several `.safetensors` LoRAs (kohya or `diffusion_model.` format), each with its own strength. They are injected into a rank-48 slot graph without re-exporting.
- Multiple LoRAs are merged exactly via ΔW sum → SVD (in a Web Worker). With a single LoRA the strength slider applies on the next image without recompiling.
- Keys that cannot be used (text-encoder LoRA, LoKr, …) are reported, not silently dropped.

**Advanced options (gear icon)** — all off / standard by default; turning them off gives the same result as before.
- Precision "Fast": the inside of each block in fp16 (norms, residuals and int8 inputs stay fp32). −10% per step; only fine details change.
- Precision "WebNN (experimental)": runs the DiT and the VAE in fp16 through the browser's WebNN. A 1024² image takes about 3.4 s with Turbo and about 24 s with Base (30 steps) on an RTX 5070 Ti. Needs Chrome's WebNN flag and a model deployed with WebNN graphs; 1024×1024 only and without SPEED (NegPip and LoRA work). The weights are fp16 (default) or int8: int8 reuses the Standard weight files, so nothing more is downloaded, and uses about 1.6 GB less GPU memory but is about 20% slower (Turbo about 4.1 s). Loads as Standard where it is unavailable. The converter page can also write the WebNN files, so it works with models you convert yourself.
- Spectrum (WebNN only): on some steps the transformer blocks are skipped; the feature is forecast from earlier steps and only the output layer runs ([Han et al.](https://arxiv.org/abs/2603.01623)). With Base at 30 steps only 17 steps are computed in full, about 24 s → 15 s, and only fine details change. Works with the euler and er_sde samplers.
- Cross-attention padding compaction: replaces the 512-token padding with "one pad token + score bias ln(m)" (mathematically equivalent). −5.6% per step.
- EasyCache: skips DiT calls on steps with little change and extrapolates from the previous result (port of the ComfyUI node). −25–29% for Base, 0–11% for Turbo.
- Low-VRAM mode (for 8 GB-class GPUs): unloads the models from GPU memory right before the VAE decode and loads them back once the image is out. Without it, on a GPU that is short of memory the VAE's leftover buffers push the DiT out to system memory and every image after the first gets slower. With 8 GB simulated on a 16 GB card, Turbo 1024² went from 35–38 s to about 19 s per image (the image is identical). With enough memory it only adds the reload, about 4–6 s per image. Standard and Fast only (not WebNN).

**Results**
- Gallery, lightbox, ZIP download. Each entry shows its generation settings (seed, size, time, options in use).
- Auto-save to a folder (Chromium): writes `{prefix}_00001.png` straight into the chosen folder. Only file names are read to continue the numbering; existing file contents are never read.
- Custom model: pick a folder and the page lists the models it finds inside (subfolders included), whether browser-converter output or build output, and loads the chosen one straight into memory (no OPFS). Chromium remembers the folder for the next visit.

## Documentation

This repo contains code only, no model weights. To convert and host the models yourself, see:

- [MAINTENANCE.md](MAINTENANCE.md) — how it works, accuracy checks, model conversion, deployment, development and verification tools, inference contract, pitfalls.
- Per-folder notes (Korean): [export/webnn/](export/webnn/README.md) (building WebNN graphs), [tools/graph/](tools/graph/README.md) (graph analysis and experiments), [tools/webnn/](tools/webnn/README.md) (WebNN measurement).

## License

- **Code in this repo**: MIT (`LICENSE`).
- **The Anima model and every converted, quantized or merged derivative**: [CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md). Non-commercial use only; distributing derivatives requires the specified notice, a "modified" statement, and not implying official endorsement (`web/index.html` includes the required notice). Generated images may be used commercially under the license. As a Cosmos derivative, the NVIDIA Open Model License also applies ("Built on NVIDIA Cosmos").
- **Qwen3-0.6B-Base**: Apache 2.0.
- **Tag autocomplete data** ([mengod-hf/danbooru-tags-kr](https://huggingface.co/datasets/mengod-hf/danbooru-tags-kr)): the Korean translations, classifications and keywords are CC BY-SA 4.0 per the dataset's [LICENSE](https://huggingface.co/datasets/mengod-hf/danbooru-tags-kr/blob/main/LICENSE) (the dataset card lists MIT); third-party material inside keeps its own terms. The page serves a modified copy: `tools/convert-tags-kr.py` only rewrites the keyword marker (`<키워드>` → `키워드:`) and drops the trailing period. The CSV is not in this repo.

This repo **does not include** model weights. Convert them yourself from the official releases and keep the license obligations when you distribute the results.

## Acknowledgements

CircleStone Labs & Comfy Org (Anima), NVIDIA (Cosmos-Predict2), Alibaba (Qwen3, Qwen-Image VAE), ComfyUI (sampler, scheduler and EasyCache reference implementations), sorryhyun/ComfyUI-Spectrum-KSampler (SPEED), [mengod-hf/danbooru-tags-kr](https://huggingface.co/datasets/mengod-hf/danbooru-tags-kr) (Korean Danbooru tags for autocomplete), and the ONNX Runtime Web and transformers.js teams.
