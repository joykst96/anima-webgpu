"""
End-to-end 검증: export된 ONNX 4종(Qwen3 인코더, LLMAdapter, DiT, VAE 디코더)을
파이썬 onnxruntime으로 이어붙여 실제 이미지를 생성한다.

샘플러: er_sde (ComfyUI sample_er_sde의 CONST/flow 경로 충실 포팅) / euler
스케줄러: simple (ComfyUI simple_scheduler 충실 포팅) / normal
Anima 권장: er_sde + simple + 30스텝 (Turbo 머지본이면 euler + 8~12스텝 + cfg 1)

사용 예 (ComfyUI portable 루트에서):
  python export/run_pipeline.py ^
    --dit out\\dit\\anima_dit_1024.onnx ^
    --adapter out\\adapter\\anima_llm_adapter.onnx ^
    --te out\\text_encoder\\qwen3_06b.onnx ^
    --vae out\\vae\\qwen_image_vae_decoder_1024.onnx ^
    --prompt "masterpiece, best quality, 1girl, silver hair" ^
    --negative "worst quality, low quality" ^
    --steps 30 --cfg 4.0 --seed 42 --out result.png

필요 패키지: onnxruntime(-directml/-gpu), transformers, sentencepiece, pillow, numpy
"""
import argparse
import os
import sys
import time

import numpy as np

# ── ORT가 CUDA DLL을 찾도록 torch lib + nvidia pip 패키지 경로 등록 (portable 대응) ──
try:
    import torch  # noqa: F401
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.name == "nt" and os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
except ImportError:
    pass

if os.name == "nt":
    import glob as _glob
    for _sp in sys.path:
        for _bin in _glob.glob(os.path.join(_sp, "nvidia", "*", "bin")):
            os.add_dll_directory(_bin)

import onnxruntime as ort

SHIFT = 3.0          # supported_models.py Anima sampling_settings 확인값
CONTEXT_LEN = 512
CONTEXT_DIM = 1024


def make_session(path, name):
    priority = ["CUDAExecutionProvider", "DmlExecutionProvider",
                "CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [p for p in priority if p in available]
    sess = ort.InferenceSession(path, providers=providers)
    active = sess.get_providers()[0]
    print(f"[load] {name}: {os.path.basename(path)} ({active})")
    if active == "CPUExecutionProvider" and name == "DiT":
        print("[warn] DiT가 CPU로 떨어짐 — onnxruntime-directml 설치 권장.")
    return sess


# ──────────────────────── 스케줄러 ────────────────────────

def _shift_sigma(t):
    return SHIFT * t / (1.0 + (SHIFT - 1.0) * t)


def simple_schedule(steps: int) -> np.ndarray:
    """ComfyUI simple_scheduler 포팅.
    model_sampling.sigmas 테이블(t=j/1000, j=1..1000의 shifted sigma, 오름차순)에서
    끝(σ=1.0)부터 균등 간격 인덱스로 steps개 추출 후 0.0 추가."""
    table = _shift_sigma(np.arange(1, 1001, dtype=np.float64) / 1000.0)
    ss = len(table) / steps
    sigs = [float(table[-(1 + int(x * ss))]) for x in range(steps)]
    sigs.append(0.0)
    return np.array(sigs, dtype=np.float64)


def normal_schedule(steps: int) -> np.ndarray:
    """선형 t를 shift 변환. sigma[0]=1.0, sigma[-1]=0.0"""
    t = np.linspace(1.0, 0.0, steps + 1)
    return _shift_sigma(t)


# ──────────────────────── 샘플러 ────────────────────────

def sample_euler(denoise_fn, x, sigmas, rng, log):
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
        denoised = denoise_fn(x, sigma)
        if sigma_next == 0:
            x = denoised
        else:
            # x' = x + (σ_next - σ) * (x - denoised)/σ
            d = (x - denoised) / sigma
            x = x + (sigma_next - sigma) * d
        log(i, sigma, sigma_next)
    return x


def sample_er_sde(denoise_fn, x, sigmas, rng, log, s_noise=1.0, max_stage=3):
    """ComfyUI sample_er_sde의 CONST(flow) 모델 경로 포팅.
    half_log_snr = -logit(σ) → er_lambda = σ/(1-σ), alpha = 1-σ.
    원본: comfy/k_diffusion/sampling.py (VP ER-SDE-Solver-3, arXiv 2309.06169)
    """
    sigmas = sigmas.astype(np.float64).copy()
    # offset_first_sigma_for_snr: σ=1.0이면 logit 발산 → percent_to_sigma(1e-4)
    if sigmas[0] >= 1.0:
        sigmas[0] = _shift_sigma(1.0 - 1e-4)

    def noise_scaler(lam):
        return lam * (np.exp(lam ** 0.3) + 10.0)

    er_lambdas = sigmas / (1.0 - sigmas)          # σ_t / α_t  (σ=0 → 0)
    n_int = 200.0
    point_indice = np.arange(0, int(n_int), dtype=np.float64)

    old_denoised = None
    old_denoised_d = None

    for i in range(len(sigmas) - 1):
        denoised = denoise_fn(x, float(sigmas[i]))
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            ls, lt = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / ls               # = 1 - σ_s
            alpha_t = sigmas[i + 1] / lt           # = 1 - σ_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(lt) / noise_scaler(ls)

            # Stage 1 (Euler)
            x = (r_alpha * r) * x + alpha_t * (1.0 - r) * denoised

            if stage_used >= 2:
                dt = lt - ls
                step = -dt / n_int
                lambda_pos = lt + point_indice * step
                scaled_pos = noise_scaler(lambda_pos)

                # Stage 2
                s = np.sum(1.0 / scaled_pos) * step
                denoised_d = (denoised - old_denoised) / (ls - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * noise_scaler(lt)) * denoised_d

                if stage_used >= 3:
                    # Stage 3
                    s_u = np.sum((lambda_pos - ls) / scaled_pos) * step
                    denoised_u = ((denoised_d - old_denoised_d)
                                  / ((ls - er_lambdas[i - 2]) / 2.0))
                    x = x + alpha_t * ((dt ** 2) / 2.0
                                       + s_u * noise_scaler(lt)) * denoised_u
                old_denoised_d = denoised_d

            if s_noise > 0:
                var = lt ** 2 - (ls ** 2) * (r ** 2)
                noise = rng.standard_normal(x.shape).astype(np.float32)
                x = x + alpha_t * noise * s_noise * np.sqrt(max(var, 0.0))

        old_denoised = denoised
        log(i, float(sigmas[i]), float(sigmas[i + 1]))
    return x.astype(np.float32)


# ──────────────────────── 텍스트/메인 ────────────────────────

def encode_prompt(prompt, te_sess, adapter_sess, qwen_tok, t5_tok):
    q = qwen_tok(prompt, return_tensors="np")
    hidden = te_sess.run(None, {
        "input_ids": q["input_ids"].astype(np.int64),
        "attention_mask": q["attention_mask"].astype(np.int64),
    })[0].astype(np.float16)

    t5_ids = t5_tok(prompt, return_tensors="np")["input_ids"].astype(np.int64)
    if t5_ids.shape[1] > CONTEXT_LEN:
        t5_ids = t5_ids[:, :CONTEXT_LEN]

    ctx = adapter_sess.run(None, {
        "qwen_hidden_states": hidden,
        "t5_input_ids": t5_ids,
    })[0]

    out = np.zeros((1, CONTEXT_LEN, CONTEXT_DIM), dtype=np.float16)
    out[:, :ctx.shape[1]] = ctx
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dit", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--te", required=True)
    p.add_argument("--vae", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="worst quality, low quality")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=1024,
                   help="DiT export 해상도와 반드시 일치")
    p.add_argument("--sampler", default="er_sde", choices=["er_sde", "euler"])
    p.add_argument("--scheduler", default="simple", choices=["simple", "normal"])
    p.add_argument("--s-noise", type=float, default=1.0, dest="s_noise")
    p.add_argument("--out", default="result.png")
    p.add_argument("--qwen-tokenizer", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--t5-tokenizer", default="google/t5-v1_1-xxl")
    args = p.parse_args()

    from transformers import AutoTokenizer

    print("[load] 토크나이저 2종 (Qwen3 + T5)")
    qwen_tok = AutoTokenizer.from_pretrained(args.qwen_tokenizer)
    t5_tok = AutoTokenizer.from_pretrained(args.t5_tokenizer)

    te = make_session(args.te, "TE")
    adapter = make_session(args.adapter, "Adapter")
    dit = make_session(args.dit, "DiT")
    vae = make_session(args.vae, "VAE")

    use_cfg = args.cfg > 1.0
    print(f"[prep] 프롬프트 인코딩 (CFG {'on' if use_cfg else 'off'})")
    ctx_pos = encode_prompt(args.prompt, te, adapter, qwen_tok, t5_tok)
    ctx_neg = (encode_prompt(args.negative, te, adapter, qwen_tok, t5_tok)
               if use_cfg else None)

    def denoise_fn(x, sigma):
        """comfy 규약의 denoised 반환. CFG는 denoised 공간과 v 공간이 등가."""
        t = np.array([sigma], dtype=np.float32)  # multiplier=1.0 → t = σ
        v = dit.run(None, {"latent": x.astype(np.float16),
                           "timesteps": t, "context": ctx_pos}
                    )[0].astype(np.float32)
        if use_cfg:
            v_u = dit.run(None, {"latent": x.astype(np.float16),
                                 "timesteps": t, "context": ctx_neg}
                          )[0].astype(np.float32)
            v = v_u + args.cfg * (v - v_u)
        return x - sigma * v                     # CONST: denoised = x - σ·v

    h = w = args.size // 8
    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((1, 16, 1, h, w)).astype(np.float32)  # σ_max = 1.0

    sched = simple_schedule if args.scheduler == "simple" else normal_schedule
    sigmas = sched(args.steps)

    t0 = time.time()

    def log(i, s, sn):
        print(f"  step {i + 1}/{args.steps}  sigma {s:.4f} → {sn:.4f}"
              f"  ({time.time() - t0:.1f}s)")

    print(f"[run] {args.sampler}/{args.scheduler}, {args.steps} steps, "
          f"cfg={args.cfg}, seed={args.seed}")
    sampler = sample_er_sde if args.sampler == "er_sde" else sample_euler
    if args.sampler == "er_sde":
        x = sampler(denoise_fn, x, sigmas, rng, log, s_noise=args.s_noise)
    else:
        x = sampler(denoise_fn, x, sigmas, rng, log)

    print("[run] VAE decode")
    img = vae.run(None, {"latent": x.astype(np.float16)})[0]
    img = ((img[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    from PIL import Image
    Image.fromarray(img).save(args.out)
    print(f"[done] {args.out} 저장 ({time.time() - t0:.1f}s 소요)")


if __name__ == "__main__":
    main()
