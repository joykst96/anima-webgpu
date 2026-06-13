"""
공통 유틸: ComfyUI 경로 등록, ONNX 비호환 커스텀 op 치환, 검증 헬퍼.

모든 export 스크립트가 가장 먼저 setup_comfyui()를 호출해야 함.
"""
import argparse
import os
import sys

import torch


def setup_comfyui(comfyui_path: str):
    """ComfyUI 루트를 sys.path에 등록. comfy 패키지를 import 가능하게 만든다."""
    comfyui_path = os.path.abspath(os.path.expanduser(comfyui_path))
    if not os.path.isdir(os.path.join(comfyui_path, "comfy")):
        raise FileNotFoundError(
            f"'{comfyui_path}' 안에 comfy/ 디렉토리가 없음. --comfyui 경로 확인 필요."
        )
    sys.path.insert(0, comfyui_path)


def _apply_rope_split_half_torch(xq: torch.Tensor, xk: torch.Tensor,
                                 freqs_cis: torch.Tensor):
    """
    comfy_kitchen.apply_rope_split_half의 순수 PyTorch 등가 구현.
    comfy_kitchen 소스의 docstring에 명시된 레퍼런스 수식 그대로:

        t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
        t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
        t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)

    freqs_cis shape: (..., head_dim//2, 2, 2)  — cos/sin 실수 표현 (복소수 없음)
    """
    def _rope_one(t: torch.Tensor) -> torch.Tensor:
        t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
        t_ = t_.to(freqs_cis.dtype)
        t_out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]
        return t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)

    return _rope_one(xq), _rope_one(xk)


def _rms_norm_manual(input, normalized_shape, weight=None, eps=None):
    """
    F.rms_norm의 수동 분해 구현.
    신버전 torch에서 F.rms_norm이 aten._fused_rms_norm으로 내려가는데,
    ONNX exporter에 이 op의 변환이 없어서 export가 실패한다.
    기본 연산(pow/mean/rsqrt)으로 풀어주면 그대로 변환됨.
    공식 decomposition과 동일하게 fp32로 승격해 계산 후 원래 dtype으로 복귀.
    """
    if eps is None:
        eps = torch.finfo(input.dtype).eps
    dims = tuple(range(input.dim() - len(normalized_shape), input.dim()))
    x = input.float()
    out = x * torch.rsqrt(x.pow(2).mean(dim=dims, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(input.dtype)


def _rope_generate_embeddings_dynamic(self, B_T_H_W_C, fps=None,
                                      h_ntk_factor=None, w_ntk_factor=None,
                                      t_ntk_factor=None, device=None, dtype=None):
    """VideoRopePosition3DEmb.generate_embeddings의 동적 H/W 안전 버전.

    원본은 `seq = arange(max(H, W, T))` 후 seq[:H]/seq[:W]/seq[:T]로 슬라이싱하는데,
    max(H,W,T)가 파이썬 정수 비교라 dynamo trace 시 상수로 박힌다. 정사각형이든
    비정사각형이든 trace 한 번의 H/W 관계(특히 H≤W 여부)가 그래프에 굳어,
    landscape(W>H) 해상도에서 Reshape가 깨진다('{52}→{76,1}').
    H/W/T 각각 독립 arange를 써서 max 비교와 공유 슬라이싱을 제거 → 세 축이
    완전히 독립적인 동적 길이가 된다. 수치는 원본과 동일.
    """
    import torch
    from einops import repeat, rearrange

    h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
    w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
    t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

    h_theta = 10000.0 * h_ntk_factor
    w_theta = 10000.0 * w_ntk_factor
    t_theta = 10000.0 * t_ntk_factor

    h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.to(device=device))
    w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.to(device=device))
    temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.to(device=device))

    B, T, H, W, _ = B_T_H_W_C
    # 핵심 차이: max(H,W,T) + 슬라이싱 대신 축별 독립 arange
    seq_h = torch.arange(H, dtype=torch.float, device=device)
    seq_w = torch.arange(W, dtype=torch.float, device=device)
    seq_t = torch.arange(T, dtype=torch.float, device=device)

    uniform_fps = (fps is None) or isinstance(fps, (int, float)) or (fps.min() == fps.max())
    assert uniform_fps or B == 1 or T == 1, \
        "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"

    half_emb_h = torch.outer(seq_h, h_spatial_freqs)
    half_emb_w = torch.outer(seq_w, w_spatial_freqs)
    if fps is None or self.enable_fps_modulation is False:
        half_emb_t = torch.outer(seq_t, temporal_freqs)
    else:
        half_emb_t = torch.outer(seq_t / fps * self.base_fps, temporal_freqs)

    half_emb_h = torch.stack([torch.cos(half_emb_h), -torch.sin(half_emb_h), torch.sin(half_emb_h), torch.cos(half_emb_h)], dim=-1)
    half_emb_w = torch.stack([torch.cos(half_emb_w), -torch.sin(half_emb_w), torch.sin(half_emb_w), torch.cos(half_emb_w)], dim=-1)
    half_emb_t = torch.stack([torch.cos(half_emb_t), -torch.sin(half_emb_t), torch.sin(half_emb_t), torch.cos(half_emb_t)], dim=-1)

    em_T_H_W_D = torch.cat([
        repeat(half_emb_t, "t d x -> t h w d x", h=H, w=W),
        repeat(half_emb_h, "h d x -> t h w d x", t=T, w=W),
        repeat(half_emb_w, "w d x -> t h w d x", t=T, h=H),
    ], dim=-2)

    return rearrange(em_T_H_W_D, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()


def patch_comfy_for_export():
    """
    ONNX export를 막는 ComfyUI 런타임 요소들을 치환.
    1) comfy.quant_ops.ck.apply_rope_split_half (torch.ops 커스텀 op) → 순수 PyTorch
    2) F.rms_norm (aten._fused_rms_norm) → 수동 분해 구현
    3) VideoRopePosition3DEmb.generate_embeddings → 동적 H/W 안전 버전
       (max(H,W,T) 상수화로 landscape 해상도가 깨지는 문제 수정)
    반드시 모델 로드 전에 호출할 것.
    """
    import comfy.quant_ops as qo

    if getattr(qo, "ck", None) is not None:
        qo.ck.apply_rope_split_half = _apply_rope_split_half_torch
    else:
        # comfy_kitchen 미설치 환경: 더미 네임스페이스를 만들어 끼움
        class _CKShim:
            apply_rope_split_half = staticmethod(_apply_rope_split_half_torch)
        qo.ck = _CKShim()
    print("[patch] apply_rope_split_half → pure PyTorch 치환 완료")

    # F.rms_norm 치환 (nn.RMSNorm.forward도 내부적으로 이걸 호출하므로 함께 커버됨)
    torch.nn.functional.rms_norm = _rms_norm_manual
    # ComfyUI가 자체 모듈에서 rms_norm을 직접 바인딩해 둔 경우까지 커버
    try:
        import comfy.rmsnorm
        if hasattr(comfy.rmsnorm, "rms_norm"):
            comfy.rmsnorm.rms_norm = _rms_norm_manual
    except ImportError:
        pass
    print("[patch] F.rms_norm → 수동 분해 구현 치환 완료")

    # RoPE 임베딩: max(H,W,T) 상수화 → landscape Reshape 실패 수정
    try:
        from comfy.ldm.cosmos.position_embedding import VideoRopePosition3DEmb
        VideoRopePosition3DEmb.generate_embeddings = _rope_generate_embeddings_dynamic
        print("[patch] VideoRopePosition3DEmb.generate_embeddings → 동적 H/W 버전 치환 완료")
    except ImportError:
        print("[patch] (경고) VideoRopePosition3DEmb 미발견 — RoPE 패치 건너뜀")


def save_onnx_external(onnx_program_or_none, wrapper, args_tuple, out_path,
                       input_names, output_names, dynamic_axes=None,
                       opset=18, use_dynamo=False):
    """
    legacy tracer 우선, 실패 시(특히 2GB protobuf 한계) --dynamo로 재시도하라고 안내.
    최신 torch의 legacy exporter는 2GB 초과 시 자동으로 external data로 떨어지지만,
    버전에 따라 실패할 수 있으므로 dynamo 경로도 제공.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if use_dynamo:
        ep = torch.onnx.export(wrapper, args_tuple, dynamo=True)
        try:
            ep.save(out_path, external_data=True)
        except TypeError:
            ep.save(out_path)  # 구버전 시그니처
    else:
        torch.onnx.export(
            wrapper, args_tuple, out_path,
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=opset,
            do_constant_folding=True,
        )
    size = sum(
        os.path.getsize(os.path.join(d, f))
        for d, _, fs in os.walk(os.path.dirname(os.path.abspath(out_path)))
        for f in fs
    )
    print(f"[export] {out_path} 저장 완료 (출력 디렉토리 총 {size / 1e9:.2f} GB)")


@torch.no_grad()
def verify_onnx(onnx_path, wrapper, args_tuple, input_names, atol=5e-2):
    """
    PyTorch 원본 vs ONNX Runtime 출력 비교.
    fp16 + fp32-residual 혼합 그래프라 오차 허용치는 느슨하게(기본 5e-2) 잡는다.
    절대오차보다 '이미지가 깨질 수준의 발산'을 잡는 게 목적.
    """
    import numpy as np
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"[verify] ORT providers: {sess.get_providers()}")

    feed = {
        name: t.detach().cpu().numpy()
        for name, t in zip(input_names, args_tuple)
    }
    ort_out = sess.run(None, feed)

    torch_out = wrapper(*args_tuple)
    if isinstance(torch_out, torch.Tensor):
        torch_out = (torch_out,)

    ok = True
    for i, (a, b) in enumerate(zip(torch_out, ort_out)):
        a = a.detach().float().cpu().numpy()
        b = b.astype(np.float32)
        max_abs = float(np.max(np.abs(a - b)))
        denom = float(np.max(np.abs(a))) + 1e-6
        rel = max_abs / denom
        print(f"[verify] output[{i}]: max_abs_diff={max_abs:.5f} "
              f"(rel={rel:.4f}, ref_max={denom:.3f})")
        if rel > atol:
            ok = False
    print("[verify] PASS" if ok else "[verify] FAIL — 오차가 큼. fp16 수치 문제 의심.")
    return ok


def base_argparser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--comfyui", required=True, help="ComfyUI 루트 디렉토리 경로")
    p.add_argument("--out", required=True, help="출력 .onnx 경로")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--verify", action="store_true", help="export 후 ORT로 출력 비교")
    p.add_argument("--dynamo", action="store_true",
                   help="legacy tracer 실패 시 dynamo exporter 사용")
    return p
