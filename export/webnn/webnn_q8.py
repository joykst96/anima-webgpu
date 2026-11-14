# WebNN용 fp16 그래프(webnn_fix.py 산출물)의 선형층 가중치를 int8(블록 32, 대칭 RTN)로 바꾼다 — 다운로드 크기·GPU 메모리용.
#   python -I export/webnn/webnn_q8.py <src.onnx> <dst.onnx> [--dq-cast] [--rot 64]      (dst 옆에 <dst>.data를 쓴다)
#   python -I export/webnn/webnn_q8.py <src.onnx> <dst.onnx> --like <ref.onnx>            (그래프만: ref의 int8 가중치를 가리킨다)
#
# 계산은 여전히 fp16이다. 가중치만 int8로 두고 그래프 안에서 풀어 쓴다("수동 역양자화"):
#   Wq int8 [K/32, 32, N] → Cast(fp32) → Mul(scale fp32 [K/32, 1, N]) → Cast(fp16) → Reshape([K, N]) → MatMul
# DequantizeLinear를 블록 단위로 쓰지 않는 이유(2026-11-13 실측):
#   - 블록 단위 DequantizeLinear(axis=0, block_size=32)는 네이티브 DML EP는 받지만 Chrome 157 WebNN 백엔드가 세션 생성에서 거부한다.
#   - 열 단위(per-channel)는 ORT Web 1.26의 WebNN EP가 scale 랭크 오류를 낸다.
#   - Cast/Mul/Reshape만 쓰면 어디서나 받고, 속도도 fp16 가중치와 같다.
# 양자화 단위는 운영 그래프(MatMulNBits bits 8, block 32)와 같다: 입력 차원 K를 32개씩 묶고 출력 열마다 scale 하나.
#
# --dq-cast: int8→fp32를 Cast 대신 DequantizeLinear(scale 1, 스칼라)로 한다. ORT의 상수 접기는 DequantizeLinear를 건너뛰므로
#   로드 때 wasm에서 가중치를 fp16으로 풀어 버리는 일(전체 모델에서 4GB 힙 초과 → std::bad_alloc)이 없다.
#   동적 그래프는 shape 계산을 접으려면 최적화를 켜야 해서 이 옵션이 필요하다.
#
# --rot G (ConvRot): 입력축 K를 G개씩 묶어 정규 Hadamard H(h4의 크로네커 곱, 1/√G)로 회전한 가중치 H·W를 양자화하고,
#   그래프에서는 MatMul 입력에 같은 회전 x·H를 넣는다(H 대칭·직교 → (xH)(HW) = xW). 회전은 같은 입력을 쓰는 MatMul끼리 공유.
#   회전값은 float64로 정확히 계산한 뒤 float32로 한 번만 반올림한다: 가중치가 fp16 값이고 H가 ±1/√G(G = 4^k → 2의 거듭제곱)라
#   합이 float64에서 오차 없이 나온다 → 계산 순서와 무관하게 같은 값이라 브라우저 변환기(web/js/convert-core.js quant8rot)가 바이트 단위로 재현한다.
#   효과(2026-11-14, Turbo 동적 1024², PyTorch 기준 DiT 1회 오차 σ1/0.5/0.1): 2.50/1.13/0.96% → 1.89/0.96/0.89%, 스텝 +1.5%.
#
# --like <ref.onnx>: 가중치 계산 없이 그래프만 바꾼다. 같은 가중치를 쓰는 다른 fp16 그래프(NegPip 입력·LoRA 슬롯을 넣은 것)를
#   ref(이 스크립트로 만들고 샤딩한 int8 그래프)와 같은 노드로 바꾸고, 초기값은 ref의 것(샤드 위치)을 가리키게 한다. 회전·dq-cast 설정도 ref를 따른다.
#   ref에 없는 초기값은 LoRA 슬롯의 .lora.bin 것과 그래프 안 상수만 허용한다(그대로 둔다). manifest는 ref 것을 복사한다.
#
# 양자화하지 않는 것: fp16 2D가 아닌 것, K가 32의 배수가 아닌 것(x_embedder K=68), 작은 것(원소 64K 미만), MatMul의 B 말고도 쓰이는 것,
#   LoRA 슬롯 가중치(.lora.bin에 있는 것 — 페이지가 채운다).
import os
import shutil
import sys

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

BLOCK, MIN_ELEMS = 32, 1 << 16


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


ROT = int(opt("--rot", "0"))
LIKE = opt("--like")
skip = {str(ROT), LIKE}
args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in skip]
src, dst = args[0], args[1]
DQ_CAST = "--dq-cast" in sys.argv

ref = None
if LIKE:
    ref = onnx.load(LIKE, load_external_data=False)
    rinits = {t.name: t for t in ref.graph.initializer}
    DQ_CAST = any(n.op_type == "DequantizeLinear" and n.name.endswith("_deq_dq") for n in ref.graph.node)
    ROT = rinits["rot_H"].dims[0] if "rot_H" in rinits else 0

m = onnx.load(src, load_external_data=not LIKE)
g = m.graph
inits = {t.name: t for t in g.initializer}
users = {}
for n in g.node:
    for i in n.input:
        users.setdefault(i, []).append(n)


def lora_bin(t):
    return any(e.key == "location" and e.value.endswith(".lora.bin") for e in t.external_data)


def hadamard_signs(G):
    """정규 Hadamard의 부호(±1, int8): h4의 크로네커 곱. G는 4의 거듭제곱."""
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], np.int8)
    h = h4
    while h.shape[0] < G:
        h = np.kron(h, h4)
    assert h.shape[0] == G, "정규 Hadamard는 4의 거듭제곱 크기만"
    return h


def rotate(W, G):
    """W fp32 [K, N] (값은 fp16) → H·W를 행 묶음(G)마다. float64로 정확히 합한 뒤 float32로 한 번 반올림."""
    K, N = W.shape
    S = hadamard_signs(G).astype(np.float64)
    inv = 1.0 / np.sqrt(G)                       # G = 4^k → 2^-k (정확)
    R = np.matmul(S, W.astype(np.float64).reshape(K // G, G, N)) * inv
    return R.reshape(K, N).astype(np.float32)


def quantize(W):
    """W fp32 [K, N] → q int8 [K/32, 32, N], scale fp32 [K/32, 1, N]"""
    K, N = W.shape
    W = W.reshape(K // BLOCK, BLOCK, N)
    scale = (np.abs(W).max(axis=1, keepdims=True) / np.float32(127)).astype(np.float32)
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(W / scale), -127, 127).astype(np.int8)
    return q, scale


if ROT:
    H16 = (hadamard_signs(ROT).astype(np.float32) / np.float32(np.sqrt(ROT))).astype(np.float16)
    g.initializer.extend([numpy_helper.from_array(H16, "rot_H"), numpy_helper.from_array(np.array([0], np.int64), "rot_z"),
                          numpy_helper.from_array(np.array([-1], np.int64), "rot_m1"), numpy_helper.from_array(np.array([ROT], np.int64), "rot_G")])

new_nodes, done, rotated = [], {}, {}
n_q = n_rot = bytes_before = bytes_after = 0
for n in g.node:
    w = inits.get(n.input[1]) if n.op_type == "MatMul" and len(n.input) == 2 else None
    if w is None or w.data_type != TP.FLOAT16 or len(w.dims) != 2 or w.dims[0] % BLOCK or w.dims[0] * w.dims[1] < MIN_ELEMS \
            or lora_bin(w) or any(u.op_type != "MatMul" or u.input[1] != w.name for u in users[w.name]):
        new_nodes.append(n)
        continue
    K, N = w.dims
    rot_here = ROT and K % ROT == 0
    if w.name not in done:
        qn, sn, shp, out = w.name + "_q8", w.name + "_q8s", w.name + "_q8shape", w.name + "_deq"
        g.initializer.remove(w)
        if LIKE:
            g.initializer.extend([onnx.TensorProto(name=qn), onnx.TensorProto(name=sn), onnx.TensorProto(name=shp)])  # 자리표시 — 아래에서 ref 것으로 바뀐다
        else:
            W = numpy_helper.to_array(w).astype(np.float32)
            if rot_here:
                W = rotate(W, ROT)
            q, scale = quantize(W)
            bytes_before += K * N * 2
            bytes_after += q.nbytes + scale.nbytes
            g.initializer.extend([numpy_helper.from_array(q, qn), numpy_helper.from_array(scale, sn),
                                  numpy_helper.from_array(np.array([K, N], np.int64), shp)])
        new_nodes += [helper.make_node("DequantizeLinear", [qn, "wq8_one", "wq8_zero"], [out + "_f32"], name=out + "_dq") if DQ_CAST else
                      helper.make_node("Cast", [qn], [out + "_f32"], to=TP.FLOAT, name=out + "_cast"),
                      helper.make_node("Mul", [out + "_f32", sn], [out + "_s"], name=out + "_mul"),
                      helper.make_node("Cast", [out + "_s"], [out + "_f16"], to=TP.FLOAT16, name=out + "_cast16"),
                      helper.make_node("Reshape", [out + "_f16", shp], [out], name=out + "_reshape")]
        done[w.name] = out
        n_q += 1
    n.input[1] = done[w.name]
    if rot_here:
        x = n.input[0]
        if x not in rotated:
            p = x + "_rot"
            # x[..., K] → [..., K/G, G] @ H → 원래 모양 (동적 그래프에서도 Shape로 모양을 그대로 따른다)
            new_nodes += [helper.make_node("Shape", [x], [p + "_shp"], name=p + "_shape"),
                          helper.make_node("Slice", [p + "_shp", "rot_z", "rot_m1"], [p + "_lead"], name=p + "_slice"),
                          helper.make_node("Concat", [p + "_lead", "rot_m1", "rot_G"], [p + "_s4"], axis=0, name=p + "_cat"),
                          helper.make_node("Reshape", [x, p + "_s4"], [p + "_g"], name=p + "_r1"),
                          helper.make_node("MatMul", [p + "_g", "rot_H"], [p + "_gh"], name=p + "_mm"),
                          helper.make_node("Reshape", [p + "_gh", p + "_shp"], [p], name=p + "_r2")]
            rotated[x] = p
            n_rot += 1
        n.input[0] = rotated[x]
    new_nodes.append(n)
del g.node[:]
g.node.extend(new_nodes)
if DQ_CAST:
    # 영점을 생략하면 ORT Web 1.26의 WebNN EP가 scale과 모양이 다른 기본 영점을 만들어 dequantizeLinear가 거부된다 → 명시
    g.initializer.extend([numpy_helper.from_array(np.array(1.0, np.float32), "wq8_one"), numpy_helper.from_array(np.array(0, np.int8), "wq8_zero")])

if LIKE:
    # 초기값을 ref의 것(외부 데이터 위치 포함)으로. ref에 없는 것은 LoRA 슬롯(.lora.bin)만 남긴다.
    keep, missing = [], []
    for t in g.initializer:
        if t.name in rinits:
            keep.append(rinits[t.name])
        elif lora_bin(t) or t.data_location != TP.EXTERNAL:   # LoRA 슬롯 가중치, NegPip·슬롯이 넣은 그래프 안 상수
            keep.append(t)
        else:
            missing.append(t.name)
    if missing:
        sys.exit(f"[webnn_q8 --like] ref에 없는 초기값 {len(missing)}개: {missing[:5]}")
    del g.initializer[:]
    g.initializer.extend(keep)
    onnx.save(m, dst)
    shutil.copy(LIKE + ".manifest.json", dst + ".manifest.json")
    print(f"[webnn_q8 --like] 가중치 {n_q}개·입력 회전 {n_rot}개를 {os.path.basename(LIKE)}와 같게 (그래프만, 초기값 {len(keep)}개) -> {dst}")
    sys.exit(0)

data = os.path.basename(dst) + ".data"
if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(dst)), data)):
    os.remove(os.path.join(os.path.dirname(os.path.abspath(dst)), data))
onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True, location=data, size_threshold=1024)
print(f"[webnn_q8] MatMul 가중치 {n_q}개를 int8(블록 {BLOCK}{f', 회전 G={ROT}' if ROT else ''})로: "
      f"{bytes_before / 2**30:.2f} → {bytes_after / 2**30:.2f} GiB, 입력 회전 {n_rot}개 -> {dst}")
