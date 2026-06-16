"""
DiT ONNX → MatMulNBits weight-only 양자화 (브라우저 탑재용).

fp16 DiT(4.18GB)는 브라우저 ArrayBuffer/wasm 힙 한계를 넘어 로드 불가.
MatMul 가중치를 int4(기본) 또는 int8로 블록 양자화해 크기를 줄인다.
- weight-only: 가중치만 양자화하고 activation dtype은 원본 그래프 그대로 둔다.
  (fp32-act 모델이면 활성 fp32, fp16-act 모델이면 fp16 — 이 단계는 안 건드림)
  셰이더에서 역양자화 후 곱셈. accuracy_level=4면 활성을 int8로 동적 양자화해
  DP4A 정수 커널을 탄다(활성 원본 dtype과 무관하게 발동).
- ORT WebGPU EP에 MatMulNBits 전용 커널 존재 (Phi-3 웹 데모 등에서 검증된 경로)
- norm/embedding/conv 등 비양자화 레이어는 원본 dtype 그대로 유지됨

사용 예 (portable 루트에서):
  python export/quantize_dit.py ^
    --src out\\dit\\anima_dit_1024.onnx ^
    --out out\\dit\\anima_dit_1024_q4.onnx --bits 4

예상 크기: int4 ~1.2GB / int8 ~2.2GB
RAM 사용: 원본+결과 동시 적재로 10GB 안팎 필요.
"""
import argparse
import os

import onnx


def find_exclude_nodes(model):
    """민감 레이어 MatMul 식별 — dynamo exporter가 이름을 익명화(val_N)하므로
    이름이 아닌 그래프 구조로 잡는다:
    1) timesteps 입력에서 도달 가능 AND latent 입력에서 도달 불가능
       → t_embedder + 모든 adaln_modulation (시간 임베딩 전용 가지)
    2) 가중치에 64 차원 포함 (patch 2x2x16ch) → x_embedder / final_layer 투영
    """
    g = model.graph
    init_dims = {i.name: list(i.dims) for i in g.initializer}
    producers = {}
    for n in g.node:
        for o in n.output:
            producers[o] = n

    def resolve(name, d=0):
        if name in init_dims or d > 4:
            return name
        pn = producers.get(name)
        if pn is not None and pn.op_type in ("Transpose", "Cast", "Reshape",
                                             "Identity"):
            return resolve(pn.input[0], d + 1)
        return name

    def reachable_from(start_tensor):
        """ONNX 그래프는 위상 정렬돼 있으므로 단일 순방향 스캔으로 충분."""
        seen = {start_tensor}
        nodes = set()
        for idx, n in enumerate(g.node):
            if any(i in seen for i in n.input):
                nodes.add(idx)
                seen.update(n.output)
        return nodes

    from_t = reachable_from("timesteps")
    from_x = reachable_from("latent")

    excluded = []
    n_topo = n_shape = 0
    for idx, n in enumerate(g.node):
        if n.op_type not in ("MatMul", "Gemm") or len(n.input) < 2:
            continue
        is_t_branch = idx in from_t and idx not in from_x
        dims = init_dims.get(resolve(n.input[1]))
        is_io_proj = dims is not None and any(d == 64 for d in dims)
        if is_t_branch or is_io_proj:
            excluded.append(n.name)
            n_topo += is_t_branch
            n_shape += (not is_t_branch) and is_io_proj
    print(f"[exclude] 시간임베딩 가지(t_embedder/adaln) {n_topo}개, "
          f"입출력 투영(64ch) {n_shape}개")
    return excluded


def get_quantizer(model, bits, block_size, nodes_to_exclude, algo="rtn",
                  accuracy_level=None):
    """ORT 버전별 API 차이 흡수: MatMulNBitsQuantizer(신) → MatMul4BitsQuantizer(구)"""
    try:
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            MatMulNBitsQuantizer)
        algo_config = None
        if algo == "hqq":
            from onnxruntime.quantization.matmul_nbits_quantizer import (
                HQQWeightOnlyQuantConfig)
            algo_config = HQQWeightOnlyQuantConfig(block_size=block_size,
                                                   bits=bits)
            print(f"[quant] HQQ 알고리즘 사용 (최적화 기반 — RTN보다 오래 걸림)")
        try:
            q = MatMulNBitsQuantizer(model, block_size=block_size,
                                     is_symmetric=True, bits=bits,
                                     nodes_to_exclude=nodes_to_exclude,
                                     algo_config=algo_config,
                                     accuracy_level=accuracy_level)
            print(f"[quant] MatMulNBitsQuantizer, bits={bits}, block={block_size}")
            return q
        except TypeError:
            if bits != 4:
                raise RuntimeError("이 ORT 버전은 bits 지정 미지원 — 4비트만 가능. "
                                   "--bits 4로 재시도하거나 onnxruntime 업그레이드.")
            q = MatMulNBitsQuantizer(model, block_size=block_size,
                                     is_symmetric=True,
                                     nodes_to_exclude=nodes_to_exclude)
            print(f"[quant] MatMulNBitsQuantizer(4bit 고정), block={block_size}")
            return q
    except ImportError:
        from onnxruntime.quantization.matmul_4bits_quantizer import (
            MatMul4BitsQuantizer)
        if bits != 4:
            raise RuntimeError("이 ORT 버전은 4비트만 지원. --bits 4로 재시도.")
        q = MatMul4BitsQuantizer(model, block_size=block_size, is_symmetric=True,
                                 nodes_to_exclude=nodes_to_exclude)
        print(f"[quant] MatMul4BitsQuantizer, block={block_size}")
        return q


def apply_accuracy_level(qmodel, level):
    """모든 MatMulNBits 노드에 accuracy_level 속성을 강제 설정.

    양자화기 kwarg가 무시되는 경로(HQQ algo_config, 구버전 API)까지 보장하는
    사후 패스. level=4 = "활성화 int8 동적 양자화 허용" → ORT WebGPU 네이티브
    EP(onnxruntime-web jspi 빌드)의 DP4A MatMulNBits 커널 발동 조건.
    나머지 발동 조건(block_size%32==0, K%128==0, N%16==0)도 함께 점검해 출력.
    """
    import onnx.helper as oh
    n_set, n_dp4a_ok, n_dp4a_no = 0, 0, 0
    for n in qmodel.graph.node:
        if n.op_type != "MatMulNBits":
            continue
        attrs = {a.name: a for a in n.attribute}
        if "accuracy_level" in attrs:
            attrs["accuracy_level"].i = level
        else:
            n.attribute.append(oh.make_attribute("accuracy_level", level))
        n_set += 1
        K = attrs["K"].i if "K" in attrs else 0
        N = attrs["N"].i if "N" in attrs else 0
        bs = attrs["block_size"].i if "block_size" in attrs else 0
        if level == 4:
            if K % 128 == 0 and N % 16 == 0 and bs % 32 == 0:
                n_dp4a_ok += 1
            else:
                n_dp4a_no += 1
                print(f"  [dp4a-skip] {n.name}: K={K} N={N} block={bs} "
                      f"— 차원 조건 미충족, 기본 커널로 동작")
    print(f"[accuracy_level] MatMulNBits {n_set}개에 level={level} 설정")
    if level == 4:
        print(f"[accuracy_level] DP4A 차원 조건 충족 {n_dp4a_ok}개 / "
              f"미충족 {n_dp4a_no}개")


SCRIPT_VERSION = "v5 (accuracy_level)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="원본 DiT .onnx (옆에 .data 필요)")
    p.add_argument("--out", required=True)
    p.add_argument("--bits", type=int, default=4, choices=[4, 8])
    p.add_argument("--block-size", type=int, default=32,
                   help="양자화 블록 크기 (작을수록 정확, 클수록 작은 파일)")
    p.add_argument("--no-exclude", action="store_true",
                   help="민감 레이어 제외 없이 전부 양자화. 제외 레이어는 "
                        "원본 정밀도(fp32)로 남아 용량이 커지므로, int8은 "
                        "제외 없이 가는 것을 권장 (int8은 제외 불필요할 만큼 "
                        "정확). 제외는 int4가 아슬아슬할 때의 카드.")
    p.add_argument("--accuracy-level", type=int, default=None,
                   choices=[1, 2, 3, 4], dest="accuracy_level",
                   help="MatMulNBits accuracy_level 속성 (1=fp32 2=fp16 "
                        "3=bf16 4=int8). 4를 주면 ORT WebGPU 네이티브 EP가 "
                        "활성화를 int8로 동적 양자화해 DP4A 정수 커널을 탐 — "
                        "브라우저(jspi 빌드) 가속의 핵심. 미지정 시 기존 동작. "
                        "주의: 활성화 양자화는 가중치 양자화와 별개의 정확도 "
                        "트레이드오프 — 브라우저 이미지 비교 필수.")
    p.add_argument("--algo", default="rtn", choices=["rtn", "hqq"],
                   help="양자화 알고리즘. rtn=기본(빠름, 거침), "
                        "hqq=최적화 기반(느림, int4에서 훨씬 정확, torch 필요)")
    args = p.parse_args()

    print(f"[quantize_dit {SCRIPT_VERSION}]")
    print(f"[load] {args.src} (external data 포함 메모리 적재)")
    model = onnx.load(args.src)  # 같은 디렉토리의 .data 자동 로드

    if args.no_exclude:
        excluded = []
    else:
        excluded = find_exclude_nodes(model)
    total_mm = sum(1 for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    print(f"[exclude] 민감 레이어 MatMul {len(excluded)}개 제외 "
          f"(전체 {total_mm}개) — t_embedder/adaln/final_layer/x_embedder는 "
          f"fp 유지")
    if 0 < len(excluded) <= 12:
        for name in excluded:
            print(f"  - {name}")

    quant = get_quantizer(model, args.bits, args.block_size, excluded,
                          args.algo, args.accuracy_level)
    quant.process()
    qmodel = quant.model.model if hasattr(quant.model, "model") else quant.model

    if args.accuracy_level is not None:
        apply_accuracy_level(qmodel, args.accuracy_level)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    data_name = os.path.basename(args.out) + ".data"

    # protobuf 2GB 한계 — 저장 전 크기를 직접 계산해서 분기 (try/except 의존 제거)
    approx = sum(len(i.raw_data) for i in qmodel.graph.initializer
                 if i.raw_data) + 64 * 2**20  # 그래프 오버헤드 여유
    print(f"[save] 예상 크기 {approx / 2**30:.2f} GiB")
    if approx < 1.8 * 2**30:
        onnx.save_model(qmodel, args.out)
        print(f"[save] 단일 파일 저장: {args.out}")
    else:
        onnx.save_model(qmodel, args.out, save_as_external_data=True,
                        all_tensors_to_one_file=True, location=data_name,
                        convert_attribute=False)
        print(f"[save] external data 저장: {args.out} + {data_name}")

    total = os.path.getsize(args.out)
    dpath = os.path.join(os.path.dirname(os.path.abspath(args.out)), data_name)
    if os.path.exists(dpath):
        total += os.path.getsize(dpath)
    print(f"[done] 총 크기 {total / 2**30:.2f} GiB "
          f"(원본 fp16 대비 {total / (4.18 * 2**30) * 100:.0f}%)")
    print("웹페이지의 DiT 경로를 이 파일로 바꾸면 됨 (.data는 자동 감지).")


if __name__ == "__main__":
    main()
