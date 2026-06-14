"""
VAE 디코더 ONNX에 타일링을 깨는 전역 연산(attention/softmax/global pool)이 있는지 진단.

tiled VAE(blend든 crop이든)는 RF가 유한해야 성립. Wan VAE mid block의
attention이 그래프에 남아있으면 RF=전역 → 타일 경계가 서로의 전체를 참조해
어떤 overlap으로도 seam이 안 사라진다. 이 경우 타일링 불가(또는 attention만
별도 처리 필요).

사용 (한 줄):
  python_embeded\python.exe export\diag_vae_global.py --vae out\vae\qwen_image_vae_decoder_dyn32_2d.onnx
"""
import argparse
from collections import Counter

import onnx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", required=True)
    args = p.parse_args()

    g = onnx.load(args.vae, load_external_data=False).graph
    c = Counter(n.op_type for n in g.node)

    print(f"[vae] 총 노드 {sum(c.values())}, op 종류 {len(c)}")
    print("[vae] op_type 분포:")
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"   {k:24s} {v}")

    # 타일링을 깨는 전역 연산 후보
    GLOBAL_OPS = {"Softmax", "Attention", "MultiHeadAttention",
                  "GlobalAveragePool", "ReduceMean", "ReduceMax", "ReduceSum",
                  "InstanceNormalization", "GroupNormalization", "LayerNormalization"}
    found = {k: c[k] for k in GLOBAL_OPS if k in c}
    print()
    if found:
        print(f"[!] 전역/준전역 연산 발견: {found}")
        # Softmax/Attention은 진짜 전역. Norm류는 채널 단위라 공간 타일링엔 보통 무해
        hard = {k: v for k, v in found.items()
                if k in ("Softmax", "Attention", "MultiHeadAttention",
                         "GlobalAveragePool")}
        soft = {k: v for k, v in found.items() if k not in hard}
        if hard:
            print(f"[!!] 공간 전역 연산(타일링 차단): {hard}")
            print("     → mid-block attention 추정. blend overlap으로 해결 불가.")
            print("     대안: (a) attention을 패치로 제거/근사 후 재export,")
            print("           (b) 전역 attention만 풀해상도로 1회 + 나머지 타일,")
            print("           (c) 타일링 포기하고 다른 메모리 절감(예: VAE fp16 입력).")
        if soft:
            print(f"[i] 정규화류(공간 타일링엔 보통 무해, 채널/그룹 단위): {soft}")
            print("     단 GroupNorm이 공간평균을 포함하면 타일 경계서 미세차 — blend로 흡수 가능.")
    else:
        print("[ok] 전역 연산 없음 — conv-only. RF 유한, 타일링 성립.")
        print("     (RF가 크게 측정됐다면 단순히 깊은 conv 스택 때문 — overlap 키우면 됨.)")

    # Conv 커널/패딩 요약 (RF 추정 보조)
    convs = [n for n in g.node if n.op_type == "Conv"]
    print(f"\n[conv] Conv 노드 {len(convs)}개")
    ksizes = Counter()
    for n in convs:
        ks = next((tuple(a.ints) for a in n.attribute if a.name == "kernel_shape"), None)
        if ks:
            ksizes[ks] += 1
    print(f"[conv] kernel_shape 분포: {dict(ksizes)}")
    # Resize/Upsample (업샘플 단계 수 = RF 증폭 요인)
    ups = [n for n in g.node if n.op_type in ("Resize", "Upsample", "ConvTranspose")]
    print(f"[conv] 업샘플 노드(Resize/Upsample/ConvTranspose): {len(ups)}")


if __name__ == "__main__":
    main()
