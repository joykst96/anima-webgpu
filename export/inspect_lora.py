"""
Anima LoRA 파일 일괄 진단 — 슬롯 변환기 설계/검증용.

폴더 내 *.safetensors를 스캔해서 파일별로:
  - 키 네이밍 포맷 (kohya lora_down/up | PEFT lora_A/B | 기타)
  - 랭크 분포, dtype, alpha 유무, DoRA 여부
  - 표준 타깃 10모듈(self/cross qkvo + mlp1/2) 커버리지
  - 매핑 안 되는 모듈 패턴 상위 N개 (정규화된 이름)

사용:
  python export/inspect_lora.py --dir ComfyUI\\models\\loras
  python export/inspect_lora.py --dir ... --file 특정파일.safetensors --verbose
"""
import argparse
import collections
import glob
import os
import re

from safetensors import safe_open

# 표준 타깃 (블록당 10모듈) — add_lora_slots 슬롯과 1:1
CANON = re.compile(
    r"^blocks[._](\d+)[._]("
    r"self_attn[._](?:q|k|v|output)_proj|"
    r"cross_attn[._](?:q|k|v|output)_proj|"
    r"mlp[._]layer[12])$")

# 키 → (모듈명, 역할) 분해. 역할: down/up/alpha/dora/기타
SUFFIXES = [
    (".lora_down.weight", "down"), (".lora_up.weight", "up"),
    (".lora_A.weight", "down"), (".lora_B.weight", "up"),
    (".lora_down", "down"), (".lora_up", "up"),
    (".lora_A", "down"), (".lora_B", "up"),
    (".alpha", "alpha"), (".dora_scale", "dora"),
    (".diff", "diff"), (".diff_b", "diff_b"),
]
PREFIXES = ["lora_unet_", "lora_te_", "lora_te1_", "lora_te2_",
            "diffusion_model.", "model.diffusion_model.", "transformer.",
            "lora_transformer_", "base_model.model.", "unet."]


def split_key(key):
    role = None
    base = key
    for suf, r in SUFFIXES:
        if key.endswith(suf):
            base, role = key[: -len(suf)], r
            break
    prefix = ""
    for p in PREFIXES:
        if base.startswith(p):
            prefix, base = p, base[len(p):]
            break
    return prefix, base, role


def norm_module(base):
    """프리픽스 제거된 모듈 경로를 정규화: 구분자 통일."""
    return base.replace(".", "_")


def inspect_file(path, verbose=False):
    f = safe_open(path, framework="pt")
    keys = list(f.keys())
    prefixes = collections.Counter()
    roles = collections.Counter()
    ranks = collections.Counter()
    dtypes = collections.Counter()
    modules = collections.defaultdict(set)     # 정규화 모듈명 -> 보유 역할
    unmapped_suffix = collections.Counter()

    for k in keys:
        prefix, base, role = split_key(k)
        prefixes[prefix or "(없음)"] += 1
        if role is None:
            unmapped_suffix[k.rsplit(".", 1)[-1]] += 1
            continue
        roles[role] += 1
        mod = norm_module(base)
        modules[mod].add(role)
        sl = f.get_slice(k)
        if role == "down":
            shp = sl.get_shape()
            if len(shp) == 2:
                ranks[min(shp)] += 1
        dtypes[str(sl.get_dtype())] += 1

    canon_mods = [m for m in modules
                  if CANON.match(re.sub(r"^(blocks)[._]", r"\1_", m))]
    # 블록 수 추정
    blk_ids = set()
    other_mods = []
    for m in modules:
        mm = re.match(r"^blocks[._]?(\d+)[._](.+)$", m)
        if mm:
            blk_ids.add(int(mm.group(1)))
            if not CANON.match(m):
                other_mods.append(mm.group(2))
        elif not CANON.match(m):
            other_mods.append(m)

    name = os.path.basename(path)
    print(f"== {name}")
    print(f"   키 {len(keys)} · 프리픽스 {dict(prefixes)}")
    print(f"   역할 {dict(roles)} · dtype {dict(dtypes)}")
    print(f"   랭크 {dict(ranks)} · 블록 {len(blk_ids)}개"
          f" · 표준타깃 모듈 {len(canon_mods)}/{len(modules)}")
    if "dora" in roles:
        print("   [주의] DoRA 가중치 포함 — 단순 슬롯으론 근사 적용")
    if unmapped_suffix:
        print(f"   [미해석 접미사] {dict(unmapped_suffix.most_common(5))}")
    nonstd = collections.Counter(other_mods)
    if nonstd:
        print(f"   [비표준 모듈 상위] {dict(nonstd.most_common(6))}")
    if verbose:
        for k in keys[:12]:
            print("    ", k, tuple(f.get_slice(k).get_shape()))
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=".")
    p.add_argument("--file", default=None, help="특정 파일만")
    p.add_argument("--verbose", action="store_true", help="앞 12개 키 원문 출력")
    args = p.parse_args()

    paths = ([os.path.join(args.dir, args.file)] if args.file
             else sorted(glob.glob(os.path.join(args.dir, "*.safetensors"))))
    for path in paths:
        try:
            inspect_file(path, args.verbose)
        except Exception as e:
            print(f"== {os.path.basename(path)}\n   [에러] {e}\n")


if __name__ == "__main__":
    main()
