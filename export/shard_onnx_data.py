"""
ONNX 모델의 가중치를 N개의 샤드 파일로 분할.

목적:
- 브라우저 단일 ArrayBuffer 상한(~2GB) 우회: 샤드 각각을 따로 fetch
- CDN 캐시 한도 대응 (Cloudflare 무료 플랜 단일 파일 512MB)

출력: <out>.onnx (작은 그래프) + <out>.onnx.0.bin, .1.bin, ... + manifest.json
웹페이지(index.html)는 manifest를 자동 감지해서 샤드들을 모아 로드한다.

사용 예:
  python export/shard_onnx_data.py ^
    --src out\\dit\\anima_dit_dyn32_q8.onnx ^
    --out out\\dit\\anima_dit_dyn32_q8s.onnx --shard-mb 480
"""
import argparse
import json
import os

import onnx
from onnx.external_data_helper import set_external_data

ALIGN = 4096  # 오프셋 정렬 (안전 여유)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="원본 .onnx (옆에 .data 있어도 됨)")
    p.add_argument("--out", required=True, help="출력 .onnx 경로")
    p.add_argument("--shard-mb", type=int, default=480,
                   help="샤드당 최대 크기 MB (CDN 캐시 한도 고려 기본 480)")
    args = p.parse_args()

    print(f"[load] {args.src}")
    model = onnx.load(args.src)  # external data 포함 메모리 적재

    outdir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(outdir, exist_ok=True)
    base = os.path.basename(args.out)
    shard_limit = args.shard_mb * 2**20

    shard_names = []
    cur_f = None
    cur_off = 0

    def open_new_shard():
        nonlocal cur_f, cur_off
        if cur_f:
            cur_f.close()
        name = f"{base}.{len(shard_names)}.bin"
        shard_names.append(name)
        cur_f = open(os.path.join(outdir, name), "wb")
        cur_off = 0

    open_new_shard()
    moved = 0
    for init in model.graph.initializer:
        data = init.raw_data
        if not data or len(data) < 1024:
            continue  # 소형/비raw 텐서는 그래프에 인라인 유지
        if cur_off > 0 and cur_off + len(data) > shard_limit:
            open_new_shard()
        # 정렬 패딩
        pad = (-cur_off) % ALIGN
        if pad:
            cur_f.write(b"\x00" * pad)
            cur_off += pad
        cur_f.write(data)
        set_external_data(init, location=shard_names[-1],
                          offset=cur_off, length=len(data))
        init.ClearField("raw_data")
        cur_off += len(data)
        moved += len(data)
    cur_f.close()

    onnx.save_model(model, args.out)
    manifest = {"shards": shard_names}
    with open(args.out + ".manifest.json", "w") as f:
        json.dump(manifest, f)

    sizes = [os.path.getsize(os.path.join(outdir, s)) / 2**20 for s in shard_names]
    print(f"[done] 그래프: {args.out} ({os.path.getsize(args.out) / 2**20:.1f} MB)")
    print(f"[done] 샤드 {len(shard_names)}개: " +
          ", ".join(f"{s:.0f}MB" for s in sizes) +
          f" (총 {moved / 2**30:.2f} GiB)")
    print(f"[done] manifest: {args.out}.manifest.json")
    print("웹페이지는 manifest를 자동 감지함. .onnx와 .bin들, manifest를 "
          "같은 디렉토리에 둘 것.")


if __name__ == "__main__":
    main()
