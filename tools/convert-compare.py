"""브라우저 변환기 결과(tools/convert-folder.mjs 또는 변환 페이지 폴더)와 export 파이프라인 산출물을 바이트 대조한다.

  python tools/convert-compare.py --conv <변환 폴더>/base --py <빌드 폴더> --prefix <빌드 접두사> --variant dyn32
  python tools/convert-compare.py --conv <변환 폴더>/turbo --py <빌드 폴더> --prefix <빌드 접두사> --variant turbo32

변환기 산출물은 이름이 매핑표의 stem(animav10_dit_*)으로 고정이라, 파이프라인 쪽 이름(<prefix>_dit_*)을 같은 이름으로 바꿔 비교한다:
샤드·manifest는 파일 이름만, 그래프는 external_data location 문자열을 바꿔 다시 직렬화한 바이트로 (직렬화 왕복이 원본과 같은지 먼저 확인).
샤드가 다르면 매핑표(web/convert/anima2b.map.json)로 어느 텐서가 다른지 짚는다. ComfyUI python(onnx 설치됨)으로 실행.
"""
import argparse
import hashlib
import json
import os

import onnx

ap = argparse.ArgumentParser()
ap.add_argument("--conv", required=True)
ap.add_argument("--py", required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--variant", required=True, choices=["dyn32", "turbo32"])
ap.add_argument("--map", default=os.path.join(os.path.dirname(__file__), "..", "web", "convert", "anima2b.map.json"))
a = ap.parse_args()

M = json.load(open(a.map, encoding="utf-8"))
vkey = "base" if a.variant == "dyn32" else "turbo"
cstem = M["stems"][vkey]  # animav10_dit_dyn32 / animav10_dit_turbo32
pstem = f"{a.prefix}_dit_{a.variant}"
ok = True


def rd(p):
    with open(p, "rb") as f:
        return f.read()


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


# ── 샤드
cs, ps = {}, {}
for i, s in enumerate(M["shards"]):
    cn = s["name"].replace("{stem}", cstem)
    cs[i], ps[i] = rd(os.path.join(a.conv, cn)), rd(os.path.join(a.py, cn.replace(cstem, pstem)))
    same = cs[i] == ps[i]
    ok &= same
    print(f"[shard] {cn}: {'동일' if same else '다름'} {sha(cs[i])} / {sha(ps[i])}")

# 다르면 텐서 단위로
if not all(cs[i] == ps[i] for i in cs):
    bad = []
    for t in M["tensors"]:
        if t["kind"] == "q8":
            nb = -(-t["K"] // M["block"])
            regions = [("B", t["shard"], t["offset"], t["N"] * nb * M["block"]), ("scales", t["sshard"], t["soffset"], t["N"] * nb * 4)]
        else:
            n = 4
            for d in t["shape"]:
                n *= d
            regions = [(t["kind"], t["shard"], t["offset"], n)]
        for what, sh, off, ln in regions:
            x, y = cs[sh][off:off + ln], ps[sh][off:off + ln]
            if x != y:
                diff = sum(1 for p, q in zip(x, y) if p != q)
                bad.append(f"{t['key']} {what}: {diff}/{ln} 바이트 다름")
    print(f"[shard] 다른 텐서 {len(bad)}개")
    for b in bad[:30]:
        print("   ", b)

# ── 그래프
for gk, g in M["graphs"].items():
    if not gk.startswith(vkey):
        continue
    cb = rd(os.path.join(a.conv, g["file"]))
    pf = os.path.join(a.py, g["file"].replace(cstem, pstem))
    pm = onnx.load(pf, load_external_data=False)
    rt = pm.SerializeToString() == rd(pf)  # 직렬화 왕복이 원본 바이트와 같아야 이름 치환 비교가 의미 있다
    n = 0
    for init in pm.graph.initializer:
        for e in init.external_data:
            if e.key == "location" and pstem in e.value:
                e.value = e.value.replace(pstem, cstem)
                n += 1
    pb = pm.SerializeToString()
    same = rt and pb == cb
    ok &= same
    msg = f"[graph] {g['file']}: {'동일' if same else '다름'} (location {n}개 치환, 왕복 {'OK' if rt else '불일치'})"
    if not same and rt:
        cm = onnx.load_from_string(cb)
        ci = {i.name: i for i in cm.graph.initializer}
        di = [i.name for i in pm.graph.initializer if i.name not in ci or i.SerializeToString() != ci[i.name].SerializeToString()]
        nodes_same = [x.SerializeToString() for x in pm.graph.node] == [x.SerializeToString() for x in cm.graph.node]
        msg += f" — initializer 다름 {len(di)}개 {di[:5]}, 노드 {'같음' if nodes_same else '다름'}"
    print(msg)

# ── manifest·lora_manifest (이름만 치환)
for f in [f"{cstem}_q8a4fns.onnx.manifest.json", f"{cstem}_q8a4fnLs.onnx.manifest.json", f"{cstem}_q8a4fnLs.onnx.lora_manifest.json"]:
    c = rd(os.path.join(a.conv, f)).decode()
    p = rd(os.path.join(a.py, f.replace(cstem, pstem))).decode().replace(pstem, cstem)
    same = c == p
    ok &= same
    print(f"[json] {f}: {'동일' if same else '다름'}")

print("[compare] 통과" if ok else "[compare] 실패")
raise SystemExit(0 if ok else 1)
