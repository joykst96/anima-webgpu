"""
브라우저 변환기(web/workers/convert-worker.js)용 매핑표 생성.

Anima 2B(28블록) 계열은 그래프 구조가 같고, 그래프의 가중치는 전부 체크포인트 텐서의 복사/전치/int8 양자화다.
그래서 기존 빌드(animav10_*) 그래프 12개를 그대로 템플릿으로 쓰고,
변환기는 체크포인트에서 가중치 바이트만 다시 만들어 템플릿과 같은 자리에 쓴다. 이 스크립트는 그 '자리'를 뽑는다.

대응은 이름이 아니라 값으로 찾는다: 체크포인트(bf16→fp32) 텐서마다 후보 바이트(그대로·전치·fp16·MLAS int8 B/scales)를
만들어 해시하고, 그래프 initializer(샤드 외부 데이터 + 그래프 안 인라인)의 바이트와 맞춘다. 같은 값의 텐서가 둘 이상이면
(예: 학습 안 된 norm이 전부 같은 값) 값으로는 못 가르므로 이름(dit.X ↔ net.X)으로 가르고, 그래도 안 되면 실패한다.
체크포인트에서 오지 않은 외부 데이터(LayerNorm 1.0, RoPE 표 등 구조 상수)는 베이스·Turbo 샤드에서 바이트가 같은지 확인하고
매핑표에 그대로 싣는다(base64). 샤드의 나머지(정렬 패딩)는 0.

검증: 생성 후 매핑표만으로 베이스 샤드·그래프를 다시 조립해(체크포인트 → 바이트) 원본과 바이트 동일한지 확인한다(--check).

사용 예 (ComfyUI portable 루트에서):
  python_embeded\\python.exe <repo>\\export\\make_convert_map.py ^
    --ckpt ComfyUI\\models\\diffusion_models\\anima-base-v1.0.safetensors ^
    --build ..\\work\\animav10 --prefix animav10 --out <repo>\\web\\convert\\anima2b.map.json --check
  --prefix: 템플릿 빌드의 build_dit.bat 접두사(그래프 이름 <prefix>_dit_{dyn32,turbo32}_*). 변환 결과 파일 이름도 이것을 따른다.
"""
import argparse
import base64
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np

STEMS = {}  # main()에서 --prefix로 채운다: {"base": "<prefix>_dit_dyn32", "turbo": "<prefix>_dit_turbo32"}
GRAPHS = ["_q8a4fns", "_q8a4fns_x", "_q8a4fns_f16", "_q8a4fnLs", "_q8a4fnLs_x", "_q8a4fnLs_f16"]
BLOCK = 32
# ONNX TensorProto.DataType
FLOAT, UINT8, FLOAT16 = 1, 2, 10


# ───────── safetensors (bf16 → fp32, torch 없이) ─────────
def read_safetensors(path, skip=lambda k: False):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__" or skip(k):
                continue
            assert v["dtype"] == "BF16", f"{k}: {v['dtype']} (BF16만 지원)"
            a, b = v["data_offsets"]
            f.seek(base + a)
            u16 = np.frombuffer(f.read(b - a), dtype="<u2")
            out[k] = (u16.astype(np.uint32) << 16).view(np.float32).reshape(v["shape"])
    return out


# ───────── MLAS MlasQuantizeBlockwise<float, 8> (대칭, 열 방향) 재현 ─────────
def quant8(w):
    """w: Linear weight [N, K] (= MatMul B[K,N]의 전치). → (B uint8 [N, ceil(K/32)*32], scales f32 [N, ceil(K/32)])"""
    N, K = w.shape
    nb = (K + BLOCK - 1) // BLOCK
    Kp = nb * BLOCK
    pad = np.zeros((N, Kp), np.float32)
    pad[:, :K] = w
    blk = pad.reshape(N, nb, BLOCK)
    # 패딩 칸은 min/max에 넣지 않는다(MLAS는 행 범위만 훑음) → 패딩을 첫 원소로 채워 영향 제거
    if Kp != K:
        tmp = blk.copy().reshape(N, Kp)
        tmp[:, K:] = tmp[:, (nb - 1) * BLOCK:(nb - 1) * BLOCK + 1]
        blk_mm = tmp.reshape(N, nb, BLOCK)
    else:
        blk_mm = blk
    mn = blk_mm.min(axis=2)
    mx = blk_mm.max(axis=2)
    v = np.where(np.abs(mx) > np.abs(mn), mx, mn).astype(np.float32)
    scale = (-v / np.float32(128)).astype(np.float32)
    with np.errstate(divide="ignore"):
        recip = np.where(scale != 0, np.float32(1) / scale, np.float32(0)).astype(np.float32)
    x = (blk * recip[:, :, None]).astype(np.float32)          # float 곱
    x = (x + np.float32(128)).astype(np.float32)               # float 덧셈
    xd = x.astype(np.float64)
    r = np.sign(xd) * np.floor(np.abs(xd) + 0.5)               # roundf (0에서 먼 쪽)
    q = np.clip(r, 0, 255).astype(np.uint8).reshape(N, Kp)
    if Kp != K:
        q[:, K:] = 0  # 패딩 칸은 양자화기가 쓰지 않아 0으로 남는다
    return q, scale


# ───────── ONNX protobuf 최소 파서 (initializer의 파일 내 바이트 위치) ─────────
def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def _fields(b, start, end):
    i = start
    while i < end:
        tag, i = _varint(b, i)
        fn, wt = tag >> 3, tag & 7
        if wt == 0:
            v, i = _varint(b, i)
            yield fn, wt, v, None
        elif wt == 2:
            ln, i = _varint(b, i)
            yield fn, wt, i, i + ln
            i += ln
        elif wt == 1:
            yield fn, wt, i, i + 8
            i += 8
        elif wt == 5:
            yield fn, wt, i, i + 4
            i += 4
        else:
            raise ValueError(f"wire type {wt}")


def graph_initializers(b):
    """ModelProto(7=graph) → GraphProto(5=initializer) → TensorProto.
    반환: [{name, dtype, dims, raw:(start,end)|None, ext:{location,offset,length}|None}]"""
    g = next((s, e) for fn, wt, s, e in _fields(b, 0, len(b)) if fn == 7)
    out = []
    for fn, wt, s, e in _fields(b, *g):
        if fn != 5:
            continue
        t = {"dims": [], "raw": None, "ext": None, "dtype": None, "name": None}
        ext = {}
        for f2, w2, s2, e2 in _fields(b, s, e):
            if f2 == 1:
                if w2 == 0:
                    t["dims"].append(s2)
                else:  # packed
                    j = s2
                    while j < e2:
                        v, j = _varint(b, j)
                        t["dims"].append(v)
            elif f2 == 2:
                t["dtype"] = s2
            elif f2 == 8:
                t["name"] = bytes(b[s2:e2]).decode()
            elif f2 == 9:
                t["raw"] = (s2, e2)
            elif f2 == 13:
                kv = {}
                for f3, w3, s3, e3 in _fields(b, s2, e2):
                    kv[f3] = bytes(b[s3:e3]).decode()
                ext[kv[1]] = kv[2]
            elif f2 in (4, 5, 6, 7, 10, 11):
                raise ValueError(f"{t['name']}: typed data field {f2} (raw_data만 지원)")
        if ext:
            t["ext"] = {"location": ext["location"], "offset": int(ext.get("offset", 0)),
                        "length": int(ext["length"])}
        out.append(t)
    return out


def h(x):
    return hashlib.sha1(x).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="anima-base-v1.0.safetensors (템플릿 빌드에 쓴 체크포인트)")
    p.add_argument("--build", required=True, help="템플릿 빌드 폴더 (그래프 12개 + 샤드)")
    p.add_argument("--prefix", default="animav10", help="템플릿 빌드 접두사 (build_dit.bat의 prefix)")
    p.add_argument("--out", required=True, help="매핑표 JSON")
    p.add_argument("--check", action="store_true", help="매핑표만으로 베이스 샤드·그래프를 재조립해 바이트 대조")
    args = p.parse_args()
    STEMS.update(base=f"{args.prefix}_dit_dyn32", turbo=f"{args.prefix}_dit_turbo32")

    t0 = time.time()
    sd = read_safetensors(args.ckpt, skip=lambda k: ".llm_adapter." in k)
    print(f"[ckpt] DiT 텐서 {len(sd)}개 ({time.time() - t0:.1f}s)")

    # 후보 바이트 해시: (key, kind) — kind: f32 | f32T | f16 | f16T | q8B | q8S
    cand = {}

    def add(key, kind, data):
        cand.setdefault(h(data), []).append((key, kind))

    for k, w in sd.items():
        add(k, "f32", w.tobytes())
        add(k, "f16", w.astype(np.float16).tobytes())
        if w.ndim == 2:
            add(k, "f32T", np.ascontiguousarray(w.T).tobytes())
            add(k, "f16T", np.ascontiguousarray(w.T).astype(np.float16).tobytes())
            q, s = quant8(w)
            add(k, "q8B", q.tobytes())
            add(k, "q8S", s.tobytes())
    print(f"[cand] 후보 해시 {len(cand)}개 ({time.time() - t0:.1f}s)")

    def resolve(name, hits):
        if len(hits) == 1:
            return hits[0]
        # 같은 값의 텐서 여럿 — 이름으로 가른다 (dynamo는 직접 쓰는 파라미터를 'dit.<FQN>'으로 둔다)
        by_name = [x for x in hits if name.startswith("dit.") and x[0] == "net." + name[4:]]
        if len(by_name) == 1:
            return by_name[0]
        raise SystemExit(f"[모호] {name}: {hits} — 값만으로 체크포인트 키를 가를 수 없음")

    shard_cache = {}

    def shard_bytes(path):
        if path not in shard_cache:
            with open(path, "rb") as f:
                shard_cache[path] = f.read()
        return shard_cache[path]

    result = {"format": "anima2b-convert-map", "version": 1, "block": BLOCK, "stems": STEMS}
    used = {}  # key → set(kind)

    # ── 1) 샤드 배치 (베이스 _q8a4fns 기준, Turbo는 같은 배치인지 확인) ──
    layouts = {}
    for variant, stem in STEMS.items():
        gpath = os.path.join(args.build, stem + "_q8a4fns.onnx")
        b = open(gpath, "rb").read()
        inits = graph_initializers(b)
        layouts[variant] = [(t["name"], t["dtype"], tuple(t["dims"]), t["ext"]["location"].replace(stem, "{stem}"),
                             t["ext"]["offset"], t["ext"]["length"]) for t in inits if t["ext"]]
    assert layouts["base"] == layouts["turbo"], "베이스·Turbo 샤드 배치가 다름"
    print(f"[layout] 외부 initializer {len(layouts['base'])}개 — 베이스·Turbo 배치 동일")

    mani = json.load(open(os.path.join(args.build, STEMS["base"] + "_q8a4fns.onnx.manifest.json")))
    shard_names = [s.replace(STEMS["base"], "{stem}") for s in mani["shards"]]
    shard_sizes = [os.path.getsize(os.path.join(args.build, s)) for s in mani["shards"]]
    result["shards"] = [{"name": n, "size": z} for n, z in zip(shard_names, shard_sizes)]
    sidx = {n: i for i, n in enumerate(shard_names)}

    entries, consts = [], []
    for name, dtype, dims, loc, off, ln in layouts["base"]:
        si = sidx[loc]
        data_b = shard_bytes(os.path.join(args.build, loc.format(stem=STEMS["base"])))[off:off + ln]
        hits = cand.get(h(data_b))
        if hits:
            key, kind = resolve(name, hits)
            used.setdefault(key, set()).add(kind)
            entries.append({"name": name, "key": key, "kind": kind, "shard": si, "offset": off, "length": ln})
        else:
            data_t = shard_bytes(os.path.join(args.build, loc.format(stem=STEMS["turbo"])))[off:off + ln]
            if data_b != data_t:
                # Turbo LoRA가 바꾼 체크포인트 가중치일 수는 없다(베이스 체크포인트와 안 맞았으므로) → 미확인 가중치
                raise SystemExit(f"[미확인] {name} {dims}: 체크포인트와 안 맞고 베이스·Turbo가 다름")
            consts.append({"name": name, "shard": si, "offset": off, "length": ln, "dtype": dtype, "dims": list(dims),
                           "data": base64.b64encode(data_b).decode()})
    # q8B/q8S는 MatMulNBits 한 쌍 — 변환기는 B 항목에서 둘을 같이 만든다
    byk = {}
    for e in entries:
        byk.setdefault(e["key"], {})[e["kind"]] = e
    ent_out = []
    for k, d in byk.items():
        w = sd[k]
        if "q8B" in d:
            assert "q8S" in d and set(d) == {"q8B", "q8S"}, (k, list(d))
            N, K = w.shape
            ent_out.append({"key": k, "kind": "q8", "N": N, "K": K, "shard": d["q8B"]["shard"], "offset": d["q8B"]["offset"],
                            "sshard": d["q8S"]["shard"], "soffset": d["q8S"]["offset"]})
        else:
            assert len(d) == 1, (k, list(d))
            (kind, e), = d.items()
            ent_out.append({"key": k, "kind": kind, "shape": list(w.shape), "shard": e["shard"], "offset": e["offset"]})
    result["tensors"] = ent_out
    result["consts"] = consts
    cbytes = sum(c["length"] for c in consts)
    print(f"[shard] 체크포인트 텐서 {len(ent_out)}개 "
          f"({', '.join(f'{k} {sum(1 for e in ent_out if e['kind'] == k)}' for k in sorted({e['kind'] for e in ent_out}))}), "
          f"구조 상수 {len(consts)}개 {cbytes / 1024:.0f}KB")

    # ── 2) 그래프 인라인 initializer (12개 그래프 각각) ──
    graphs = {}
    for variant, stem in STEMS.items():
        for g in GRAPHS:
            fname = stem + g + ".onnx"
            b = open(os.path.join(args.build, fname), "rb").read()
            patches = []
            for t in graph_initializers(b):
                if t["ext"] or not t["raw"]:
                    continue
                s, e = t["raw"]
                hits = cand.get(h(b[s:e]))
                if not hits:
                    continue
                key, kind = resolve(t["name"], hits)
                used.setdefault(key, set()).add(kind)
                patches.append({"name": t["name"], "key": key, "kind": kind, "offset": s, "length": e - s})
            graphs[f"{variant}{g}"] = {"file": fname, "size": len(b), "sha1": h(b), "patches": patches}
            print(f"[graph] {fname}: 인라인 체크포인트 텐서 {len(patches)}개 "
                  f"({sorted({p['kind'] for p in patches})})")
    result["graphs"] = graphs

    missing = sorted(set(sd) - set(used))
    if missing:
        raise SystemExit(f"[누락] 그래프 어디에도 안 쓰인 체크포인트 텐서 {len(missing)}개: {missing[:10]}")
    result["keys"] = {k: list(w.shape) for k, w in sd.items()}
    print(f"[cover] 체크포인트 DiT 텐서 {len(sd)}개 전부 사용됨")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"[done] {args.out} ({os.path.getsize(args.out) / 1024:.0f}KB, {time.time() - t0:.1f}s)")

    if args.check:
        check(result, sd, args.build)


def tensor_bytes(w, kind):
    if kind == "f32":
        return w.tobytes()
    if kind == "f32T":
        return np.ascontiguousarray(w.T).tobytes()
    if kind == "f16":
        return w.astype(np.float16).tobytes()
    if kind == "f16T":
        return np.ascontiguousarray(w.T).astype(np.float16).tobytes()
    raise ValueError(kind)


def check(m, sd, build):
    """매핑표 + 체크포인트만으로 베이스 샤드·그래프를 재조립해 원본과 바이트 대조 (JS 변환기와 같은 절차의 파이썬판)."""
    t0 = time.time()
    shards = [bytearray(s["size"]) for s in m["shards"]]
    for e in m["tensors"]:
        w = sd[e["key"]]
        if e["kind"] == "q8":
            q, s = quant8(w)
            shards[e["shard"]][e["offset"]:e["offset"] + q.nbytes] = q.tobytes()
            shards[e["sshard"]][e["soffset"]:e["soffset"] + s.nbytes] = s.tobytes()
        else:
            d = tensor_bytes(w, e["kind"])
            shards[e["shard"]][e["offset"]:e["offset"] + len(d)] = d
    for c in m["consts"]:
        shards[c["shard"]][c["offset"]:c["offset"] + c["length"]] = base64.b64decode(c["data"])
    ok = True
    for i, s in enumerate(m["shards"]):
        ref = open(os.path.join(build, s["name"].format(stem=m["stems"]["base"])), "rb").read()
        same = ref == bytes(shards[i])
        ok &= same
        print(f"[check] 샤드 {i}: {'동일' if same else '다름'} ({len(ref)}B)")
    for gk, g in m["graphs"].items():
        if not gk.startswith("base"):
            continue
        ref = open(os.path.join(build, g["file"]), "rb").read()
        b = bytearray(ref)
        for p in g["patches"]:
            b[p["offset"]:p["offset"] + p["length"]] = b"\0" * p["length"]  # 지운 뒤 다시 채워 실제로 다시 쓰는지 확인
        for p in g["patches"]:
            b[p["offset"]:p["offset"] + p["length"]] = tensor_bytes(sd[p["key"]], p["kind"])
        same = bytes(b) == ref
        ok &= same
        print(f"[check] {g['file']}: {'동일' if same else '다름'}")
    print(f"[check] {'통과' if ok else '실패'} ({time.time() - t0:.1f}s)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
