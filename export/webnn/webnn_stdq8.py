# WebNN int8 그래프 — 운영 표준(WebGPU) DiT 샤드를 그대로 쓴다. WebNN 전용 가중치 다운로드가 없다(그래프 + 수십 KB).
#
#   map  : python -I export/webnn/webnn_stdq8.py map <…_q8a4fns_wnn1024.onnx> <…_q8a4fns.onnx> <map.json>
#          WebNN fp16 그래프의 가중치를 표준 그래프의 가중치와 값으로 짝짓는다(모델 변형마다 한 번, 두 그래프의 샤드가 옆에 있어야 함).
#          짝이 없는 작은 상수(RoPE 표 등)는 WebNN 그래프 옆의 <그래프 앞부분>_wnn1024q.extra.bin에 모은다(배포 대상).
#   apply: python -I export/webnn/webnn_stdq8.py apply <WebNN fp16 그래프.onnx> <map.json> [출력.onnx]
#          출력 이름을 빼면 입력 이름의 _wnn1024를 _wnn1024q로 바꾼다(_wnn1024n_sp → _wnn1024qn_sp — 페이지가 찾는 이름).
#          그래프만 편집. 같은 샤드를 쓰는 fp16 그래프(일반·NegPip·LoRA 슬롯·Spectrum 본체)는 가중치 이름이 같아 map 하나로 된다.
#          출력 옆에 .manifest.json(표준 샤드 + extra), 슬롯 그래프면 .lora_manifest.json도 복사.
#
# 표준 그래프의 선형층은 MatMulNBits(8비트, 블록 32, 영점 없음 = 128): B uint8 [N, K/32, 32], scales fp32 [N, K/32].
# WebNN에서는 매 실행 GPU에서 푼다: DequantizeLinear(B, 1.0, 128) → Mul(scales [N,K/32,1]) → Cast(fp16) → Reshape [N, K'] (→ Slice K) → Transpose → MatMul.
# DequantizeLinear를 스칼라 scale·영점으로 쓰는 이유는 webnn_q8.py --dq-cast와 같다(ORT Web이 로드 때 접어 wasm 힙을 넘기지 않게,
# Chrome WebNN이 블록 단위 DequantizeLinear를 거부해서). 그 밖의 가중치(노름·바이어스, t_embed 등 fp32로 남긴 선형층)는 Cast(fp16)만.
#
# 실측(2026-11-14, RTX 5070 Ti, Chrome 157 플래그, Turbo 1024² 정적, DiT 1스텝, 기준 = PyTorch fp32):
#   fp16 그래프 385.6ms, 오차 1.09 / 0.48 / 0.50% (σ 1 / 0.5 / 0.1) — 다운로드 3.64GiB
#   webnn_q8.py --dq-cast(자체 int8 샤드) 463.6ms, 2.50 / 1.13 / 0.96% — 2.05GiB
#   이 스크립트(표준 샤드) 466.6ms, 2.34 / 1.24 / 0.86% — 그래프 5.7MB + extra 75KB. 짝: MatMulNBits 454/454, fp32 87개, extra 3개.
#   Transpose 대신 Gemm(transB)도 같은 시간이라 Transpose로 둔다.
import sys, os, json, shutil, numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto as TP
from onnx.external_data_helper import load_external_data_for_tensor

QTOL, FTOL = 0.02, 2e-3   # 짝으로 인정하는 상대오차: 양자화 가중치 / fp32→fp16 가중치


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def ext_info(t):
    return {e.key: e.value for e in t.external_data}


def do_map(src, std, out):
    sdir = os.path.dirname(os.path.abspath(std))
    sm = onnx.load(std, load_external_data=False)
    sinit = {t.name: t for t in sm.graph.initializer}

    def sarr(name):
        t = onnx.TensorProto(); t.CopyFrom(sinit[name])
        if t.data_location == TP.EXTERNAL: load_external_data_for_tensor(t, sdir)
        return numpy_helper.to_array(t)

    def sref(name):
        t = sinit[name]; e = ext_info(t)
        assert t.data_location == TP.EXTERNAL, f"표준 가중치 {name}이 외부 데이터가 아니다"
        return {"name": name, "location": e["location"], "offset": int(e.get("offset", 0)), "length": int(e["length"]),
                "dims": list(t.dims), "dtype": t.data_type}

    mnb = []
    for n in sm.graph.node:
        if n.op_type != "MatMulNBits": continue
        a = {x.name: x.i for x in n.attribute}
        assert a["bits"] == 8 and a["block_size"] == 32 and (len(n.input) < 4 or not n.input[3]), "8비트·블록 32·영점 없음만 지원"
        B, S = sarr(n.input[1]), sarr(n.input[2])
        K, N = a["K"], a["N"]
        mnb.append({"K": K, "N": N, "nb": int(B.shape[1]), "B": sref(n.input[1]), "S": sref(n.input[2]),
                    "deq": ((B.astype(np.float32) - 128.0) * S.reshape(N, -1, 1)).reshape(N, -1)[:, :K].T})
    qn = {x for e in mnb for x in (e["B"]["name"], e["S"]["name"])}
    sf = {k: sarr(k).astype(np.float32) for k, t in sinit.items() if t.data_location == TP.EXTERNAL and k not in qn}

    ext_names = {t.name for t in onnx.load(src, load_external_data=False).graph.initializer if t.data_location == TP.EXTERNAL}
    m = onnx.load(src)
    weights, others, extra = {}, {}, []
    by_shape = {}
    for i, e in enumerate(mnb): by_shape.setdefault((e["K"], e["N"]), []).append(i)
    used, worst = set(), 0.0
    for t in m.graph.initializer:   # 양자화 가중치: 2D fp16, 같은 [K,N]의 MatMulNBits 중 가장 가까운 것
        if t.name not in ext_names or t.data_type != TP.FLOAT16 or len(t.dims) != 2: continue
        cands = [i for i in by_shape.get(tuple(t.dims), []) if i not in used]
        if not cands: continue
        W = numpy_helper.to_array(t).astype(np.float32)
        best = min(cands, key=lambda i: rel(mnb[i]["deq"][:64], W[:64]))
        r = rel(mnb[best]["deq"], W)
        if r > QTOL: continue
        used.add(best); worst = max(worst, r)
        e = mnb[best]
        weights[t.name] = {"K": e["K"], "N": e["N"], "nb": e["nb"], "B": e["B"], "S": e["S"], "rel": round(r, 5)}
    for t in m.graph.initializer:   # 그 밖의 외부 가중치: 표준의 fp32 가중치 중 같은 값
        if t.name not in ext_names or t.name in weights: continue
        a = numpy_helper.to_array(t).astype(np.float32)
        hit = next((k for k, v in sf.items() if v.size == a.size and rel(v.reshape(a.shape), a) < FTOL), None)
        if hit: others[t.name] = {"ref": sref(hit), "dims": list(t.dims), "dtype": t.data_type}
        else: extra.append(t)
    left = len(mnb) - len(used)
    assert left == 0, f"짝이 없는 표준 MatMulNBits {left}개 — 다른 체크포인트이거나 그래프 구조가 다르다"
    # extra 파일: WebNN 그래프 앞부분(_wnn1024 앞까지) + _wnn1024q.extra.bin
    base = os.path.basename(src)
    stem = base[: base.index("_wnn1024")] if "_wnn1024" in base else os.path.splitext(base)[0]
    xname = stem + "_wnn1024q.extra.bin"
    xpath = os.path.join(os.path.dirname(os.path.abspath(src)), xname)   # 그래프·샤드와 같은 폴더(배포 대상)
    xmap, off = {}, 0
    if extra:   # 동적 그래프는 RoPE 표를 실행 중에 계산해 짝 없는 상수가 없을 수 있다 — 그때는 파일을 만들지 않는다
        with open(xpath, "wb") as f:
            for t in extra:
                raw = numpy_helper.to_array(t).tobytes(); pad = (-off) % 64
                f.write(b"\0" * pad); off += pad; f.write(raw)
                xmap[t.name] = {"location": xname, "offset": off, "length": len(raw), "dims": list(t.dims), "dtype": t.data_type}
                off += len(raw)
    shards = sorted({v["B"]["location"] for v in weights.values()} | {v["S"]["location"] for v in weights.values()} | {v["ref"]["location"] for v in others.values()})
    json.dump({"std": os.path.basename(std), "shards": shards, "extra": xname if extra else None, "weights": weights, "others": others, "extra_tensors": xmap},
              open(out, "w", encoding="utf-8"))
    print(f"[webnn_stdq8 map] MatMulNBits {len(weights)}/{len(mnb)} 짝(역양자화 vs fp16 상대오차 최대 {worst * 100:.2f}%), "
          f"fp32 {len(others)}개, extra {len(extra)}개 {off / 2**10:.0f} KiB -> {out}")


def do_apply(src, mp, dst):
    M = json.load(open(mp, encoding="utf-8"))
    m = onnx.load(src, load_external_data=False)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    W, O, X = M["weights"], M["others"], M["extra_tensors"]

    def ext_tensor(name, info):
        t = onnx.TensorProto()
        t.name = name; t.data_type = info["dtype"]; t.dims.extend(info["dims"]); t.data_location = TP.EXTERNAL
        for k, v in (("location", info["location"]), ("offset", str(info["offset"])), ("length", str(info["length"]))):
            e = t.external_data.add(); e.key, e.value = k, v
        return t

    new_inits, pre, made, rename = [], [], set(), {}
    new_inits += [numpy_helper.from_array(np.array(1.0, np.float32), "sq_one"), numpy_helper.from_array(np.array(128, np.uint8), "sq_zp")]
    for name, w in W.items():
        if name not in inits: continue
        p = name + "_sq"
        for k in ("B", "S"):
            sn = "std::" + w[k]["name"]
            if sn not in made: new_inits.append(ext_tensor(sn, w[k])); made.add(sn)
        new_inits += [numpy_helper.from_array(np.array([w["N"], w["nb"], 1], np.int64), p + "_s3"),
                      numpy_helper.from_array(np.array([w["N"], w["nb"] * 32], np.int64), p + "_nk")]
        pre += [helper.make_node("DequantizeLinear", ["std::" + w["B"]["name"], "sq_one", "sq_zp"], [p + "_f32"], name=p + "_dq"),
                helper.make_node("Reshape", ["std::" + w["S"]["name"], p + "_s3"], [p + "_sc"], name=p + "_scr"),
                helper.make_node("Mul", [p + "_f32", p + "_sc"], [p + "_m"], name=p + "_mul"),
                helper.make_node("Cast", [p + "_m"], [p + "_h"], to=TP.FLOAT16, name=p + "_c16"),
                helper.make_node("Reshape", [p + "_h", p + "_nk"], [p + "_nkt"], name=p + "_r")]
        cur = p + "_nkt"
        if w["nb"] * 32 != w["K"]:   # 패딩된 K(예: x_embedder 68 → 96)
            new_inits += [numpy_helper.from_array(np.array([0], np.int64), p + "_k0"), numpy_helper.from_array(np.array([w["K"]], np.int64), p + "_k1"),
                          numpy_helper.from_array(np.array([1], np.int64), p + "_ax")]
            pre.append(helper.make_node("Slice", [cur, p + "_k0", p + "_k1", p + "_ax"], [p + "_sl"], name=p + "_slice")); cur = p + "_sl"
        pre.append(helper.make_node("Transpose", [cur], [p], perm=[1, 0], name=p + "_t"))
        rename[name] = p
    for name, o in O.items():
        if name not in inits: continue
        p = name + "_sq"; sn = "std::" + o["ref"]["name"]
        if sn not in made: new_inits.append(ext_tensor(sn, o["ref"])); made.add(sn)
        new_inits.append(numpy_helper.from_array(np.array(o["dims"], np.int64), p + "_shape"))
        pre += [helper.make_node("Reshape", [sn, p + "_shape"], [p + "_r"], name=p + "_rs"),
                helper.make_node("Cast", [p + "_r"], [p], to=o["dtype"], name=p + "_c")]
        rename[name] = p
    for n in g.node:
        for k, i in enumerate(n.input):
            if i in rename: n.input[k] = rename[i]
    keep = []
    for t in g.initializer:
        if t.name in rename: continue
        if t.name in X:   # 짝 없는 상수 → extra 파일
            keep.append(ext_tensor(t.name, X[t.name])); continue
        keep.append(t)
    missing = [t.name for t in keep if t.data_location == TP.EXTERNAL and ext_info(t)["location"] not in set(M["shards"]) | {M["extra"]} and not ext_info(t)["location"].endswith(".lora.bin")]
    assert not missing, f"map에 없는 가중치(fp16 샤드를 가리킴) {len(missing)}개: {missing[:5]} — 다른 샤드의 그래프인가?"
    del g.initializer[:]
    g.initializer.extend(keep + new_inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(pre + nodes)
    onnx.save_model(m, dst)
    json.dump({"shards": M["shards"] + ([M["extra"]] if M.get("extra") else [])}, open(dst + ".manifest.json", "w"))
    if os.path.exists(src + ".lora_manifest.json"):
        shutil.copy(src + ".lora_manifest.json", dst + ".lora_manifest.json")
    print(f"[webnn_stdq8 apply] 가중치 {len(rename)}개를 표준 샤드로, extra {sum(1 for t in keep if t.name in X)}개 -> {dst}")


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "map": do_map(*args)
    elif mode == "apply":
        if len(args) == 2:   # 출력 이름 생략: _wnn1024 → _wnn1024q (…_wnn1024n_sp.onnx → …_wnn1024qn_sp.onnx)
            d, b = os.path.split(args[0])
            assert "_wnn1024" in b, "출력 이름을 정하려면 입력 이름에 _wnn1024가 있어야 한다"
            args = [args[0], args[1], os.path.join(d, b.replace("_wnn1024", "_wnn1024q", 1))]
        do_apply(*args)
    else: sys.exit("usage: webnn_stdq8.py map|apply ...")
