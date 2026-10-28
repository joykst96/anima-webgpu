// safetensors 체크포인트 → 페이지용 DiT 샤드·그래프 (ComfyUI 없이). Worker와 Node(tools/convert-check.mjs)가 같이 쓴다.
//
// 그래프 구조는 기존 빌드(animav10_*)를 템플릿으로 그대로 쓰고, 체크포인트에서 가중치 바이트만 다시 만들어
// 매핑표(web/convert/anima2b.map.json, export/make_convert_map.py 산출)가 가리키는 자리에 쓴다.
// 결과가 ComfyUI 파이프라인(export_dit → quantize_dit → … → shard)과 바이트 단위로 같도록 각 단계를 그대로 재현한다:
//   - bf16 → fp32: 정확 (상위 16비트)
//   - int8: ORT MatMulNBitsQuantizer(8bit 대칭, block 32) = MLAS MlasQuantizeBlockwise<float,8>
//   - Turbo LoRA 병합: ComfyUI ModelPatcher가 이 GPU에서 쓰는 lora_compute_dtype=fp16 경로
//     (w16 = fp16(w), d = fp32 순차합 B·A, w' = fp16(w16 + fp16(d)) → fp32 저장)

const f32b = new Float32Array(1);
const u32b = new Uint32Array(f32b.buffer);

// fp32 → fp16 비트 (최근접 짝수 반올림, 서브노멀 포함) — numpy astype(float16)·CUDA __float2half_rn과 같음
export function f16bits(x) {
  f32b[0] = x;
  const b = u32b[0];
  const sign = (b >>> 16) & 0x8000;
  let e = (b >>> 23) & 0xff;
  let m = b & 0x7fffff;
  if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  e = e - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    m |= 0x800000;
    const shift = 14 - e;
    let h = m >>> shift;
    const rem = m & ((1 << shift) - 1),
      mid = 1 << (shift - 1);
    if (rem > mid || (rem === mid && h & 1)) h++;
    return sign | h;
  }
  let h = (e << 10) | (m >>> 13);
  const rem = m & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && h & 1)) h++; // 올림이 지수로 넘어가도 비트 배치상 맞다
  return sign | h;
}

export function f16toF32(h) {
  const s = (h & 0x8000) << 16,
    e = (h >>> 10) & 0x1f,
    m = h & 0x3ff;
  if (e === 0) return (s ? -1 : 1) * m * 2 ** -24;
  if (e === 31) u32b[0] = s | 0x7f800000 | (m << 13);
  else u32b[0] = s | ((e - 15 + 127) << 23) | (m << 13);
  return f32b[0];
}

// ───────── 입력 ─────────
// File/Blob 또는 {url, size, name} — URL은 HTTP Range로 텐서 단위로 읽는다(수 GB 파일을 통째로 받지 않음).
export function sourceReader(src) {
  if (typeof Blob !== "undefined" && src instanceof Blob)
    return async (off, len) => new Uint8Array(await src.slice(off, off + len).arrayBuffer());
  return async (off, len) => {
    const r = await fetch(src.url, { headers: { range: `bytes=${off}-${off + len - 1}` } });
    if (r.status !== 206) {
      r.body?.cancel();
      throw new Error(`${src.name}: 서버가 Range 요청을 지원하지 않음 (HTTP ${r.status})`);
    }
    const b = new Uint8Array(await r.arrayBuffer());
    if (b.length !== len) throw new Error(`${src.name}: ${off}에서 ${len}B 요청, ${b.length}B 받음`);
    return b;
  };
}

export async function urlSource(url) {
  const r = await fetch(url, { headers: { range: "bytes=0-0" } });
  r.body?.cancel();
  const size = +(r.headers.get("content-range") || "").split("/")[1];
  if (r.status !== 206 || !size) throw new Error(`${url}: Range 요청 미지원 (HTTP ${r.status})`);
  return { url, size, name: decodeURIComponent(new URL(url).pathname.split("/").pop()) };
}

// ───────── safetensors ─────────
// read(offset, length) → Promise<Uint8Array>
export async function readSafetensorsHeader(read) {
  const head = await read(0, 8);
  const n = Number(new DataView(head.buffer, head.byteOffset, 8).getBigUint64(0, true));
  if (n <= 0 || n > 100 * 2 ** 20) throw new Error("safetensors 헤더가 아님");
  const hdr = JSON.parse(new TextDecoder().decode(await read(8, n)));
  const tensors = new Map();
  for (const [k, v] of Object.entries(hdr)) {
    if (k === "__metadata__") continue;
    tensors.set(k, { dtype: v.dtype, shape: v.shape, off: 8 + n + v.data_offsets[0], len: v.data_offsets[1] - v.data_offsets[0] });
  }
  return tensors;
}

export async function readTensorF32(read, t) {
  const bytes = await read(t.off, t.len);
  // safetensors 오프셋은 정렬 보장이 없다 → 정렬된 복사본으로
  const buf = bytes.byteOffset % 4 ? bytes.slice() : bytes;
  if (t.dtype === "F32") return new Float32Array(buf.buffer, buf.byteOffset, t.len / 4).slice();
  const u16 = new Uint16Array(buf.buffer, buf.byteOffset, t.len / 2);
  const out = new Float32Array(u16.length);
  if (t.dtype === "BF16") {
    const o32 = new Uint32Array(out.buffer);
    for (let i = 0; i < u16.length; i++) o32[i] = u16[i] << 16;
  } else if (t.dtype === "F16") {
    for (let i = 0; i < u16.length; i++) out[i] = f16toF32(u16[i]);
  } else throw new Error(`지원하지 않는 dtype ${t.dtype}`);
  return out;
}

// ───────── 키 대응 ─────────
// 매핑표 키는 'net.<모듈>'(anima-base 원본 형식). 체크포인트마다 앞부분이 다를 수 있다
// (ComfyUI 저장본 'model.diffusion_model.', 접두어 없음 등) → 't_embedding_norm.weight'로 앞부분을 찾는다.
export function ckptPrefix(tensors) {
  const hits = [...tensors.keys()].filter((k) => k.endsWith("t_embedding_norm.weight") && !k.includes("llm_adapter"));
  if (hits.length !== 1) throw new Error("Anima DiT 체크포인트가 아님 (t_embedding_norm.weight 없음)");
  return hits[0].slice(0, -"t_embedding_norm.weight".length);
}

export function checkCkpt(map, tensors, prefix) {
  const bad = [];
  for (const [k, shape] of Object.entries(map.keys)) {
    const t = tensors.get(prefix + k.slice(4));
    if (!t) bad.push(`${k} 없음`);
    else if (t.shape.join(",") !== shape.join(",")) bad.push(`${k} 모양 ${t.shape} ≠ ${shape}`);
    else if (!["BF16", "F16", "F32"].includes(t.dtype)) bad.push(`${k} dtype ${t.dtype}`);
  }
  if (bad.length) throw new Error(`Anima 2B(28블록) 구조와 다름: ${bad.slice(0, 5).join(", ")}${bad.length > 5 ? ` 외 ${bad.length - 5}개` : ""}`);
}

// LoRA 키 → 매핑표 키 ('blocks_0_self_attn_q_proj' → 'net.blocks.0.self_attn.q_proj.weight').
// 접미사·접두어 목록은 web/workers/lora-worker.js와 같다.
const LORA_SUFFIXES = [
  [".lora_down.weight", "down"], [".lora_up.weight", "up"], [".lora_A.weight", "down"], [".lora_B.weight", "up"],
  [".lora_down", "down"], [".lora_up", "up"], [".lora_A", "down"], [".lora_B", "up"], [".alpha", "alpha"],
];
const LORA_PREFIXES = ["lora_unet_", "diffusion_model.", "model.diffusion_model.", "transformer.", "lora_transformer_", "base_model.model.", "unet."];

export function loraModules(map, loraTensors) {
  const byUnderscore = new Map();
  for (const k of Object.keys(map.keys)) if (k.endsWith(".weight")) byUnderscore.set(k.slice(4, -7).replace(/\./g, "_"), k);
  const mods = new Map();
  let skipped = 0;
  for (const [k, t] of loraTensors) {
    let role = null,
      base = k;
    for (const [suf, r] of LORA_SUFFIXES)
      if (k.endsWith(suf)) {
        base = k.slice(0, -suf.length);
        role = r;
        break;
      }
    if (!role) {
      skipped++;
      continue;
    }
    for (const p of LORA_PREFIXES)
      if (base.startsWith(p)) {
        base = base.slice(p.length);
        break;
      }
    const key = byUnderscore.get(base.replace(/\./g, "_"));
    if (!key) {
      skipped++; // llm_adapter 등 DiT 그래프 밖 모듈
      continue;
    }
    const m = mods.get(key) || {};
    m[role] = t;
    mods.set(key, m);
  }
  for (const [k, m] of mods) if (!m.up || !m.down) throw new Error(`LoRA ${k}: up/down 짝이 없음`);
  return { mods, skipped };
}

// ComfyUI LoRA 병합 (weight_adapter/lora.py calculate_weight + ModelPatcher.patch_weight_to_device, lora_compute_dtype=fp16).
// w: [N,K] fp32 (제자리), up: [N,r], down: [r,K]. scale = strength · alpha/r (파이썬 double → 커널에서 fp32).
export function mergeLoraFp16(w, N, K, up, down, r, scale) {
  const acc = new Float32Array(K);
  const s32 = Math.fround(scale);
  for (let n = 0; n < N; n++) {
    acc.fill(0);
    // 원소마다 k = 0..r-1 순서로 fp32 누적 (Float32Array 저장이 매번 fp32 반올림). bf16×bf16 곱은 fp32에서 정확 → FMA와 같다.
    for (let k = 0; k < r; k++) {
      const b = up[n * r + k],
        o = k * K;
      for (let j = 0; j < K; j++) acc[j] += b * down[o + j];
    }
    const row = n * K;
    for (let j = 0; j < K; j++) {
      const d = s32 === 1 ? acc[j] : Math.fround(s32 * acc[j]);
      const w16 = f16toF32(f16bits(w[row + j])),
        d16 = f16toF32(f16bits(d));
      w[row + j] = f16toF32(f16bits(w16 + d16)); // fp16 + fp16 은 double에서 정확 → 한 번 반올림
    }
  }
}

// ───────── MLAS 8bit 대칭 블록 양자화 ─────────
// w: Linear weight [N,K] (= MatMul B[K,N]의 전치 — MLAS 열 방향 블록이 곧 w의 행 방향).
// → B uint8 [N, nb*32] (패딩 칸 0), scales fp32 [N, nb]
export function quant8(w, N, K, block = 32) {
  const nb = Math.ceil(K / block),
    Kp = nb * block;
  const q = new Uint8Array(N * Kp),
    s = new Float32Array(N * nb);
  for (let n = 0; n < N; n++) {
    const row = n * K;
    for (let b = 0; b < nb; b++) {
      const k0 = b * block,
        k1 = Math.min(k0 + block, K);
      let mn = 3.4028234663852886e38,
        mx = -3.4028234663852886e38;
      for (let k = k0; k < k1; k++) {
        const v = w[row + k];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      const v = Math.abs(mx) > Math.abs(mn) ? mx : mn;
      const scale = Math.fround(-v / 128);
      s[n * nb + b] = scale;
      const recip = scale ? Math.fround(1 / scale) : 0;
      const qo = n * Kp;
      for (let k = k0; k < k1; k++) {
        const x = Math.fround(Math.fround(w[row + k] * recip) + 128);
        let r = x >= 0 ? Math.floor(x + 0.5) : -Math.floor(-x + 0.5); // roundf: 0에서 먼 쪽
        q[qo + k] = r < 0 ? 0 : r > 255 ? 255 : r;
      }
    }
  }
  return { q, s };
}

export function transpose(w, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) out[j * rows + i] = w[i * cols + j];
  return out;
}

function toF16Bytes(w) {
  const o = new Uint16Array(w.length);
  for (let i = 0; i < w.length; i++) o[i] = f16bits(w[i]);
  return new Uint8Array(o.buffer);
}

// kind(매핑표) → 그래프에 들어가는 바이트
export function tensorBytes(w, shape, kind) {
  switch (kind) {
    case "f32":
      return new Uint8Array(w.buffer, w.byteOffset, w.byteLength);
    case "f16":
      return toF16Bytes(w);
    case "f32T":
      return new Uint8Array(transpose(w, shape[0], shape[1]).buffer);
    case "f16T":
      return toF16Bytes(transpose(w, shape[0], shape[1]));
  }
  throw new Error(`알 수 없는 kind ${kind}`);
}

function b64(s) {
  const bin = atob(s),
    u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}

// shard_onnx_data.py의 json.dump(manifest)와 같은 바이트
export function shardManifest(names) {
  return `{"shards": [${names.map((n) => JSON.stringify(n)).join(", ")}]}`;
}

// 변환 작업 계획: 텐서 하나 = 작업 하나 (체크포인트 파일 순서 — 디스크 순차 읽기).
//   job = { key, shape, t(체크포인트 위치), e(샤드 항목|null), inline(그래프 인라인 패치에 필요), lora({up,down,alpha,r}|null) }
export async function planConvert(map, read, loraRead) {
  const tensors = await readSafetensorsHeader(read);
  const prefix = ckptPrefix(tensors);
  checkCkpt(map, tensors, prefix);
  const ck = (k) => tensors.get(prefix + k.slice(4));
  let mods = null;
  if (loraRead) {
    mods = loraModules(map, await readSafetensorsHeader(loraRead)).mods;
    if (!mods.size) throw new Error("LoRA에 DiT 모듈이 없음");
    for (const [key, m] of mods) {
      const [N, K] = map.keys[key] || [];
      const r = m.down.shape[0];
      if (map.keys[key].length !== 2 || m.up.shape.join() !== `${N},${r}` || m.down.shape.join() !== `${r},${K}`)
        throw new Error(`LoRA ${key} 모양 불일치`);
    }
  }
  const inlineKeys = new Set();
  for (const g of Object.values(map.graphs)) for (const p of g.patches) inlineKeys.add(p.key);
  const entry = new Map(map.tensors.map((e) => [e.key, e]));
  const jobs = [...new Set([...entry.keys(), ...inlineKeys])].map((key) => {
    const m = mods?.get(key);
    return {
      key,
      shape: map.keys[key],
      t: ck(key),
      e: entry.get(key) || null,
      inline: inlineKeys.has(key),
      lora: m ? { up: m.up, down: m.down, alpha: m.alpha || null, r: m.down.shape[0] } : null,
    };
  });
  jobs.sort((a, b) => a.t.off - b.t.off);
  return { jobs, loraModules: mods ? mods.size : 0 };
}

// 작업 하나 계산 (Worker 풀에서 병렬로 돈다 — 입력은 직렬화 가능한 값만).
//   variants: ["base"] | ["base","turbo"] | ["turbo"],  strength: Turbo LoRA 세기
//   반환 { writes: {variant: [[shard, offset, bytes], ...]}, inline: {variant: Float32Array} }
export async function computeJob(job, { read, loraRead, variants, strength = 1.0, block = 32 }) {
  const w = await readTensorF32(read, job.t);
  const vals = { base: w, turbo: w };
  if (variants.includes("turbo") && job.lora) {
    const { up, down, alpha, r } = job.lora;
    const [N, K] = job.shape;
    const a = alpha ? (await readTensorF32(loraRead, alpha))[0] / r : 1.0;
    vals.turbo = w.slice();
    mergeLoraFp16(vals.turbo, N, K, await readTensorF32(loraRead, up), await readTensorF32(loraRead, down), r, strength * a);
  }
  const writes = {},
    inline = {},
    qcache = new Map();
  for (const v of variants) {
    const wv = vals[v];
    if (job.inline) inline[v] = wv;
    const e = job.e;
    if (!e) continue;
    if (e.kind === "q8") {
      // LoRA가 안 건드린 층은 베이스 결과를 그대로 쓴다
      let r = qcache.get(wv);
      if (!r) qcache.set(wv, (r = quant8(wv, e.N, e.K, block)));
      writes[v] = [
        [e.shard, e.offset, r.q],
        [e.sshard, e.soffset, new Uint8Array(r.s.buffer)],
      ];
    } else writes[v] = [[e.shard, e.offset, tensorBytes(wv, e.shape, e.kind)]];
  }
  return { writes, inline };
}

// 변환 본체.
//   map: 매핑표, read: 체크포인트 읽기, lora: {read, strength} | null (Turbo)
//   outs: { base?: Writer, turbo?: Writer }  Writer = { write(shardIdx, offset, bytes) }
//   compute: (job) → computeJob 결과 (기본: 이 스레드에서 계산. 브라우저는 Worker 풀), concurrency: 동시 작업 수
//   반환: { patches: {graphKey: [{offset,length,bytes}]}, loraModules } — 인라인 패치는 템플릿을 받는 쪽이 적용
export async function convertDit({ map, read, lora, outs, onProgress = () => {}, compute, concurrency = 1 }) {
  const variants = Object.keys(outs);
  if (variants.includes("turbo") && !lora) throw new Error("Turbo 변환에는 LoRA가 필요");
  const plan = await planConvert(map, read, variants.includes("turbo") ? lora.read : null);
  compute ??= (job) => computeJob(job, { read, loraRead: lora?.read, variants, strength: lora?.strength, block: map.block });

  // 구조 상수 (LayerNorm 1.0, RoPE 표 등)
  for (const c of map.consts) {
    const bytes = b64(c.data);
    for (const v of variants) await outs[v].write(c.shard, c.offset, bytes);
  }

  const inlineVals = { base: new Map(), turbo: new Map() };
  const totalBytes = plan.jobs.reduce((a, j) => a + j.t.len, 0);
  let done = 0,
    next = 0;
  const runner = async () => {
    for (let i; (i = next++) < plan.jobs.length; ) {
      const job = plan.jobs[i];
      const r = await compute(job);
      for (const v of variants) {
        for (const [shard, off, bytes] of r.writes[v] || []) await outs[v].write(shard, off, bytes);
        if (r.inline[v]) inlineVals[v].set(job.key, r.inline[v]);
      }
      done += job.t.len;
      onProgress(done / totalBytes, job.key);
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, concurrency) }, runner));

  // 그래프별 인라인 패치 바이트
  const patches = {};
  for (const [gk, g] of Object.entries(map.graphs)) {
    const v = gk.startsWith("turbo") ? "turbo" : "base";
    if (!outs[v]) continue;
    patches[gk] = g.patches.map((p) => ({
      offset: p.offset,
      length: p.length,
      bytes: tensorBytes(inlineVals[v].get(p.key), map.keys[p.key], p.kind),
    }));
  }
  return { patches, loraModules: plan.loraModules };
}

// 템플릿 그래프에 인라인 패치 적용 (길이 고정 — protobuf 구조는 그대로)
export function patchGraph(template, patches) {
  const out = template.slice();
  for (const p of patches) {
    if (p.bytes.length !== p.length) throw new Error("패치 길이 불일치");
    out.set(p.bytes, p.offset);
  }
  return out;
}
