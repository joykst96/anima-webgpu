// LoRA 변환 워커: safetensors 파싱 + 슬롯 레이아웃 배치(단일) / ΔW SVD 병합(다중)
let svdDecompose = null;
const svdReady = import('https://cdn.jsdelivr.net/npm/svd-js@1.1.1/build-es/svd.js').then(m => { svdDecompose = m.default; });

function f16ToF32(u16){const out=new Float32Array(u16.length);for(let i=0;i<u16.length;i++){const h=u16[i],s=(h&0x8000)<<16,e=(h>>10)&0x1f,m=h&0x3ff;let bits;if(e===0)bits=m===0?s:(()=>{let mm=m,ee=-1;do{mm<<=1;ee++;}while(!(mm&0x400));return s|((127-15-ee)<<23)|((mm&0x3ff)<<13);})();else if(e===31)bits=s|0x7f800000|(m<<13);else bits=s|((e-15+127)<<23)|(m<<13);out[i]=new Float32Array(new Uint32Array([bits]).buffer)[0];}return out;}

// safetensors: [u64 헤더길이][JSON 헤더][텐서 데이터]
function parseSafetensors(arrayBuffer) {
  const dv = new DataView(arrayBuffer);
  const hlen = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(arrayBuffer, 8, hlen)));
  const base = 8 + hlen;
  const tensors = new Map();
  for (const [k, v] of Object.entries(header)) {
    if (k === '__metadata__') continue;
    tensors.set(k, { dtype: v.dtype, shape: v.shape,
                     off: base + v.data_offsets[0], len: v.data_offsets[1] - v.data_offsets[0] });
  }
  return { tensors, buffer: arrayBuffer };
}

function readTensorF32(srcFile, t) {
  // slice 복사로 TypedArray 정렬 보장 (safetensors 오프셋은 정렬 미보장)
  const bytes = srcFile.buffer.slice(t.off, t.off + t.len);
  if (t.dtype === 'F32') return new Float32Array(bytes);
  const u16 = new Uint16Array(bytes);
  if (t.dtype === 'BF16') {
    const out = new Float32Array(u16.length);
    const u32 = new Uint32Array(1), f32v = new Float32Array(u32.buffer);
    for (let i = 0; i < u16.length; i++) { u32[0] = u16[i] << 16; out[i] = f32v[0]; }
    return out;
  }
  if (t.dtype === 'F16') return f16ToF32(u16);
  throw new Error(`지원하지 않는 dtype: ${t.dtype}`);
}

const LORA_SUFFIXES = [
  ['.lora_down.weight', 'down'], ['.lora_up.weight', 'up'],
  ['.lora_A.weight', 'down'], ['.lora_B.weight', 'up'],
  ['.lora_down', 'down'], ['.lora_up', 'up'],
  ['.lora_A', 'down'], ['.lora_B', 'up'], ['.alpha', 'alpha'],
];
const LORA_PREFIXES = ['lora_unet_', 'diffusion_model.', 'model.diffusion_model.',
                       'transformer.', 'lora_transformer_', 'base_model.model.', 'unet.'];

function classifyLoraKey(k) {
  if (k.startsWith('lora_te')) return { skip: 'te' };
  let role = null, base = k;
  for (const [suf, r] of LORA_SUFFIXES)
    if (k.endsWith(suf)) { base = k.slice(0, -suf.length); role = r; break; }
  if (!role) return { skip: 'unknown' };              // lokr_* 등
  for (const p of LORA_PREFIXES)
    if (base.startsWith(p)) { base = base.slice(p.length); break; }
  return { mod: base.replace(/\./g, '_'), role };     // blocks_0_self_attn_q_proj
}

// 멀티 LoRA를 manifest 슬롯 레이아웃의 fp32 버퍼로 병합 변환.
// 모듈별로 ΔW = Σ_i s_i·up_i·down_i 를 rank R로 정확 분해 (ΔW 미실체화,
// 스택 [up...]·[down...] 의 2단 그람 고유분해 thin SVD). A(K,R)=V·√Σ 형태가 아니라
// 슬롯 규약(y += B(Ax))에 맞춰 A=Vᵀ스케일, B=UΣ 로 재구성.

// 한 모듈의 LoRA 기여 목록 → (A: K×R, B: R×N) Float32. svd-js 2단 thin-SVD.
// ΔW = M(N×rt)·P(rt×K) 를 rank R로 정확 절단 분해. ΔW 미실체화.
function mergeModule(contribs, K, N, R) {
  const rt = contribs.reduce((s, c) => s + c.r, 0);
  if (rt === 0) return null;
  // M (N×rt) = [up_i·s_i],  P (rt×K) = [down_i]
  const M = Array.from({ length: N }, () => new Array(rt).fill(0));
  const P = Array.from({ length: rt }, () => new Float64Array(K));
  let col = 0;
  for (const c of contribs) {
    for (let n = 0; n < N; n++) for (let j = 0; j < c.r; j++) M[n][col + j] = c.up[n * c.r + j] * c.scale;
    for (let j = 0; j < c.r; j++) for (let k = 0; k < K; k++) P[col + j][k] = c.down[j * K + k];
    col += c.r;
  }
  // (1) M = Um(N×rt) Σm Wmᵀ.  svd-js는 m≥n 요구 — N≥rt는 일반 성립.
  const { u: Um, v: Wm, q: Sm } = svdDecompose(M, true, true);
  // (2) Y = Σm·Wmᵀ·P  (rt×K)
  const Y = [];
  for (let i = 0; i < rt; i++) {
    const row = new Array(K).fill(0);
    for (let k = 0; k < K; k++) { let s = 0; for (let j = 0; j < rt; j++) s += Wm[j][i] * P[j][k]; row[k] = Sm[i] * s; }
    Y.push(row);
  }
  // Yᵀ(K×rt) SVD → u=Vk(K×rt: ΔW의 우특이), v=Wy(rt×rt), q=Sy
  const Yt = [];
  for (let k = 0; k < K; k++) { const r = new Array(rt); for (let i = 0; i < rt; i++) r[i] = Y[i][k]; Yt.push(r); }
  const { u: Vk, v: Wy, q: Sy } = svdDecompose(Yt, true, true);
  // 상위 R개 (svd-js q 미정렬)
  const idx = [...Sy.keys()].sort((a, b) => Sy[b] - Sy[a]).slice(0, R);
  const A = new Float32Array(K * R), B = new Float32Array(R * N);
  // ΔW = (Um·Wy)·Σy·Vkᵀ → B = (Um·Wy·Σy)ᵀ, A = Vk
  for (let jj = 0; jj < idx.length; jj++) {
    const c = idx[jj], sv = Sy[c];
    if (sv < 1e-12) continue;
    for (let k = 0; k < K; k++) A[k * R + jj] = Vk[k][c];
    for (let n = 0; n < N; n++) { let s = 0; for (let r = 0; r < rt; r++) s += Um[n][r] * Wy[r][c]; B[jj * N + n] = s * sv; }
  }
  return { A, B };
}

// 단일 LoRA: SVD 불필요. down/up을 슬롯 레이아웃에 직접 배치 (scale 미적용 —
// 강도는 그래프 입력 lora_scale이 런타임에 곱하므로 재컴파일 없이 슬라이더 즉시 반영).
// alpha/r0 정규화만 B에 흡수. r0>R이면 상위 R랭크만 (절단, 드묾).
function convertLoraSingle(item, manifest) {
  const R = manifest.rank;
  const total = manifest.entries.reduce((m, e) => Math.max(m, e.B.offset + R * e.N * 4), 0);
  const out = new Float32Array(total / 4);
  const mods = new Map(); let skipTe = 0, skipUnknown = 0;
  for (const [k, t] of item.src.tensors) {
    const c = classifyLoraKey(k);
    if (c.skip) { c.skip === 'te' ? skipTe++ : skipUnknown++; continue; }
    if (!mods.has(c.mod)) mods.set(c.mod, {});
    mods.get(c.mod)[c.role] = t;
  }
  let applied = 0, missing = 0, mismatch = 0, trunc = 0; const seen = new Set();
  let _done = 0; const _tot = manifest.entries.length;
  for (const e of manifest.entries) {
    if ((++_done & 63) === 0) self.postMessage({ type:'progress', value: 0.1 + 0.85*_done/_tot });
    const km = e.key.match(/^b(\d+)_(.+)$/);
    const srcMod = `blocks_${km[1]}_${km[2]}`;
    const ent = mods.get(srcMod);
    if (!ent || !ent.down || !ent.up) { missing++; continue; }
    const r0 = ent.down.shape[0], K = ent.down.shape[1], N = ent.up.shape[0];
    if (K !== e.K || N !== e.N || ent.up.shape[1] !== r0) { mismatch++; continue; }
    seen.add(srcMod);
    const down = readTensorF32(item.src, ent.down);   // (r0, K)
    const up = readTensorF32(item.src, ent.up);       // (N, r0)
    const alpha = ent.alpha ? readTensorF32(item.src, ent.alpha)[0] : r0;
    const scale = alpha / r0;                          // 강도(weight)는 lora_scale이 처리 — 여기선 제외
    const rUse = Math.min(r0, R);
    if (r0 > R) trunc++;
    const aBase = e.A.offset / 4, bBase = e.B.offset / 4;
    // A(K,R) = downᵀ 제로패딩,  B(R,N) = upᵀ·scale 제로패딩
    for (let j = 0; j < rUse; j++) { const dRow = j * K; for (let k = 0; k < K; k++) out[aBase + k * R + j] = down[dRow + k]; }
    for (let n = 0; n < N; n++) { const uRow = n * r0; for (let j = 0; j < rUse; j++) out[bBase + j * N + n] = up[uRow + j] * scale; }
    applied++;
  }
  const notes = [];
  if (missing) notes.push(`빈 슬롯 ${missing}`);
  const extra = [...mods.keys()].filter(x => !seen.has(x)).length;
  if (extra) notes.push(`슬롯 밖 모듈 ${extra}`);
  if (skipTe) notes.push(`TE 키 ${skipTe} 무시`);
  if (skipUnknown) notes.push(`미지원 키 ${skipUnknown}`);
  if (trunc) notes.push(`랭크 절단 ${trunc}`);
  if (mismatch) notes.push(`차원 불일치 ${mismatch}`);
  return { bytes: new Uint8Array(out.buffer),
           report: { applied, total: manifest.entries.length, notes: notes.join(' · ') } };
}

function convertLoraMulti(items, manifest) {
  const R = manifest.rank;
  const total = manifest.entries.reduce((m, e) => Math.max(m, e.B.offset + R * e.N * 4), 0);
  const out = new Float32Array(total / 4);
  // 각 LoRA를 모듈맵으로 미리 분해
  const perLora = items.map(it => {
    const mods = new Map(); let skipTe = 0, skipUnknown = 0;
    for (const [k, t] of it.src.tensors) {
      const c = classifyLoraKey(k);
      if (c.skip) { c.skip === 'te' ? skipTe++ : skipUnknown++; continue; }
      if (!mods.has(c.mod)) mods.set(c.mod, {});
      mods.get(c.mod)[c.role] = t;
    }
    return { it, mods, skipTe, skipUnknown, seen: new Set() };
  });
  let appliedSlots = 0, missingAll = 0, mismatch = 0, trunc = 0;
  let _done = 0; const _tot = manifest.entries.length;
  for (const e of manifest.entries) {
    if ((++_done & 31) === 0) self.postMessage({ type:'progress', value: 0.1 + 0.85*_done/_tot });
    const km = e.key.match(/^b(\d+)_(.+)$/);
    const srcMod = `blocks_${km[1]}_${km[2]}`;
    const contribs = [];
    for (const pl of perLora) {
      const ent = pl.mods.get(srcMod);
      if (!ent || !ent.down || !ent.up) continue;
      const r0 = ent.down.shape[0], K = ent.down.shape[1], N = ent.up.shape[0];
      if (K !== e.K || N !== e.N || ent.up.shape[1] !== r0) { mismatch++; continue; }
      pl.seen.add(srcMod);
      const alpha = ent.alpha ? readTensorF32(pl.it.src, ent.alpha)[0] : r0;
      contribs.push({ r: r0, down: readTensorF32(pl.it.src, ent.down),
                      up: readTensorF32(pl.it.src, ent.up),
                      scale: (alpha / r0) * pl.it.weight });
      if (r0 > R) trunc++;
    }
    if (contribs.length === 0) { missingAll++; continue; }
    const merged = mergeModule(contribs, e.K, e.N, R);
    if (!merged) { missingAll++; continue; }
    const aBase = e.A.offset / 4, bBase = e.B.offset / 4;
    out.set(merged.A, aBase);
    out.set(merged.B, bBase);
    appliedSlots++;
  }
  const notes = [];
  if (missingAll) notes.push(`빈 슬롯 ${missingAll}`);
  let teTot = 0, unkTot = 0, extraTot = 0;
  for (const pl of perLora) {
    teTot += pl.skipTe; unkTot += pl.skipUnknown;
    extraTot += [...pl.mods.keys()].filter(m => !pl.seen.has(m)).length;
  }
  if (extraTot) notes.push(`슬롯 밖 모듈 ${extraTot}`);
  if (teTot) notes.push(`TE 키 ${teTot} 무시`);
  if (unkTot) notes.push(`미지원 키 ${unkTot}`);
  if (trunc) notes.push(`랭크 절단 ${trunc}`);
  if (mismatch) notes.push(`차원 불일치 ${mismatch}`);
  return { bytes: new Uint8Array(out.buffer),
           report: { applied: appliedSlots, total: manifest.entries.length, notes: notes.join(' · ') } };
}

self.onmessage = async (ev) => {
  const { manifest, loras } = ev.data;
  try {
    await svdReady;
    const items = loras.map(l => ({ src: parseSafetensors(l.buffer), weight: l.weight, fileName: l.fileName }));
    self.postMessage({ type:'progress', value: 0.05 });
    // 단일 LoRA는 SVD 없이 직접 배치 (강도는 그래프 lora_scale) → 즉시 + 정확
    const { bytes, report } = (items.length === 1)
      ? convertLoraSingle(items[0], manifest)
      : convertLoraMulti(items, manifest);
    self.postMessage({ type:'done', bytes, report }, [bytes.buffer]);
  } catch (e) { self.postMessage({ type:'error', message: e.message }); }
};
