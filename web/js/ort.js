// ONNX Runtime Web 로드 + fp16 변환·텐서 해제 헬퍼

// ── ORT 로드 ──
// JSPI 지원 브라우저(Chrome 137+): 네이티브 C++ WebGPU EP 빌드(jspi).
//   → DP4A MatMulNBits / FlashAttention 커널은 이 빌드에만 존재 (인수인계 사실 10, 14).
// 미지원: 기존 JSEP 빌드 폴백 (동작 동일, 최적화 커널 없음).
// EP 교체기라 latest 추적은 회귀 위험 → 버전 고정 (인수인계 '기타' 항목).
// ?ort=1.30.0 처럼 정확한 버전을 주면 그 버전으로 로드한다 (A/B 측정용, 기본은 고정 버전).
const ORT_PIN = "1.26.0"; // 고정 (EP 교체기라 latest 추적은 회귀 위험)
const ORT_Q = new URLSearchParams(location.search).get("ort");
const ORT_VER = ORT_Q && /^\d+\.\d+\.\d+$/.test(ORT_Q) ? ORT_Q : ORT_PIN;
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
const HAS_JSPI = typeof WebAssembly.Suspending === "function";
const ort = await import(
  ORT_BASE + (HAS_JSPI ? "ort.jspi.bundle.min.mjs" : "ort.webgpu.min.mjs")
);
ort.env.wasm.wasmPaths = ORT_BASE;
ort.env.wasm.proxy = false; // WebGPU EP는 메인 스레드 상주 (검증된 구성)
// ───────── fp16 ↔ fp32 ─────────
// 최신 ORT Web은 브라우저가 네이티브 Float16Array를 지원하면 fp16 텐서 .data를
// 비트패턴(Uint16Array)이 아닌 Float16Array(실제 float)로 반환한다. 양쪽 모두 대응.
const HAS_F16 = typeof Float16Array !== "undefined";

function toF32(data) {
  if (HAS_F16 && data instanceof Float16Array)
    return Float32Array.from(data);
  if (data instanceof Float32Array) return data;
  return f16ToF32(data); // Uint16Array 비트패턴
}

// ── 텐서 메모리 반환 헬퍼 (native WebGPU EP GPU 버퍼 누적 방지) ──
// ORT 빌드에 따라 dispose가 없을 수 있어 옵셔널 처리. 이미 놓인 텐서 재호출도 안전.
function disposeTensor(t) {
  try {
    if (t && typeof t.dispose === "function") t.dispose();
  } catch (_) {}
}
// run() 결과 객체({name: Tensor})의 모든 출력 텐서를 놓는다.
function disposeRun(out) {
  if (!out) return;
  for (const k in out) disposeTensor(out[k]);
}

function makeF16Tensor(f32, dims) {
  const payload = HAS_F16 ? new Float16Array(f32) : f32ToF16(f32);
  return new ort.Tensor("float16", payload, dims);
}

function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i],
      s = (h & 0x8000) << 16,
      e = (h >> 10) & 0x1f,
      m = h & 0x3ff;
    let bits;
    if (e === 0)
      bits =
        m === 0
          ? s
          : (() => {
              // subnormal
              let mm = m,
                ee = -1;
              do {
                mm <<= 1;
                ee++;
              } while (!(mm & 0x400));
              return s | ((127 - 15 - ee) << 23) | ((mm & 0x3ff) << 13);
            })();
    else if (e === 31) bits = s | 0x7f800000 | (m << 13);
    else bits = s | ((e - 15 + 127) << 23) | (m << 13);
    out[i] = new Float32Array(new Uint32Array([bits]).buffer)[0];
  }
  return out;
}
const _f32buf = new Float32Array(1),
  _u32buf = new Uint32Array(_f32buf.buffer);
function f32ToF16(f32) {
  const out = new Uint16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    _f32buf[0] = f32[i];
    const x = _u32buf[0],
      s = (x >>> 16) & 0x8000;
    const e = (x >>> 23) & 0xff;
    let m = x & 0x7fffff;
    if (e === 255) {
      out[i] = s | 0x7c00 | (m ? 0x200 : 0);
      continue;
    } // inf/nan
    const he = e - 127 + 15; // fp16 지수로 리바이어스
    if (he >= 31)
      out[i] = s | 0x7c00; // overflow → inf
    else if (he <= 0) {
      // subnormal/underflow
      if (he < -10) out[i] = s;
      else {
        m = (m | 0x800000) >> (1 - he); // implicit 1 복원 후 시프트
        out[i] = s | ((m + 0x1000) >> 13); // round-to-nearest
      }
    } else {
      out[i] = (s | (he << 10) | (m >> 13)) + ((m >> 12) & 1); // round
    }
  }
  return out;
}

export { HAS_F16, HAS_JSPI, disposeRun, disposeTensor, f16ToF32, f32ToF16, makeF16Tensor, ort, toF32 };
