// 커널 단위 프로파일 (전수조사 0.1) — 브라우저 콘솔에서 로드.
//   const p = await import("/__tools/profile.mjs");
//   await p.arm();                 // 페이지를 새로 연 직후, '모델 로드' 전에 (ORT wasm 초기화 전이어야 함)
//   (모델 로드 → 생성 1회)
//   const ev = await p.collect();  // endProfiling → 이벤트 배열 (세션 프로파일링은 여기서 끝난다)
//   p.summarize(ev)                // op/프로그램별 GPU 시간, CPU 배정 노드, 실행당 시간
//
// 경로 (ORT 1.26 jspi = 네이티브 C++ WebGPU EP — 확인됨):
// - env.webgpu.profiling은 JSEP 빌드 전용이라 이 빌드엔 없다. 세션 옵션 enableProfiling만 동작한다.
// - endProfiling() 때 ORT 프로파일러가 Chrome trace JSON을 콘솔(console.log)로 한 줄씩 출력한다.
//   cat "Node" = 노드별 CPU측 커널 호출 시간 + provider(EP 배정), cat "Api" = WebGPU 타임스탬프 쿼리로 잰
//   GPU 커널 시간(이름 "노드&op&셰이더프로그램", ts/dur µs), cat "Session" = model_run / SequentialExecutor::Execute.
// - 프로파일링 중에는 커널마다 compute pass를 나눠 타임스탬프를 찍는다. Chrome 기본 설정은 타임스탬프를
//   ~65µs 단위로 양자화하므로 작은 커널 개별값은 거칠다(합계·큰 커널만 신뢰).
// 페이지가 실제로 로드한 ORT(?ort= 버전 포함)와 같은 모듈 인스턴스
const ORT_URL = "/js/ort.js";

export async function arm(match = /_dit_/) {
  const { ort } = await import(ORT_URL);
  const P = (window.__prof ??= { sessions: [], collecting: false, lines: [] });
  if (ort.InferenceSession.__profArmed) return;
  // Emscripten은 wasm 초기화(첫 세션 생성) 때 console.log를 한 번 바인딩해 둔다 →
  // 수집 시점에 갈아끼우면 못 받는다. 초기화 전에 감싸 두고 collect() 동안만 모은다.
  const log = console.log;
  console.log = function (...a) {
    if (P.collecting) P.lines.push(a.map(String).join(" "));
    else log.apply(this, a);
  };
  const orig = ort.InferenceSession.create;
  ort.InferenceSession.create = async function (graph, opts = {}) {
    const hit = (opts.externalData || []).some((e) => match.test(e.path));
    const s = await orig.call(this, graph, hit ? { ...opts, enableProfiling: true } : opts);
    if (hit) P.sessions.push(s);
    return s;
  };
  ort.InferenceSession.__profArmed = true;
}

// 가장 최근에 만든 프로파일링 세션의 trace를 수집한다.
export async function collect(sess = window.__prof?.sessions.at(-1)) {
  if (!sess) throw new Error("프로파일링 세션 없음 — arm() 후 모델을 로드했는지 확인");
  const P = window.__prof;
  P.lines = [];
  P.collecting = true;
  try {
    await sess.endProfiling();
  } finally {
    P.collecting = false;
  }
  const ev = [];
  for (const s of P.lines) {
    if (!s.startsWith("{")) continue;
    try { ev.push(JSON.parse(s.replace(/,\s*$/, ""))); } catch { /* trace 외 줄 */ }
  }
  return ev;
}

const r1 = (x) => Math.round(x * 10) / 10;
// skip: 앞쪽 실행 몇 번을 제외 (첫 실행은 셰이더 특화 포함)
export function summarize(ev, { skip = 1 } = {}) {
  const runs = ev.filter((e) => e.cat === "Session" && e.name === "model_run").map((e) => e.dur / 1000);
  const execs = ev.filter((e) => e.cat === "Session" && e.name === "SequentialExecutor::Execute").map((e) => e.dur / 1000);
  const api = ev.filter((e) => e.cat === "Api"), node = ev.filter((e) => e.cat === "Node");
  const R = runs.length, per = api.length / R;
  if (!Number.isInteger(per)) throw new Error(`실행마다 GPU 이벤트 수가 다름 (${api.length}/${R})`);
  const steady = api.slice(per * skip), n = R - skip;
  const total = steady.reduce((a, e) => a + e.dur, 0);
  const group = (key) => {
    const m = new Map();
    for (const e of steady) {
      const k = key(e);
      if (k == null) continue;
      const g = m.get(k) ?? { calls: 0, us: 0 };
      g.calls++; g.us += e.dur;
      m.set(k, g);
    }
    return [...m].map(([k, g]) => ({ k, msPerRun: r1(g.us / n / 1000), pct: r1((100 * g.us) / total), callsPerRun: g.calls / n }))
      .sort((a, b) => b.msPerRun - a.msPerRun);
  };
  const [, op, prog] = [0, 1, 2];
  const parts = (e) => e.name.split("&");
  const cpu = node.filter((e) => e.args.provider !== "WebGpuExecutionProvider");
  const cpuOps = {};
  for (const e of cpu) cpuOps[e.args.op_name] = (cpuOps[e.args.op_name] || 0) + 1 / R;
  const cpuInputTypes = [...new Set(cpu.flatMap((e) => e.args.input_type_shape.map((t) => Object.keys(t)[0])))];
  return {
    runs: runs.map(r1), cpuDispatchMs: execs.map(r1),
    gpuMsPerRun: r1(total / n / 1000),
    gpuSpanMs: Array.from({ length: R }, (_, i) => { const s = api.slice(i * per, (i + 1) * per); return r1((s.at(-1).ts + s.at(-1).dur - s[0].ts) / 1000); }),
    byOp: group((e) => parts(e)[op]),
    byProgram: group((e) => parts(e)[op] + " / " + parts(e)[prog]),
    attention: group((e) => (parts(e)[op] === "MultiHeadAttention" ? parts(e)[0].replace(/\d+/g, "#") + " / " + parts(e)[prog] : null)),
    matmul: group((e) => (parts(e)[op] === "MatMulNBits" ? parts(e)[prog] + " " + e.args.shapes : null)),
    cpuNodesPerRun: Object.fromEntries(Object.entries(cpuOps).map(([k, v]) => [k, r1(v)])),
    cpuInputTypes, cpuMsPerRun: r1(cpu.reduce((a, e) => a + e.dur, 0) / R / 1000),
  };
}
