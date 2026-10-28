// 로컬 검증용 dev 서버 (의존성 없음).
//   web/ 를 정적 서빙하고, 가중치(out/)·토크나이저(tokenizers/)는 운영 서버로 프록시한다.
//   localhost라 secure context 예외가 적용돼 WebGPU/OPFS가 동작한다.
//   프록시는 GET/HEAD만 통과시킨다 — 운영 서버 상태를 바꾸는 요청은 보내지 않는다.
//
//   node tools/devserver.mjs [--port 5173] --upstream <가중치 서버 http://host:port> [--local-out out-local]
//   --upstream: 없으면 환경변수 ANIMA_UPSTREAM. 둘 다 없으면 프록시 경로는 502 (로컬 파일·--local-out만 서빙).
//   --local-out: /out/<경로>가 이 디렉터리에 있으면 프록시 대신 로컬 파일을 준다
//                (운영 서버에 올리기 전 새 그래프를 검증할 때. 없으면 평소처럼 프록시).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : def;
};
const PORT = +opt("--port", 5173);
const UPSTREAM_ARG = opt("--upstream", process.env.ANIMA_UPSTREAM);
const UPSTREAM = UPSTREAM_ARG ? new URL(UPSTREAM_ARG) : null;
const TOOLS = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(TOOLS, "../web");
const PROXY_PREFIXES = ["/out/", "/tokenizers/"];
const LOCAL_OUT = opt("--local-out", null) && path.resolve(opt("--local-out"));

// --local-out에 같은 경로의 파일이 있으면 그 경로를 돌려준다
function localOverride(req) {
  if (!LOCAL_OUT || !req.url.startsWith("/out/")) return null;
  const rel = decodeURIComponent(new URL(req.url, "http://x").pathname).slice("/out/".length);
  const file = path.resolve(LOCAL_OUT, rel);
  return file.startsWith(LOCAL_OUT + path.sep) && fs.existsSync(file) && fs.statSync(file).isFile() ? file : null;
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
};

function proxy(req, res) {
  if (req.method !== "GET" && req.method !== "HEAD") {
    res.writeHead(405).end("proxy: GET/HEAD only");
    return;
  }
  if (!UPSTREAM) {
    res.writeHead(502).end("proxy: upstream 미설정 (--upstream 또는 ANIMA_UPSTREAM)");
    return;
  }
  const up = http.request(
    {
      host: UPSTREAM.hostname,
      port: UPSTREAM.port || 80,
      path: req.url,
      method: req.method,
      headers: {
        accept: req.headers.accept ?? "*/*",
        ...(req.headers.range ? { range: req.headers.range } : {}),
      },
    },
    (r) => {
      res.writeHead(r.statusCode, r.headers);
      r.pipe(res);
    },
  );
  up.on("error", (e) => {
    if (!res.headersSent) res.writeHead(502);
    res.end(`proxy error: ${e.message}`);
  });
  up.end();
}

// PUT /__tools/ref/<이름>.png — 하네스의 기준 이미지 저장 (tools/ref/, git 미포함).
// PUT /__tools/golden.json — 하네스의 이 환경 골든 해시 (h.saveGolden). 그 외 쓰기는 없다.
function saveRef(req, res) {
  const urlPath = new URL(req.url, "http://x").pathname;
  const m = /^\/__tools\/ref\/([\w.-]+\.png)$/.exec(urlPath);
  if (!m && urlPath !== "/__tools/golden.json") {
    res.writeHead(403).end();
    return;
  }
  const chunks = [];
  req.on("data", (c) => chunks.push(c));
  req.on("end", () => {
    const buf = Buffer.concat(chunks);
    if (!m) {
      let ok = false;
      try {
        ok = Array.isArray(JSON.parse(buf).results);
      } catch {}
      if (!ok) return void res.writeHead(400).end("not a golden JSON");
      fs.writeFileSync(path.join(TOOLS, "golden.json"), buf);
      return void res.writeHead(204).end();
    }
    const dir = path.join(TOOLS, "ref");
    fs.mkdirSync(dir, { recursive: true });
    if (buf.subarray(0, 8).toString("hex") !== "89504e470d0a1a0a") {
      res.writeHead(400).end("not a PNG");
      return;
    }
    fs.writeFileSync(path.join(dir, m[1]), buf);
    res.writeHead(204).end();
  });
}

// /__tools/* 는 검증 하네스(tools/)를 서빙 — 배포 대상(web/)에는 섞지 않는다.
function serveStatic(req, res) {
  const urlPath = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const isTool = urlPath.startsWith("/__tools/");
  const base = isTool ? TOOLS : ROOT;
  const rel = isTool ? urlPath.slice("/__tools/".length) : urlPath.slice(1) + (urlPath.endsWith("/") ? "index.html" : ""); // nginx index와 같게
  const file = path.resolve(base, rel);
  if (!file.startsWith(base + path.sep)) {
    res.writeHead(403).end();
    return;
  }
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) {
      res.writeHead(404).end("not found");
      return;
    }
    // ?audit: HTML 맨 앞에 바인딩 감사 스크립트 주입 (tools/audit.js)
    if (!isTool && file.endsWith(".html") && new URL(req.url, "http://x").searchParams.has("audit")) {
      const html = fs.readFileSync(file, "utf8").replace("<head>", '<head><script src="/__tools/audit.js"></script>');
      res.writeHead(200, { "content-type": MIME[".html"], "cache-control": "no-cache" });
      res.end(req.method === "HEAD" ? undefined : html);
      return;
    }
    res.writeHead(200, {
      "content-type": MIME[path.extname(file)] ?? "application/octet-stream",
      "content-length": st.size,
      "cache-control": "no-cache",
    });
    if (req.method === "HEAD") res.end();
    else fs.createReadStream(file).pipe(res);
  });
}

http
  .createServer((req, res) => {
    const local = localOverride(req);
    if (local) {
      const st = fs.statSync(local);
      // Range(bytes=a-b) — 브라우저 변환기가 큰 체크포인트를 URL에서 텐서 단위로 읽을 때 (web/workers/convert-*.js)
      const m = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range ?? "");
      const start = m ? +m[1] : 0,
        end = m && m[2] ? Math.min(+m[2], st.size - 1) : st.size - 1;
      const head = { "content-type": "application/octet-stream", "content-length": end - start + 1, "cache-control": "no-cache", "accept-ranges": "bytes" };
      if (m) res.writeHead(206, { ...head, "content-range": `bytes ${start}-${end}/${st.size}` });
      else res.writeHead(200, head);
      if (req.method === "HEAD") res.end();
      else fs.createReadStream(local, { start, end }).pipe(res);
    } else if (PROXY_PREFIXES.some((p) => req.url.startsWith(p))) proxy(req, res);
    else if (req.method === "PUT") saveRef(req, res);
    else serveStatic(req, res);
  })
  .listen(PORT, "127.0.0.1", () =>
    console.log(
      `devserver http://localhost:${PORT}  (static ${ROOT}, proxy ${PROXY_PREFIXES.join(" ")} → ${UPSTREAM?.origin ?? "(미설정)"}` +
        (LOCAL_OUT ? `, local /out/ override ${LOCAL_OUT})` : ")"),
    ),
  );
