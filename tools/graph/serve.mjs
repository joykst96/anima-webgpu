// 실험용 그래프 정적 서버 (CORS 허용, GET만). node serve.mjs <dir> <port>
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
const [dir, port] = [path.resolve(process.argv[2]), +process.argv[3]];
http.createServer((req, res) => {
  const f = path.resolve(dir, decodeURIComponent(new URL(req.url, "http://x").pathname).slice(1));
  if (req.method !== "GET" || !f.startsWith(dir + path.sep) || !fs.existsSync(f)) return res.writeHead(404).end();
  res.writeHead(200, { "access-control-allow-origin": "*", "content-type": "application/octet-stream", "cache-control": "no-store" });
  fs.createReadStream(f).pipe(res);
}).listen(port, "127.0.0.1", () => console.log("serve", dir, port));
