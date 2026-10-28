// 브라우저 변환기와 같은 결과 폴더를 Node로 만든다 (web/js/convert-core.js 그대로 + 변환 Worker와 같은 파일 구성).
//   node tools/convert-folder.mjs --ckpt <체크포인트> [--lora <turbo.safetensors>] --out <폴더> [--templates out-local/dit]
//   → <out>/base/, <out>/turbo/ 에 그래프 6·샤드 6·manifest 2·lora_manifest (convert-meta.json은 생략)
// 다른 파이프라인 산출물과 비교할 때 쓴다 (tools/convert-compare.py).
import { createHash } from "node:crypto";
import { mkdirSync, openSync, closeSync, readFileSync, readSync, writeSync, writeFileSync, ftruncateSync, existsSync } from "node:fs";
import { join } from "node:path";
import { convertDit, patchGraph, shardManifest } from "../web/js/convert-core.js";

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => (x.startsWith("--") ? [...a, [x.slice(2), arr[i + 1]]] : a), []));
const map = JSON.parse(readFileSync(new URL("../web/convert/anima2b.map.json", import.meta.url)));
const tplDir = args.templates || "out-local/dit";
if (!args.ckpt || !args.out) throw new Error("--ckpt, --out 필요");

const reader = (path) => {
  const fd = openSync(path, "r");
  return async (off, len) => {
    const b = Buffer.allocUnsafe(len);
    let got = 0;
    while (got < len) got += readSync(fd, b, got, len - got, off + got);
    return new Uint8Array(b.buffer, b.byteOffset, len);
  };
};

const variants = args.lora ? ["base", "turbo"] : ["base"];
const outs = {},
  fds = {};
for (const v of variants) {
  const dir = join(args.out, v);
  if (existsSync(dir)) throw new Error(`${dir} 이미 있음`);
  mkdirSync(dir, { recursive: true });
  fds[v] = map.shards.map((s) => {
    const fd = openSync(join(dir, s.name.replace("{stem}", map.stems[v])), "w+");
    ftruncateSync(fd, s.size); // 정렬 패딩 = 0
    return fd;
  });
  outs[v] = { write: (i, off, bytes) => void writeSync(fds[v][i], bytes, 0, bytes.length, off) };
}

const t0 = performance.now();
const res = await convertDit({ map, read: reader(args.ckpt), lora: args.lora ? { read: reader(args.lora), strength: 1.0 } : null, outs });
for (const v of variants) fds[v].forEach(closeSync);
console.log(`[convert] ${variants.join("+")} ${((performance.now() - t0) / 1000).toFixed(1)}s, LoRA 모듈 ${res.loraModules}개`);

for (const v of variants) {
  const dir = join(args.out, v),
    stem = map.stems[v];
  for (const [gk, g] of Object.entries(map.graphs)) {
    if (!gk.startsWith(v)) continue;
    const tpl = new Uint8Array(readFileSync(join(tplDir, g.file)));
    if (createHash("sha1").update(tpl).digest("hex") !== g.sha1) throw new Error(`템플릿 sha1 불일치: ${g.file}`);
    writeFileSync(join(dir, g.file), patchGraph(tpl, res.patches[gk]));
  }
  const mani = shardManifest(map.shards.map((s) => s.name.replace("{stem}", stem)));
  for (const slot of ["_q8a4fns", "_q8a4fnLs"]) writeFileSync(join(dir, `${stem}${slot}.onnx.manifest.json`), mani);
  const lm = `${stem}_q8a4fnLs.onnx.lora_manifest.json`;
  writeFileSync(join(dir, lm), readFileSync(join(tplDir, lm)));
}
console.log(`[done] ${args.out}`);
