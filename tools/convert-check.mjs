// 브라우저 변환기 코어(web/js/convert-core.js)를 Node에서 돌려 기존 빌드와 바이트 대조한다.
//   node tools/convert-check.mjs --ckpt <anima-base-v1.0.safetensors> [--lora <turbo.safetensors>]
//        [--ref out-local/dit] [--out <임시 폴더>] [--only base|turbo]
// 템플릿 그래프는 --ref의 원본을 쓰되, 패치 자리를 먼저 0으로 지운 뒤 채워서 실제로 다시 쓰는지 확인한다.
import { createHash } from "node:crypto";
import { mkdirSync, openSync, closeSync, readFileSync, readSync, writeSync, ftruncateSync, existsSync } from "node:fs";
import { join } from "node:path";
import { convertDit, patchGraph, shardManifest } from "../web/js/convert-core.js";

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => (x.startsWith("--") ? [...a, [x.slice(2), arr[i + 1]]] : a), []));
const map = JSON.parse(readFileSync(new URL("../web/convert/anima2b.map.json", import.meta.url)));
const ref = args.ref || "out-local/dit";
const outDir = args.out || "out-local/convert-check";
mkdirSync(outDir, { recursive: true });

const reader = (path) => {
  const fd = openSync(path, "r");
  return async (off, len) => {
    const b = Buffer.allocUnsafe(len);
    let got = 0;
    while (got < len) got += readSync(fd, b, got, len - got, off + got);
    return new Uint8Array(b.buffer, b.byteOffset, len);
  };
};

const variants = args.only ? [args.only] : args.lora ? ["base", "turbo"] : ["base"];
const outs = {},
  fds = {};
for (const v of variants) {
  const stem = map.stems[v];
  fds[v] = map.shards.map((s) => {
    const fd = openSync(join(outDir, s.name.replace("{stem}", stem)), "w+");
    ftruncateSync(fd, s.size); // 정렬 패딩 = 0
    return fd;
  });
  outs[v] = { write: (i, off, bytes) => void writeSync(fds[v][i], bytes, 0, bytes.length, off) };
}

const t0 = performance.now();
let lastLog = 0;
const res = await convertDit({
  map,
  read: reader(args.ckpt),
  lora: args.lora ? { read: reader(args.lora), strength: 1.0 } : null,
  outs,
  onProgress: (f) => {
    const now = performance.now();
    if (now - lastLog > 5000) {
      lastLog = now;
      console.log(`  ${(f * 100).toFixed(0)}% (${((now - t0) / 1000).toFixed(0)}s)`);
    }
  },
});
for (const v of variants) fds[v].forEach(closeSync);
const tConv = (performance.now() - t0) / 1000;
console.log(`[convert] ${variants.join("+")} ${tConv.toFixed(1)}s, LoRA 모듈 ${res.loraModules}개`);

const sha = (b) => createHash("sha256").update(b).digest("hex");
let ok = true;
for (const v of variants) {
  const stem = map.stems[v];
  for (const s of map.shards) {
    const name = s.name.replace("{stem}", stem);
    const a = sha(readFileSync(join(outDir, name))),
      b = sha(readFileSync(join(ref, name)));
    ok &&= a === b;
    console.log(`[shard] ${name}: ${a === b ? "동일" : "다름"} ${a.slice(0, 16)}`);
  }
  for (const [gk, g] of Object.entries(map.graphs)) {
    if (!gk.startsWith(v)) continue;
    const tpl = new Uint8Array(readFileSync(join(ref, g.file)));
    if (createHash("sha1").update(tpl).digest("hex") !== g.sha1) {
      console.log(`[graph] ${g.file}: 템플릿 sha1 불일치`);
      ok = false;
      continue;
    }
    const wiped = tpl.slice();
    for (const p of g.patches) wiped.fill(0, p.offset, p.offset + p.length);
    const out = patchGraph(wiped, res.patches[gk]);
    const same = Buffer.compare(Buffer.from(out), Buffer.from(tpl)) === 0;
    ok &&= same;
    console.log(`[graph] ${g.file}: ${same ? "동일" : "다름"} (패치 ${g.patches.length}개)`);
  }
  const mani = shardManifest(map.shards.map((s) => s.name.replace("{stem}", stem)));
  for (const slot of ["_q8a4fns", "_q8a4fnLs"]) {
    const f = join(ref, `${stem}${slot}.onnx.manifest.json`);
    const same = existsSync(f) && readFileSync(f, "utf8") === mani;
    ok &&= same;
    console.log(`[manifest] ${stem}${slot}: ${same ? "동일" : "다름"}`);
  }
}
console.log(ok ? "[check] 통과" : "[check] 실패");
process.exit(ok ? 0 : 1);
