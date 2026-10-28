// 독립 변환 페이지: 로컬 safetensors → (사용자가 고른 폴더 | OPFS 임시 후 다운로드) — 생성 페이지 '사용자 설정'에서 그 파일들을 고른다.
// 다른 서브도메인에 따로 배포할 수 있다: 이 폴더(convert/) + js/convert-core.js + workers/convert-*.js + css/app.css +
// 템플릿 out/dit/animav10_*.onnx(12개)·*.lora_manifest.json(2개). 템플릿 경로는 이 페이지 기준 ../out/dit/ (?templates=로 바꿀 수 있음).
import { urlSource } from "../js/convert-core.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const MAP_URL = new URL("./anima2b.map.json", import.meta.url).href;
const TEMPLATE_DIR = new URL(params.get("templates") || "../out/dit/", location.href).href;
// 폴더 저장(File System Access)이 없으면 OPFS에 쓰고 다운로드 링크를 준다. ?download=1이면 강제로 다운로드 방식.
const PICK = typeof window.showDirectoryPicker === "function" && !params.has("download");
const TMP = "anima-convert-out";

$("saveMode").textContent = PICK
  ? "변환을 누르면 저장할 폴더를 고릅니다 — 그 안에 '<이름>_base'(와 '_turbo') 폴더가 생깁니다."
  : "이 브라우저는 폴더 저장을 지원하지 않아, 변환이 끝나면 파일별 다운로드 링크를 보여 줍니다 (그동안 브라우저 저장소를 임시로 씀).";

// ?src=<URL>: 파일 대신 URL에서 읽는다(HTTP Range로 텐서 단위 — 수 GB를 통째로 받지 않음)
let srcUrl = null;
if (params.get("src"))
  urlSource(new URL(params.get("src"), location.href).href).then(
    (s) => ((srcUrl = s), ($("convCkpt").disabled = true), ($("convStat").textContent = `입력: ${s.name} (URL)`), updateButton()),
    (e) => ($("convStat").textContent = `src 실패: ${e.message}`),
  );

function updateButton() {
  $("btnConvert").disabled = !(srcUrl || $("convCkpt").files.length) || ($("convTurbo").checked && !$("convLora").files.length);
}
$("convCkpt").addEventListener("change", updateButton);
$("convLora").addEventListener("change", updateButton);
$("convTurbo").addEventListener("change", () => {
  $("convLora").disabled = !$("convTurbo").checked;
  updateButton();
});

async function tmpRoot(clear) {
  const root = await navigator.storage.getDirectory();
  if (clear) await root.removeEntry(TMP, { recursive: true }).catch(() => {});
  return root.getDirectoryHandle(TMP, { create: true });
}

async function showDownloads(dir, folders) {
  const box = $("outList");
  box.textContent = "";
  const links = [];
  for (const name of Object.values(folders)) {
    const sub = await dir.getDirectoryHandle(name);
    const g = document.createElement("div");
    g.className = "grp";
    g.textContent = `${name}/ — 같은 폴더에 모두 받아 두세요`;
    box.appendChild(g);
    const entries = [];
    for await (const [n, h] of sub.entries()) if (h.kind === "file") entries.push([n, h]);
    entries.sort(([a], [b]) => a.localeCompare(b));
    for (const [n, h] of entries) {
      const f = await h.getFile();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(f);
      a.download = n;
      a.textContent = `${n} (${(f.size / 2 ** 20).toFixed(f.size > 2 ** 20 ? 0 : 2)}MB)`;
      box.appendChild(a);
      links.push(a);
    }
  }
  const all = document.createElement("button");
  all.className = "secondary";
  all.textContent = `모두 다운로드 (${links.length}개 — 브라우저가 여러 파일 다운로드 허용을 물을 수 있음)`;
  all.onclick = async () => {
    for (const a of links) {
      a.click();
      await new Promise((r) => setTimeout(r, 400));
    }
  };
  const clr = document.createElement("button");
  clr.className = "secondary";
  clr.textContent = "임시 저장 지우기 (다 받은 뒤)";
  clr.onclick = async () => {
    links.forEach((a) => URL.revokeObjectURL(a.href));
    await tmpRoot(true);
    await (await navigator.storage.getDirectory()).removeEntry(TMP).catch(() => {});
    box.textContent = "임시 저장을 지웠습니다.";
  };
  box.append(all, clr);
}

$("btnConvert").onclick = async () => {
  const ckpt = srcUrl || $("convCkpt").files[0],
    turbo = $("convTurbo").checked,
    lora = turbo ? $("convLora").files[0] : null;
  const folder = ($("convName").value.trim() || ckpt.name.replace(/\.safetensors$/i, "")).replace(/[\\/:*?"<>|]+/g, "_");
  const variants = turbo ? ["base", "turbo"] : ["base"];
  const stat = $("convStat"),
    bar = $("convBar");
  let dir;
  try {
    dir = PICK ? await window.showDirectoryPicker({ id: "anima-convert", mode: "readwrite" }) : await tmpRoot(true);
  } catch (e) {
    stat.textContent = e.name === "AbortError" ? "폴더 선택 취소" : `폴더 열기 실패: ${e.message}`;
    return;
  }
  $("btnConvert").disabled = true;
  $("outList").textContent = "";
  const t0 = performance.now();
  const wk = new Worker(new URL("../workers/convert-worker.js", import.meta.url), { type: "module" });
  try {
    const { meta, folders } = await new Promise((resolve, reject) => {
      wk.onmessage = ({ data }) => {
        if (data.type === "progress") {
          bar.style.width = data.frac * 100 + "%";
          stat.textContent = `${data.stage} ${(data.frac * 100) | 0}% (${((performance.now() - t0) / 1000).toFixed(0)}s)`;
        } else if (data.type === "done") resolve(data);
        else if (data.type === "error") reject(new Error(data.message));
      };
      wk.onerror = (e) => reject(new Error(e.message || "Worker 오류"));
      wk.postMessage({ type: "start", dir, folder, ckpt, lora, strength: 1.0, variants, mapUrl: MAP_URL, templateDir: TEMPLATE_DIR });
    });
    window.__convertResult = { meta, folders }; // 확인용
    stat.textContent = `완료 ${meta.seconds}s — ${Object.values(folders).join(", ")}`;
    if (PICK) $("outList").textContent = `저장됨: ${dir.name}/${Object.values(folders).join(`, ${dir.name}/`)}`;
    else await showDownloads(dir, folders);
  } catch (e) {
    stat.textContent = `실패: ${e.message}`;
  } finally {
    wk.terminate();
    updateButton();
  }
};
