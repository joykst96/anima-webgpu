// 생성 큐 + 작업 실행(인코딩 → 샘플링 → VAE → 갤러리)
import { $, log } from "./dom.js";
import { genQueue, jobState } from "./queue-state.js";
import { CTX_LEN, LAT_C } from "./constants.js";
import { disposeRun, disposeTensor, makeF16Tensor, ort, toF32 } from "./ort.js";
import { gaussianFill, mulberry32, normalSchedule, sampleCfgppUd10Ab, sampleErSde, sampleEuler, simpleSchedule } from "./sampling.js";
import { sampleSpeed } from "./speed.js";
import { createEasyCache } from "./easycache.js";
import { previewLatent } from "./preview.js";
import { lora } from "./lora.js";
import { ditSessions, loadState, sess } from "./loader.js";
import { encodePrompt, mergeNegPip } from "./prompt.js";
import { decodeLatent } from "./vae.js";
import { PRECISION_LABEL, settings } from "./settings.js";
import { canvasToBlob, folderMode, pushToGallery, pushToGalleryFromBlob, refreshPickDirBtn, saveBlobToFolder } from "./gallery.js";

// ───────── 생성 큐 + 러너 ─────────
let curCancel = false; // 현재 작업 중단 신호
let jobSeq = 0;

// '생성' = 현재 설정을 스냅샷해 큐에 추가. 비어있으면 러너가 즉시 시작.
// 성공하면 true, 검증 실패로 큐에 넣지 못하면 false (배치 루프가 조기중단에 사용).
function enqueueGenerate() {
  const pxW = +$("width").value,
    pxH = +$("height").value;
  if (pxW % 16 || pxH % 16) {
    log("해상도는 16의 배수여야 함", "err");
    return false;
  }
  if (Math.max(pxW, pxH) > 2048 || Math.min(pxW, pxH) < 384) {
    log(
      "해상도는 384~2048px 범위여야 함 (큰 해상도는 VRAM에 따라 실패)",
      "err",
    );
    return false;
  }
  let seed = +$("seed").value;
  if (seed < 0) seed = (Math.random() * 0x7fffffff) | 0;

  // SPEED 스케줄 파싱 (개발패널 입력). 켜졌을 때만.
  const speedOn = $("speedChk").checked;
  let speedStages = null,
    speedTransSig = null;
  if (speedOn) {
    const parseList = (s) =>
      s
        .split(",")
        .map((v) => parseFloat(v.trim()))
        .filter((v) => !isNaN(v));
    speedStages = parseList($("speedStages").value);
    speedTransSig = parseList($("speedTransSigmas").value);
    // 검증: stages 오름차순+1.0끝, transSig 길이=stages-1, 모두 0<σ<범위
    if (
      speedStages.length < 2 ||
      speedStages[speedStages.length - 1] !== 1.0 ||
      speedStages.some((v, i) => i > 0 && v <= speedStages[i - 1]) ||
      speedTransSig.length !== speedStages.length - 1
    ) {
      log(
        "SPEED 스케줄 오류: stages는 오름차순+1.0끝, transition σ 길이는 stages-1이어야 함",
        "err",
      );
      return false;
    }
  }

  const job = {
    id: ++jobSeq,
    pxW,
    pxH,
    seed,
    steps: +$("steps").value,
    cfg: +$("cfg").value,
    sampler: $("sampler").value,
    scheduler: $("scheduler").value,
    prompt: $("prompt").value,
    negative: $("negative").value,
    negpip: $("negpipChk").checked,
    negpipW: +$("negpipW").value,
    modelKey: $("modelSel").value,
    speed: speedOn,
    speedStages,
    speedTransSig,
    xattnCompact: settings.xattnCompact,
    easyCache: settings.easyCache,
    // LoRA는 현재 반영 상태를 그대로 씀(스냅샷 시점 lora.bin) — 별도 복사 불필요.
  };
  genQueue.push(job);
  renderQueue();
  refreshPickDirBtn(); // 첫 생성이 큐에 들어가면 폴더 지정 잠금
  pumpQueue();
  return true;
}

let pumping = false;
async function pumpQueue() {
  if (pumping) return;
  pumping = true;
  $("btnCancel").style.display = "";
  $("btnCancelTop").style.display = "";
  // 큐가 도는 동안 DiT 세션을 내리는 동작 잠금 (loadAll/applyLora도 queueBusy()로 거부)
  $("btnLoad").disabled = true;
  $("btnLoraApply").disabled = true;
  while (genQueue.length) {
    jobState.cur = genQueue.shift();
    curCancel = false;
    renderQueue();
    try {
      await runJob(jobState.cur);
    } catch (e) {
      log("생성 실패: " + e.message, "err");
      if (/external Instance|[Dd]evice.*lost|mapAsync/.test(e.message))
        log("GPU 컨텍스트가 손실됨 — 페이지를 새로고침해 주세요", "err");
      console.error(e);
    }
    jobState.cur = null;
    renderQueue();
  }
  pumping = false;
  $("btnCancel").style.display = "none";
  $("btnCancelTop").style.display = "none";
  $("btnLoad").disabled = loadState.loading;
  $("btnLoraApply").disabled = !lora.dirty || loadState.loading;
}

function cancelCurrent() {
  if (jobState.cur) {
    curCancel = true;
    log("현재 생성 중단 요청...", "dim");
  }
}

// 큐 UI: 실행 중 + 대기 항목. 대기 항목은 × 로 제거 가능.
function renderQueue() {
  const box = $("queueList");
  box.innerHTML = "";
  const rows = [];
  if (jobState.cur) rows.push({ job: jobState.cur, running: true });
  for (const j of genQueue) rows.push({ job: j, running: false });
  for (const { job, running } of rows) {
    const el = document.createElement("div");
    el.className = "qitem" + (running ? " running" : "");
    const meta = document.createElement("span");
    meta.className = "qmeta";
    meta.textContent = `${running ? "▶ " : ""}${job.pxW}×${job.pxH} · ${job.steps}스텝 · CFG ${job.cfg} · seed ${job.seed}`;
    el.appendChild(meta);
    if (!running) {
      // 대기 항목: 클릭하면 제거, hover 시 안내 표시
      el.classList.add("removable");
      el.title = "클릭하면 큐에서 제거";
      const hint = document.createElement("span");
      hint.className = "qhint";
      hint.textContent = "✕ 제거";
      el.appendChild(hint);
      el.onclick = () => {
        const i = genQueue.indexOf(job);
        if (i >= 0) genQueue.splice(i, 1);
        renderQueue();
      };
    }
    box.appendChild(el);
  }
}

// DiT의 cross-attention 입력. 그래프가 xattn_index/xattn_bias_row를 받을 때만 패딩 압축 가능 (tools/graph/xattn_compact.py).
// 압축: 어텐션에 실제 토큰 n개 + 패딩 대표 1개(점수 bias ln(512−n))만 넣는다 — 0 패딩 m개와 수학적으로 같다.
// K/V 투영은 그래프가 종전처럼 512행으로 계산하고 어텐션 직전에 행만 고른다. 켜면 합산 순서 차이(~1e-6)를 DP4A가 증폭해 그림이 조금 달라진다.
// 끄면 전체 인덱스 + bias 0 (종전과 같은 결과). context는 job 내내 같아 job마다 한 번 만든다.
function crossFeed(c, compact) {
  const feed = { context: c.ctx, ...(c.mask ? { negpip_mask: c.mask } : {}) },
    own = [];
  if (!sess.dit.inputNames.includes("xattn_index")) return { feed, own, compacted: false };
  const m = CTX_LEN - c.len;
  const T = compact && m > 1 ? c.len + 1 : CTX_LEN;
  const index = new BigInt64Array(T),
    row = new Float32Array(T);
  for (let i = 0; i < T; i++) index[i] = BigInt(i); // 압축 시 마지막 = 첫 패딩 행(c.len) — 0 행
  if (T < CTX_LEN) row[T - 1] = Math.log(m);
  feed.xattn_index = new ort.Tensor("int64", index, [T]);
  feed.xattn_bias_row = new ort.Tensor("float32", row, [1, 1, 1, T]);
  own.push(feed.xattn_index, feed.xattn_bias_row);
  return { feed, own, compacted: T < CTX_LEN };
}

async function runJob(job) {
  await loadState.warmup; // 로드 직후 셰이더 워밍업이 아직 돌고 있으면 끝난 뒤에 시작
  // job은 큐에 넣을 때의 모델로만 돈다. 그 사이 상주 DiT가 바뀌었으면(정상 흐름에선 잠금으로 불가) 건너뛴다.
  if (!sess.dit || sess.dit !== ditSessions[job.modelKey]) {
    log(`건너뜀 — 큐에 넣을 때의 모델(${job.modelKey})이 로드돼 있지 않음`, "err");
    return;
  }
  $("btnSave").disabled = true;
  const t0 = performance.now();
  {
    const pxW = job.pxW,
      pxH = job.pxH;
    const W = pxW / 8,
      H = pxH / 8,
      LAT_N = LAT_C * H * W;

    const precision = sess.dit.precision; // 같은 시드라도 모드가 다르면 그림이 달라 메타데이터에 남긴다
    const steps = job.steps,
      cfg = job.cfg;
    let seed = job.seed;
    const rand = mulberry32(seed);
    const useCfg = cfg > 1.0;

    log(`생성 시작 — seed ${seed} · ${pxW}x${pxH}`, "dim");
    const negpipOK = !!(
      sess.dit.inputNames && sess.dit.inputNames.includes("negpip_mask")
    );
    // 멀티 LoRA는 강도를 변환 단계(scale)에서 반영하므로 그래프 스칼라는 1.0 고정.
    // 단일 LoRA: 강도를 그래프 lora_scale로 전달(재컴파일 없이 슬라이더 즉시 반영).
    // 멀티 LoRA: 강도는 SVD 병합 단계에 흡수됐으므로 1.0 고정.
    const activeLoras = lora.items.filter((i) => i.src);
    const loraScale =
      activeLoras.length === 1 ? Math.max(0, activeLoras[0].weight) : 1;
    const loraFeed = sess.dit.inputNames.includes("lora_scale")
      ? {
          lora_scale: new ort.Tensor(
            "float32",
            Float32Array.of(loraScale),
            [1],
          ),
        }
      : {};

    // NegPip 체크박스: CFG=1일 때 부정 프롬프트를 음수 가중치 그룹으로 긍정에 병합.
    // 부정 프롬프트 내부의 (tag:1.2)는 -1.2가 되도록 구간마다 가중치를 곱해 풀어 쓴다(mergeNegPip — ComfyUI 덮어쓰기 규칙 대응).
    let posText = job.prompt;
    const negText = job.negative.trim();
    if (job.negpip && negText) {
      if (!negpipOK) {
        log("이 모델엔 NegPip 입력이 없음 — 체크박스 무시", "dim");
      } else if (useCfg) {
        log(
          "CFG>1에서는 기존 부정 프롬프트 경로 사용 — NegPip 체크 무시",
          "dim",
        );
      } else {
        let w = job.negpipW;
        if (!(w < 0)) {
          log("NegPip 가중치는 음수여야 함 — -1.0 사용", "dim");
          w = -1;
        }
        posText = mergeNegPip(posText, negText, w);
        log(`NegPip 적용 — 부정 프롬프트 가중치 ${w}`, "dim");
      }
    }
    const ctxPos = await encodePrompt(posText, negpipOK);
    const ctxNeg = useCfg
      ? await encodePrompt(job.negative, negpipOK)
      : null;
    const xPos = crossFeed(ctxPos, job.xattnCompact),
      xNeg = ctxNeg ? crossFeed(ctxNeg, job.xattnCompact) : null;
    const disposeCross = () => [...xPos.own, ...(xNeg ? xNeg.own : [])].forEach(disposeTensor);

    const x = new Float32Array(LAT_N);
    gaussianFill(x, rand); // σ_max = 1.0

    const sigmas = (
      job.scheduler === "simple" ? simpleSchedule : normalSchedule
    )(steps);

    // EasyCache(선택): 변화가 작은 스텝의 DiT 호출을 건너뛰고 직전 (출력 − 입력) 차이를 재사용 (easycache.js)
    const ec = job.easyCache ? createEasyCache(LAT_C, sigmas[sigmas.length - 2]) : null;
    const denoise = async (xf32, sigma, curH = H, curW = W) => {
      const cached = ec && ec.before(xf32, sigma, curH, curW);
      let v,
        vu = null,
        lat = null,
        t = null,
        o1 = null,
        o2 = null;
      if (cached) [v, vu] = cached;
      else {
        lat = makeF16Tensor(xf32, [1, LAT_C, 1, curH, curW]);
        t = new ort.Tensor("float32", new Float32Array([sigma]), [1]); // t = σ
        o1 = await sess.dit.run({
          latent: lat,
          timesteps: t,
          ...xPos.feed,
          ...loraFeed,
        });
        v = toF32(o1[Object.keys(o1)[0]].data);
        if (useCfg) {
          o2 = await sess.dit.run({
            latent: lat,
            timesteps: t,
            ...xNeg.feed,
            ...loraFeed,
          });
          vu = toF32(o2[Object.keys(o2)[0]].data);
        }
        if (ec) ec.after(xf32, v, vu);
      }
      // CFG++ 샘플러(cfgpp_ud10_ab)는 부정 프롬프트 denoised도 쓴다 → den.uncond (CFG 1이면 없음 = 긍정과 같음)
      let denU = null;
      if (useCfg) {
        if (job.sampler === "cfgpp_ud10_ab") {
          denU = new Float32Array(xf32.length);
          for (let k = 0; k < denU.length; k++) denU[k] = xf32[k] - sigma * vu[k];
        }
        for (let k = 0; k < v.length; k++)
          v[k] = vu[k] + cfg * (v[k] - vu[k]);
      }
      const den = new Float32Array(xf32.length);
      for (let k = 0; k < den.length; k++)
        den[k] = xf32[k] - sigma * v[k];
      if (denU) den.uncond = denU;
      // 매 스텝 새로 만들어진 입력/출력 텐서 반환 (native EP GPU 버퍼 누적 방지).
      // context/mask/loraFeed는 job 내내 재사용 → 여기서 놓지 않음(job 끝에 정리).
      disposeTensor(lat);
      disposeTensor(t);
      disposeRun(o1);
      disposeRun(o2);
      return den; // denoised = x - σ·v
    };

    let lastPreview = 0;
    const onStep = (i, s, sn, den, curH = H, curW = W) => {
      if (curCancel) return false; // 중단 신호 → 샘플러 루프 탈출
      $("genBar").style.width = ((i + 1) / steps) * 100 + "%";
      $("genStat").textContent =
        `step ${i + 1}/${steps}  σ ${s.toFixed(4)} → ${sn.toFixed(4)}  ` +
        `${curW * 8}×${curH * 8}  ` +
        `(${((performance.now() - t0) / 1000).toFixed(1)}s)`;
      // latent 프리뷰: 체크박스 ON일 때만, ~250ms 간격으로 (마지막 스텝은 항상).
      const now = performance.now();
      if (
        den &&
        $("previewChk").checked &&
        (now - lastPreview > 250 || sn === 0)
      ) {
        lastPreview = now;
        try {
          previewLatent(den, curH, curW);
        } catch (_) {}
      }
      return true;
    };

    let xf;
    if (job.speed) {
      // SPEED: 멀티해상도. 목표 H/W 기준, stages/transSig는 job에서.
      xf = await sampleSpeed(denoise, x, sigmas, rand, onStep, {
        stages: job.speedStages,
        transSig: job.speedTransSig,
        fullH: H,
        fullW: W,
        C: LAT_C,
        patch: 2,
      });
    } else {
      const sampler =
        job.sampler === "er_sde" ? sampleErSde : job.sampler === "cfgpp_ud10_ab" ? sampleCfgppUd10Ab : sampleEuler;
      xf = await sampler(denoise, x, sigmas, rand, onStep);
    }

    if (xf === null) {
      // 중단됨 — VAE 디코드 스킵
      disposeCross();
      disposeTensor(ctxPos.ctx);
      disposeTensor(ctxPos.mask);
      if (ctxNeg) {
        disposeTensor(ctxNeg.ctx);
        disposeTensor(ctxNeg.mask);
      }
      log(`생성 중단됨 — seed ${seed}`, "dim");
      $("genStat").textContent = "중단됨";
      $("genBar").style.width = "0%";
      return;
    }

    if (ec)
      log(
        `EasyCache — 스텝 ${ec.calls}회 중 ${ec.skipped.length}회 DiT 건너뜀` +
          (ec.skipped.length ? ` (${ec.skipped.join(", ")}번째)` : ""),
        "dim",
      );
    log("VAE 디코드...", "dim");
    const dec = await decodeLatent(xf, H, W);
    const img = dec.img; // fp32 (1,3,H,W) [-1,1]

    const cv = $("canvas");
    cv.width = pxW;
    cv.height = pxH;
    const cx = cv.getContext("2d");
    const im = cx.createImageData(pxW, pxH);
    const plane = pxW * pxH;
    for (let p = 0; p < plane; p++) {
      im.data[p * 4] = Math.min(255, Math.max(0, (img[p] + 1) * 127.5));
      im.data[p * 4 + 1] = Math.min(
        255,
        Math.max(0, (img[plane + p] + 1) * 127.5),
      );
      im.data[p * 4 + 2] = Math.min(
        255,
        Math.max(0, (img[2 * plane + p] + 1) * 127.5),
      );
      im.data[p * 4 + 3] = 255;
    }
    cx.putImageData(im, 0, 0);
    // 픽셀 판독 완료 → 이 job에서 생성된 텐서 전부 반환.
    // (img는 VAE 출력 버퍼 참조일 수 있으므로 반드시 판독 후에 놓는다)
    dec.release();
    disposeCross();
    disposeTensor(ctxPos.ctx);
    disposeTensor(ctxPos.mask);
    if (ctxNeg) {
      disposeTensor(ctxNeg.ctx);
      disposeTensor(ctxNeg.mask);
    }
    log(
      `완료 — ${((performance.now() - t0) / 1000).toFixed(1)}s · seed ${seed}`,
      "ok",
    );
    const metaTxt =
      `seed ${seed} · ${pxW}x${pxH} · ${steps}스텝 · CFG ${cfg}` +
      (precision === "fast" ? ` · ${PRECISION_LABEL.fast}` : "") +
      (xPos.compacted ? " · 패딩 압축" : "") +
      (ec ? " · EasyCache" : "");
    $("genStat").textContent = metaTxt;
    if (folderMode.active) {
      // 폴더 자동저장: 디스크에 쓰고 objectURL 썸네일. dataURL 미보관.
      try {
        const blob = await canvasToBlob(cv);
        const name = await saveBlobToFolder(blob);
        pushToGalleryFromBlob(blob, pxW, pxH, `${metaTxt} · ${name}`, name);
        log(`저장됨 — ${name}`, "ok");
      } catch (e) {
        // 쓰기 실패(권한 회수/용량 등): 세션 메모리로 폴백해 결과 유실 방지
        log(`폴더 저장 실패 — 세션 보관으로 폴백: ${e.message}`, "err");
        pushToGallery(cv, metaTxt);
      }
      // 폴더 모드에선 수동 저장 버튼은 계속 비활성(자동저장 안내 유지)
    } else {
      $("btnSave").disabled = false;
      pushToGallery(cv, metaTxt); // 스택에 썸네일 추가 (최신이 앞)
    }
  }
}
// 생성 버튼 = 배치. 클릭 시점(사용자 의도 지점)에서 장수를 읽어 enqueueGenerate를
// 그만큼 반복 호출한다. enqueueGenerate는 '1장 확정' 단일 책임만 지므로, 시드(-1
// 랜덤)·향후 와일드카드 등 '생성마다 달라지는' 로직이 매 호출마다 독립 적용된다.
const runBatch = () => {
  const raw = parseInt($("batch").value, 10);
  const n = Math.min(100, Math.max(1, isNaN(raw) ? 1 : raw));
  // 검증 실패(해상도/SPEED)면 첫 호출에서 false → 즉시 중단(로그 N번 반복 방지)
  for (let i = 0; i < n; i++) {
    if (!enqueueGenerate()) break;
  }
};

export { cancelCurrent, runBatch };
