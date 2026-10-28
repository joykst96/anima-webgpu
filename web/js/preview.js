// latent → RGB 근사 프리뷰 (VAE 미사용)
import { $ } from "./dom.js";

// ───────── latent → RGB 프리뷰 (Wan2.1 16채널 근사, VAE 미사용) ─────────
// 계수 출처: ComfyUI latent_formats.py (Wan/Qwen 16ch). denoised latent를 latent 해상도로
// 빠르게 RGB 근사 → 생성 중 캔버스에 표시 (VAE 호출 없음).
const LATENT_RGB = [
  // 16×3 (R,G,B)
  [-0.1299, -0.1692, 0.2932],
  [0.0671, 0.0406, 0.0442],
  [0.3568, 0.2548, 0.1747],
  [0.0372, 0.2344, 0.142],
  [0.0313, 0.0189, -0.0328],
  [0.0296, -0.0956, -0.0665],
  [-0.3477, -0.4059, -0.2925],
  [0.0166, 0.1902, 0.1975],
  [-0.0412, 0.0267, -0.1364],
  [-0.1293, 0.074, 0.1636],
  [0.068, 0.3019, 0.1128],
  [0.0032, 0.0581, 0.0639],
  [-0.1251, 0.0927, 0.1699],
  [0.006, -0.0633, 0.0005],
  [0.3477, 0.2275, 0.295],
  [0.1984, 0.0913, 0.1861],
];
const LATENT_RGB_BIAS = [-0.1835, -0.0868, -0.336];

// latentF32: (1,16,1,H,W) 평면 packed. latent 해상도(H×W) RGB 근사를 캔버스에 표시.
function previewLatent(latentF32, H, W) {
  const plane = H * W;
  const im = new ImageData(W, H);
  for (let p = 0; p < plane; p++) {
    let r = LATENT_RGB_BIAS[0],
      g = LATENT_RGB_BIAS[1],
      b = LATENT_RGB_BIAS[2];
    for (let c = 0; c < 16; c++) {
      const v = latentF32[c * plane + p];
      r += v * LATENT_RGB[c][0];
      g += v * LATENT_RGB[c][1];
      b += v * LATENT_RGB[c][2];
    }
    // factors 출력은 대략 ±0.5 범위 — 그대로 ×255면 채도가 과함. 진폭을 줄여 완화.
    im.data[p * 4] = Math.min(255, Math.max(0, (r * 0.5 + 0.5) * 255));
    im.data[p * 4 + 1] = Math.min(
      255,
      Math.max(0, (g * 0.5 + 0.5) * 255),
    );
    im.data[p * 4 + 2] = Math.min(
      255,
      Math.max(0, (b * 0.5 + 0.5) * 255),
    );
    im.data[p * 4 + 3] = 255;
  }
  const cv = $("canvas");
  cv.width = W;
  cv.height = H;
  cv.getContext("2d").putImageData(im, 0, 0);
}

export { previewLatent };
