/**
 * ╔══════════════════════════════════════════════════════════════╗
 * ║            DARK VISION — THE BIG THREE  (Browser Port)      ║
 * ║  #1  Manual ISO + Exposure  (ImageCapture API)              ║
 * ║  #2  Multi-Frame Stacking + Motion Alignment                ║
 * ║  #3  Multi-Scale CLAHE + Unsharp Mask + Gamma Lift          ║
 * ╚══════════════════════════════════════════════════════════════╝
 *
 * IMPROVEMENTS FOR NEAR-PITCH-BLACK:
 *
 *  A. grabFrame() polling — bypasses live-stream frame rate,
 *     lets the sensor integrate longer between captures.
 *
 *  B. Hot-pixel suppression — 3×3 median pre-filter before
 *     stacking. Sensor hot pixels dominate dark frames and
 *     create fixed-pattern noise; median removes them cleanly.
 *
 *  C. Stack depth 48 in darkness — √48 = 6.9× SNR gain.
 *
 *  D. Shadow gamma lift — non-linear curve f(x) = 255*(x/255)^γ
 *     with γ=0.35 in pitch black. Lifts deep shadows without
 *     crushing midtones the way linear gain does.
 *
 *  E. Multi-scale CLAHE — run at tile-8 AND tile-4, blend 60/40.
 *     Tile-8 recovers fine local contrast; tile-4 recovers
 *     broader regional structure.
 *
 *  F. Unsharp mask post-CLAHE — 5×5 Gaussian blur subtracted
 *     at 0.6 strength. Recovers perceived edge detail destroyed
 *     by noise averaging.
 */

import { useState, useRef, useCallback, useEffect } from "react";

// ══════════════════════════════════════════════════════════════
// §01  CONSTANTS
// ══════════════════════════════════════════════════════════════
const C = {
  WIDTH:  640,
  HEIGHT: 480,

  // Stack: up to 48 frames in pitch black → 6.9× SNR
  STACK_MIN:  6,
  STACK_MAX:  48,

  MAX_SHIFT_PX:         30,
  STABLE_MOT_THRESHOLD: 5.0,
  SIGMA_CLIP:           2.0,

  // Multi-scale CLAHE
  CLAHE_TILES_FINE:   8,
  CLAHE_TILES_COARSE: 4,
  CLAHE_CLIP:         3.5,
  CLAHE_BLEND:        0.6,   // fine weight; coarse = 1 - this

  // Gamma curve exponent in pitch black (< 1 = shadow lift)
  GAMMA_DARK:   0.35,
  GAMMA_BRIGHT: 1.0,

  // Unsharp mask
  USM_STRENGTH: 0.65,

  // Darkness threshold (mean luma)
  DARK_THRESHOLD: 40,

  // grabFrame interval ms — lower = more frames but more CPU
  GRAB_INTERVAL_MS: 80,
};

// ══════════════════════════════════════════════════════════════
// §02  Hardware exposure
// ══════════════════════════════════════════════════════════════
async function applyHardwareExposure(track) {
  try {
    const caps = track.getCapabilities?.();
    if (!caps) return false;
    const c = {};
    let hit = false;
    if (caps.exposureMode?.includes?.("manual"))  { c.exposureMode = "manual"; hit = true; }
    if (caps.exposureTime)  { c.exposureTime = caps.exposureTime.max;  hit = true; }
    if (caps.iso)           { c.iso          = caps.iso.max;           hit = true; }
    if (caps.whiteBalanceMode?.includes?.("manual")) c.whiteBalanceMode = "manual";
    if (caps.focusMode?.includes?.("manual"))        { c.focusMode = "manual"; c.focusDistance = 0; }
    if (hit) { await track.applyConstraints({ advanced: [c] }); return true; }
  } catch (_) {}
  return false;
}

// ══════════════════════════════════════════════════════════════
// §03  Hot-pixel suppression — 3×3 median, subsampled for speed
//      Applied once per raw frame before accumulation.
//      Targets isolated bright pixels (sensor defects) that
//      appear as fixed-pattern noise in dark frames.
// ══════════════════════════════════════════════════════════════
function medianFilter(luma, width, height) {
  const out = new Uint8Array(luma);
  const buf = new Uint8Array(9);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      // Only process pixels significantly brighter than neighbours
      // (avoids blurring real edges)
      const center = luma[i];
      const n = [
        luma[(y-1)*width+(x-1)], luma[(y-1)*width+x], luma[(y-1)*width+(x+1)],
        luma[y    *width+(x-1)],  center,              luma[y    *width+(x+1)],
        luma[(y+1)*width+(x-1)], luma[(y+1)*width+x], luma[(y+1)*width+(x+1)],
      ];
      const sorted = n.slice().sort((a, b) => a - b);
      const med    = sorted[4];
      // Only correct if pixel is an outlier (hot pixel heuristic)
      if (center > med + 30) out[i] = med;
    }
  }
  return out;
}

// ══════════════════════════════════════════════════════════════
// §04  MotionEstimator
// ══════════════════════════════════════════════════════════════
class MotionEstimator {
  constructor() { this.refLuma = null; this.motionMag = 0; }
  lockReference(luma) { this.refLuma = new Uint8Array(luma); }

  estimate(luma, width, height) {
    if (!this.refLuma) return { dx: 0, dy: 0, magnitude: 0 };
    const step = 4;
    let sumW = 0, sumDx = 0, sumDy = 0, totalDiff = 0;
    for (let y = step; y < height - step; y += step) {
      for (let x = step; x < width - step; x += step) {
        const i    = y * width + x;
        const diff = Math.abs(luma[i] - this.refLuma[i]);
        totalDiff += diff;
        if (diff > 6) {
          const gx = luma[i + 1] - luma[i - 1];
          const gy = luma[i + width] - luma[i - width];
          sumW  += diff;
          sumDx += gx * diff;
          sumDy += gy * diff;
        }
      }
    }
    const magnitude = totalDiff / ((width / step) * (height / step));
    if (sumW < 1) return { dx: 0, dy: 0, magnitude };
    const sc = 0.12;
    const dx = Math.max(-C.MAX_SHIFT_PX, Math.min(C.MAX_SHIFT_PX, (sumDx / sumW) * sc));
    const dy = Math.max(-C.MAX_SHIFT_PX, Math.min(C.MAX_SHIFT_PX, (sumDy / sumW) * sc));
    this.motionMag = magnitude;
    return { dx, dy, magnitude };
  }

  align(luma, width, height, dx, dy) {
    if (Math.abs(dx) < 0.01 && Math.abs(dy) < 0.01) return luma;
    const dst = new Uint8Array(luma.length);
    const ox = -dx, oy = -dy;
    for (let y = 0; y < height; y++) {
      const sy = y + oy;
      const y0 = Math.max(0, Math.min(height - 1, Math.floor(sy)));
      const y1 = Math.min(height - 1, y0 + 1);
      const fy = sy - y0;
      for (let x = 0; x < width; x++) {
        const sx = x + ox;
        const x0 = Math.max(0, Math.min(width - 1, Math.floor(sx)));
        const x1 = Math.min(width - 1, x0 + 1);
        const fx = sx - x0;
        dst[y * width + x] = Math.round(
          luma[y0*width+x0]*(1-fx)*(1-fy) + luma[y0*width+x1]*fx*(1-fy) +
          luma[y1*width+x0]*(1-fx)*fy     + luma[y1*width+x1]*fx*fy
        );
      }
    }
    return dst;
  }
}

// ══════════════════════════════════════════════════════════════
// §05  StackEngine (unchanged core, deeper max)
// ══════════════════════════════════════════════════════════════
class StackEngine {
  constructor(w, h) { this.w = w; this.h = h; this.reset(); }

  reset() {
    const sz = this.w * this.h;
    this.weightedSum = new Float32Array(sz);
    this.weightSum   = new Float32Array(sz);
    this.runMean     = new Float32Array(sz);
    this.runM2       = new Float32Array(sz);
    this.count = 0; this.zoneTargets = null; this.zoneCounts = null; this.zoneMeta = null;
  }

  targetDepth(meanLuma) {
    const t = Math.min(1, meanLuma / C.DARK_THRESHOLD);
    return Math.round(C.STACK_MAX - t * (C.STACK_MAX - C.STACK_MIN));
  }

  _computeZoneTargets(luma, gd) {
    const COLS = 4, ROWS = 4;
    const zw = Math.floor(this.w / COLS), zh = Math.floor(this.h / ROWS);
    const targets = new Int32Array(COLS * ROWS), counts = new Int32Array(COLS * ROWS);
    for (let zy = 0; zy < ROWS; zy++) for (let zx = 0; zx < COLS; zx++) {
      let sum = 0;
      for (let y = zy*zh; y < (zy+1)*zh; y++)
        for (let x = zx*zw; x < (zx+1)*zw; x++) sum += luma[y*this.w+x];
      const t = Math.min(1, (sum/(zw*zh))/200);
      targets[zy*COLS+zx] = Math.min(gd, Math.round(C.STACK_MAX - t*(C.STACK_MAX-C.STACK_MIN)));
    }
    return { targets, counts, COLS, ROWS, zw, zh };
  }

  accumulate(luma, motionMag, meanLuma) {
    const n = this.count + 1;
    if (this.count === 0) {
      const z = this._computeZoneTargets(luma, this.targetDepth(meanLuma));
      this.zoneTargets = z.targets; this.zoneCounts = z.counts; this.zoneMeta = z;
    }
    const w = 1 / (1 + motionMag / C.STABLE_MOT_THRESHOLD);
    for (let i = 0; i < luma.length; i++) {
      const pf = luma[i], oldMean = this.runMean[i], delta = pf - oldMean;
      this.runMean[i] += delta / n;
      this.runM2[i]   += delta * (pf - this.runMean[i]);
      const sigma = n > 1 ? Math.sqrt(this.runM2[i] / (n-1)) : Infinity;
      if (!(n > 2 && Math.abs(pf - oldMean) > C.SIGMA_CLIP * sigma)) {
        this.weightedSum[i] += pf * w; this.weightSum[i] += w;
      }
    }
    for (let z = 0; z < this.zoneTargets.length; z++)
      if (this.zoneCounts[z] < this.zoneTargets[z]) this.zoneCounts[z]++;
    this.count++;
  }

  allZonesDone() {
    if (!this.zoneCounts) return false;
    for (let z = 0; z < this.zoneTargets.length; z++)
      if (this.zoneTargets[z] > 0 && this.zoneCounts[z] < this.zoneTargets[z]) return false;
    return true;
  }

  exportAndReset() {
    const out = new Uint8Array(this.weightedSum.length);
    for (let i = 0; i < out.length; i++)
      out[i] = this.weightSum[i] > 0
        ? Math.max(0, Math.min(255, Math.round(this.weightedSum[i]/this.weightSum[i]))) : 0;
    this.reset();
    return out;
  }
}

// ══════════════════════════════════════════════════════════════
// §06  ClaheEngine — single-scale worker
// ══════════════════════════════════════════════════════════════
function runClahe(src, width, height, tiles, clipFactor) {
  const TX = tiles, TY = tiles;
  const tW = Math.floor(width/TX), tH = Math.floor(height/TY);
  const tSz = tW * tH;
  const clip = Math.max(1, Math.floor(clipFactor * tSz / 256));

  const luts = Array.from({length:TY}, () => Array.from({length:TX}, () => new Uint8Array(256)));
  for (let ty = 0; ty < TY; ty++) for (let tx = 0; tx < TX; tx++) {
    const hist = new Int32Array(256);
    for (let y = ty*tH; y < (ty+1)*tH; y++)
      for (let x = tx*tW; x < (tx+1)*tW; x++) hist[src[y*width+x]]++;
    let excess = 0;
    for (let b = 0; b < 256; b++) { if (hist[b]>clip) { excess+=hist[b]-clip; hist[b]=clip; } }
    const spread = Math.floor(excess/256);
    for (let b = 0; b < 256; b++) hist[b] += spread;
    let cdf=0, cdfMin=-1; const lut=luts[ty][tx];
    for (let b = 0; b < 256; b++) {
      cdf+=hist[b]; if (cdfMin<0&&cdf>0) cdfMin=cdf;
      lut[b]=tSz-cdfMin>0 ? Math.max(0,Math.min(255,Math.round((cdf-cdfMin)*255/(tSz-cdfMin)))) : 0;
    }
  }
  const dst = new Uint8Array(src.length);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const v=src[y*width+x], txF=x/tW-0.5, tyF=y/tH-0.5;
    const tx0=Math.max(0,Math.min(TX-1,Math.floor(txF))), ty0=Math.max(0,Math.min(TY-1,Math.floor(tyF)));
    const tx1=Math.min(TX-1,tx0+1), ty1=Math.min(TY-1,ty0+1);
    const wx=Math.max(0,Math.min(1,txF-tx0)), wy=Math.max(0,Math.min(1,tyF-ty0));
    dst[y*width+x]=Math.round(
      luts[ty0][tx0][v]*(1-wx)*(1-wy)+luts[ty0][tx1][v]*wx*(1-wy)+
      luts[ty1][tx0][v]*(1-wx)*wy    +luts[ty1][tx1][v]*wx*wy
    );
  }
  return dst;
}

// ══════════════════════════════════════════════════════════════
// §07  Multi-scale CLAHE blend
// ══════════════════════════════════════════════════════════════
function multiScaleClahe(src, width, height) {
  const fine   = runClahe(src, width, height, C.CLAHE_TILES_FINE,   C.CLAHE_CLIP);
  const coarse = runClahe(src, width, height, C.CLAHE_TILES_COARSE, C.CLAHE_CLIP * 0.8);
  const blend  = C.CLAHE_BLEND;
  const dst    = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i++)
    dst[i] = Math.round(fine[i]*blend + coarse[i]*(1-blend));
  return dst;
}

// ══════════════════════════════════════════════════════════════
// §08  Shadow gamma lift
//      Non-linear: f(x) = 255 * (x/255)^gamma
//      gamma < 1 lifts shadows; gamma = 1 is linear (no change)
//      In near-pitch-black we go to gamma=0.35 which is
//      equivalent to a very aggressive shadow recovery curve.
// ══════════════════════════════════════════════════════════════
function buildGammaLut(gamma) {
  const lut = new Uint8Array(256);
  for (let i = 0; i < 256; i++)
    lut[i] = Math.round(255 * Math.pow(i / 255, gamma));
  return lut;
}

function applyGammaLut(luma, lut) {
  const out = new Uint8Array(luma.length);
  for (let i = 0; i < luma.length; i++) out[i] = lut[luma[i]];
  return out;
}

// ══════════════════════════════════════════════════════════════
// §09  Unsharp mask — 5×5 Gaussian blur → subtract at strength
//      Restores edge crispness lost during multi-frame averaging.
// ══════════════════════════════════════════════════════════════
const GAUSS5 = [
  1,  4,  7,  4, 1,
  4, 16, 26, 16, 4,
  7, 26, 41, 26, 7,
  4, 16, 26, 16, 4,
  1,  4,  7,  4, 1,
];
const GAUSS5_SUM = GAUSS5.reduce((a, b) => a + b, 0); // 273

function unsharpMask(luma, width, height, strength) {
  const blur = new Float32Array(luma.length);
  for (let y = 2; y < height - 2; y++) {
    for (let x = 2; x < width - 2; x++) {
      let acc = 0;
      for (let ky = -2; ky <= 2; ky++)
        for (let kx = -2; kx <= 2; kx++)
          acc += luma[(y+ky)*width+(x+kx)] * GAUSS5[(ky+2)*5+(kx+2)];
      blur[y*width+x] = acc / GAUSS5_SUM;
    }
  }
  const out = new Uint8Array(luma.length);
  for (let i = 0; i < luma.length; i++)
    out[i] = Math.max(0, Math.min(255, Math.round(luma[i] + strength*(luma[i]-blur[i]))));
  return out;
}

// ══════════════════════════════════════════════════════════════
// §10  HELPERS
// ══════════════════════════════════════════════════════════════
function extractLuma(data, width, height) {
  const luma = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++)
    luma[i] = Math.round(0.299*data[i*4] + 0.587*data[i*4+1] + 0.114*data[i*4+2]);
  return luma;
}

function meanOf(luma) {
  let s = 0; for (let i = 0; i < luma.length; i++) s += luma[i];
  return s / luma.length;
}

function lumaToImageData(luma, width, height, ctx) {
  const id = ctx.createImageData(width, height);
  for (let i = 0; i < luma.length; i++) {
    const v = luma[i];
    id.data[i*4]=v; id.data[i*4+1]=v; id.data[i*4+2]=v; id.data[i*4+3]=255;
  }
  return id;
}

// ══════════════════════════════════════════════════════════════
// §11  COMPONENT
// ══════════════════════════════════════════════════════════════
export default function DarkVision() {
  const videoRef   = useRef(null);
  const srcRef     = useRef(null);
  const outRef     = useRef(null);
  const motionRef  = useRef(new MotionEstimator());
  const stackRef   = useRef(null);
  const rafRef     = useRef(null);
  const timerRef   = useRef(null);
  const streamRef  = useRef(null);
  const imgCapRef  = useRef(null);  // ImageCapture instance
  const showRawRef = useRef(false);

  const [status,  setStatus]  = useState("idle");
  const [hud,     setHud]     = useState(null);
  const [showRaw, setShowRaw] = useState(false);
  const [hwExp,   setHwExp]   = useState(false);

  useEffect(() => { showRawRef.current = showRaw; }, [showRaw]);

  const startCamera = useCallback(async () => {
    setStatus("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: C.WIDTH }, height: { ideal: C.HEIGHT }, facingMode: "environment" },
        audio: false,
      });
      streamRef.current = stream;
      const track = stream.getVideoTracks()[0];

      // Hardware exposure
      const gotHw = await applyHardwareExposure(track);
      setHwExp(gotHw);

      // ImageCapture for grabFrame() — avoids live-stream frame-rate limit
      if (typeof ImageCapture !== "undefined") {
        imgCapRef.current = new ImageCapture(track);
      }

      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      stackRef.current = new StackEngine(C.WIDTH, C.HEIGHT);
      setStatus("live");
      beginPipeline();
    } catch (e) {
      console.error(e);
      setStatus("error");
    }
  }, []);

  const stopCamera = useCallback(() => {
    clearInterval(timerRef.current);
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    imgCapRef.current = null;
    setStatus("idle");
    setHud(null);
  }, []);

  const beginPipeline = useCallback(() => {
    const video  = videoRef.current;
    const srcCtx = srcRef.current.getContext("2d", { willReadFrequently: true });
    const outCtx = outRef.current.getContext("2d");
    const motion = motionRef.current;
    const stack  = stackRef.current;
    let fCount   = 0, t0 = performance.now();

    // Gamma LUT cache — recomputed when darkness changes
    let lastGamma = -1, gammaLut = null;

    async function processFrame() {
      const W = C.WIDTH, H = C.HEIGHT;
      let imgData;

      // Prefer grabFrame() — gives sensor more integration time
      try {
        if (imgCapRef.current) {
          const bmp = await imgCapRef.current.grabFrame();
          srcCtx.drawImage(bmp, 0, 0, W, H);
          bmp.close?.();
        } else {
          srcCtx.drawImage(video, 0, 0, W, H);
        }
      } catch (_) {
        srcCtx.drawImage(video, 0, 0, W, H);
      }

      imgData = srcCtx.getImageData(0, 0, W, H);
      const rawLuma = extractLuma(imgData.data, W, H);
      const mean    = meanOf(rawLuma);

      // A. Hot-pixel suppression before stacking
      const clean = medianFilter(rawLuma, W, H);

      if (stack.count === 0) motion.lockReference(clean);

      // B. Motion align
      const { dx, dy, magnitude } = motion.estimate(clean, W, H);
      const aligned = motion.align(clean, W, H, dx, dy);

      // C. Accumulate
      stack.accumulate(aligned, magnitude, mean);
      fCount++;

      if (!stack.allZonesDone()) return;

      const t1  = performance.now();
      const avg = stack.exportAndReset();

      // D. Shadow gamma lift (adaptive to darkness)
      const t        = Math.min(1, mean / C.DARK_THRESHOLD);
      const gamma    = C.GAMMA_DARK + t * (C.GAMMA_BRIGHT - C.GAMMA_DARK);
      const gRounded = Math.round(gamma * 100) / 100;
      if (gRounded !== lastGamma) { gammaLut = buildGammaLut(gRounded); lastGamma = gRounded; }
      const lifted = applyGammaLut(avg, gammaLut);

      // E. Multi-scale CLAHE
      const eq = multiScaleClahe(lifted, W, H);

      // F. Unsharp mask
      const sharp = unsharpMask(eq, W, H, C.USM_STRENGTH);

      if (!showRawRef.current) {
        outCtx.putImageData(lumaToImageData(sharp, W, H, outCtx), 0, 0);
      } else {
        outCtx.putImageData(imgData, 0, 0);
      }

      setHud({
        mean:   mean.toFixed(1),
        stack:  stack.targetDepth(mean),
        gamma:  gamma.toFixed(2),
        dx:     dx.toFixed(1),
        dy:     dy.toFixed(1),
        mag:    magnitude.toFixed(1),
        procMs: Math.round(t1 - t0),
        fps:    fCount,
      });
      fCount = 0; t0 = t1;
    }

    // Use setInterval so grabFrame() awaits don't stack up
    timerRef.current = setInterval(() => {
      processFrame().catch(console.error);
    }, C.GRAB_INTERVAL_MS);
  }, []);

  useEffect(() => () => {
    clearInterval(timerRef.current);
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
  }, []);

  const toggleRaw = useCallback(() => setShowRaw(v => !v), []);

  return (
    <div style={s.root}>
      <video ref={videoRef} style={{ display:"none" }} playsInline muted />
      <canvas ref={srcRef} width={C.WIDTH} height={C.HEIGHT} style={{ display:"none" }} />

      <header style={s.header}>
        <span style={s.hTitle}>▌ DARK VISION</span>
        <span style={s.hSub}>THE BIG THREE</span>
        {hwExp && <span style={s.hwBadge}>HW EXP ●</span>}
      </header>

      <div style={s.viewport}>
        <canvas ref={outRef} width={C.WIDTH} height={C.HEIGHT} style={s.canvas} />

        {status === "idle" && (
          <div style={s.overlay}>
            <div style={s.splash}>
              <div style={s.splashTitle}>DARK VISION</div>
              <div style={s.splashSub}>THE BIG THREE</div>
              <div style={s.splashDesc}>
                Hot-pixel suppression · 48-frame stack<br/>
                Gamma shadow lift · Multi-scale CLAHE<br/>
                Unsharp mask · Motion alignment
              </div>
              <button style={s.btn} onClick={startCamera}>▶ START CAMERA</button>
            </div>
          </div>
        )}

        {status === "starting" && (
          <div style={s.overlay}>
            <div style={s.dimText}>INITIALISING PIPELINE…</div>
          </div>
        )}

        {status === "error" && (
          <div style={s.overlay}>
            <div style={s.splash}>
              <div style={{ ...s.splashTitle, color:"#ff4444" }}>CAMERA UNAVAILABLE</div>
              <div style={{ ...s.splashDesc, color:"#884444" }}>Check browser permissions.</div>
              <button style={s.btn} onClick={startCamera}>↺ RETRY</button>
            </div>
          </div>
        )}

        {hud && status === "live" && (
          <div style={s.hud}>
            <div style={s.hudH}>▌ DARK VISION  BIG THREE</div>
            <R label="BRIGHTNESS" v={`${hud.mean} luma`}            dim />
            <R label="STACK"      v={`${hud.stack} frames`}              />
            <R label="GAMMA"      v={`γ ${hud.gamma}`}                    />
            <R label="MOTION"     v={`Δ${hud.dx},${hud.dy} m${hud.mag}`} dim />
            <R label="PROC"       v={`${hud.procMs}ms  ${hud.fps}f`}      dim />
          </div>
        )}

        {status === "live" && (
          <button
            style={{ ...s.rawBtn, background: showRaw ? G : "rgba(0,0,0,0.6)", color: showRaw ? "#000" : G }}
            onClick={toggleRaw}
          >
            {showRaw ? "RAW ON" : "RAW OFF"}
          </button>
        )}
      </div>

      {status === "live" && (
        <div style={s.ctrl}>
          <button style={s.stopBtn} onClick={stopCamera}>■ STOP</button>
        </div>
      )}

      <footer style={s.footer}>
        <Leg n="A" t="Hot-pixel suppression"   d="3×3 median pre-filter. Removes sensor defects before stacking." />
        <Leg n="B" t="48-frame stack"           d="grabFrame() polling. σ-clip + motion weighting. SNR gain ×6.9." />
        <Leg n="C" t="Shadow gamma lift"        d="γ=0.35 in pitch black. Non-linear — lifts shadows, preserves midtones." />
        <Leg n="D" t="Multi-scale CLAHE"        d="Tile-8 + tile-4 blend. Fine local contrast + broad regional structure." />
        <Leg n="E" t="Unsharp mask"             d="5×5 Gaussian USM at 0.65 strength. Recovers edge detail after averaging." />
      </footer>
    </div>
  );
}

const R   = ({ label, v, dim }) => (
  <div style={{ display:"flex", gap:8, marginBottom:2 }}>
    <span style={{ ...s.hudL, opacity: dim?0.55:1 }}>{label}</span>
    <span style={{ ...s.hudV, opacity: dim?0.7:1  }}>{v}</span>
  </div>
);
const Leg = ({ n, t, d }) => (
  <div style={s.legRow}>
    <span style={s.legN}>{n}</span>
    <div><div style={s.legT}>{t}</div><div style={s.legD}>{d}</div></div>
  </div>
);

const G    = "#39ff14";
const DIM  = "#1a7a09";
const BG   = "#020702";
const PNL  = "rgba(2,10,2,0.90)";
const MONO = "'Courier New', Courier, monospace";

const s = {
  root:       { background:BG, minHeight:"100vh", display:"flex", flexDirection:"column", alignItems:"center", fontFamily:MONO, color:G, padding:"12px 8px 28px" },
  header:     { width:"100%", maxWidth:680, display:"flex", alignItems:"baseline", gap:10, paddingBottom:8, borderBottom:`1px solid ${DIM}`, marginBottom:10 },
  hTitle:     { fontSize:18, fontWeight:"bold", letterSpacing:3 },
  hSub:       { fontSize:10, color:DIM, letterSpacing:4 },
  hwBadge:    { marginLeft:"auto", fontSize:9, color:G, letterSpacing:1, border:`1px solid ${DIM}`, padding:"2px 6px" },
  viewport:   { position:"relative", width:"100%", maxWidth:680, aspectRatio:"4/3", background:"#000", border:`1px solid ${DIM}`, overflow:"hidden" },
  canvas:     { width:"100%", height:"100%", display:"block", imageRendering:"pixelated" },
  overlay:    { position:"absolute", inset:0, display:"flex", alignItems:"center", justifyContent:"center", background:"rgba(0,0,0,0.86)" },
  splash:     { textAlign:"center", display:"flex", flexDirection:"column", alignItems:"center", gap:10 },
  splashTitle:{ fontSize:30, fontWeight:"bold", letterSpacing:6, textShadow:`0 0 24px ${G}` },
  splashSub:  { fontSize:10, letterSpacing:6, color:DIM, marginTop:-6 },
  splashDesc: { fontSize:12, color:DIM, lineHeight:1.9, marginTop:4 },
  dimText:    { fontSize:12, letterSpacing:3, color:DIM },
  btn:        { marginTop:8, background:"transparent", border:`1px solid ${G}`, color:G, fontFamily:MONO, fontSize:13, letterSpacing:2, padding:"8px 22px", cursor:"pointer" },
  stopBtn:    { background:"transparent", border:"1px solid #ff4444", color:"#ff4444", fontFamily:MONO, fontSize:11, letterSpacing:2, padding:"5px 16px", cursor:"pointer", marginTop:8 },
  ctrl:       { width:"100%", maxWidth:680, display:"flex", justifyContent:"flex-end" },
  hud:        { position:"absolute", top:10, left:10, background:PNL, border:`1px solid ${DIM}`, padding:"9px 13px", fontSize:11, lineHeight:1.6, minWidth:240, backdropFilter:"blur(4px)" },
  hudH:       { fontSize:9, letterSpacing:2, color:G, marginBottom:5, fontWeight:"bold" },
  hudL:       { color:DIM, minWidth:74, display:"inline-block", fontSize:9, letterSpacing:1 },
  hudV:       { color:G, fontSize:10 },
  rawBtn:     { position:"absolute", bottom:10, right:10, border:`1px solid ${G}`, fontFamily:MONO, fontSize:9, letterSpacing:2, padding:"4px 9px", cursor:"pointer", transition:"background 0.1s,color 0.1s" },
  footer:     { width:"100%", maxWidth:680, borderTop:`1px solid ${DIM}`, marginTop:16, paddingTop:14, display:"flex", flexDirection:"column", gap:10 },
  legRow:     { display:"flex", gap:14, alignItems:"flex-start" },
  legN:       { color:G, fontSize:13, fontWeight:"bold", minWidth:20, paddingTop:1 },
  legT:       { color:G, fontSize:10, letterSpacing:1, marginBottom:1 },
  legD:       { color:DIM, fontSize:9, lineHeight:1.5 },
};
