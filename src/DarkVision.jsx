/**
 * ╔══════════════════════════════════════════════════════════════╗
 * ║            DARK VISION — THE BIG THREE  (Browser Port)      ║
 * ║  #1  Manual ISO + Exposure  (ImageCapture API where avail)  ║
 * ║  #2  Multi-Frame Stacking + Motion Alignment                ║
 * ║  #3  CLAHE Adaptive Histogram Equalisation                  ║
 * ╚══════════════════════════════════════════════════════════════╝
 */

import { useState, useRef, useCallback, useEffect } from "react";

// ══════════════════════════════════════════════════════════════
// §01  CONSTANTS
// ══════════════════════════════════════════════════════════════
const C = {
  WIDTH:  640,
  HEIGHT: 480,

  STACK_MIN: 6,
  STACK_MAX: 24,

  MAX_SHIFT_PX: 30,
  STABLE_MOT_THRESHOLD: 5.0,

  SIGMA_CLIP: 2.0,

  CLAHE_TILES: 8,
  CLAHE_CLIP:  3.5,

  GAIN_MAX: 6.0,
  GAIN_MIN: 1.0,

  DARK_THRESHOLD: 40,
};

// ══════════════════════════════════════════════════════════════
// §02  ImageCapture exposure control
//      Attempts to set real ISO + shutter on Chrome Android.
//      Silent no-op on browsers that don't support it.
// ══════════════════════════════════════════════════════════════
async function applyHardwareExposure(track) {
  try {
    const caps = track.getCapabilities?.();
    if (!caps) return false;

    const constraints = {};
    let changed = false;

    if (caps.exposureMode?.includes?.("manual")) {
      constraints.exposureMode = "manual";
      changed = true;
    }
    if (caps.exposureTime) {
      // Push to maximum allowed exposure
      constraints.exposureTime = caps.exposureTime.max;
      changed = true;
    }
    if (caps.iso) {
      constraints.iso = caps.iso.max;
      changed = true;
    }
    if (caps.whiteBalanceMode?.includes?.("manual")) {
      constraints.whiteBalanceMode = "manual";
    }
    if (caps.focusMode?.includes?.("manual")) {
      constraints.focusMode = "manual";
      constraints.focusDistance = 0; // infinity
    }

    if (changed) {
      await track.applyConstraints({ advanced: [constraints] });
      return true;
    }
  } catch (e) {
    // Not supported — silent
  }
  return false;
}

// ══════════════════════════════════════════════════════════════
// §03  MotionEstimator
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
        if (diff > 8) {
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
    const s  = 0.12;
    const dx = Math.max(-C.MAX_SHIFT_PX, Math.min(C.MAX_SHIFT_PX, (sumDx / sumW) * s));
    const dy = Math.max(-C.MAX_SHIFT_PX, Math.min(C.MAX_SHIFT_PX, (sumDy / sumW) * s));
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
          luma[y0 * width + x0] * (1 - fx) * (1 - fy) +
          luma[y0 * width + x1] *      fx  * (1 - fy) +
          luma[y1 * width + x0] * (1 - fx) *      fy  +
          luma[y1 * width + x1] *      fx  *      fy
        );
      }
    }
    return dst;
  }
}

// ══════════════════════════════════════════════════════════════
// §04  StackEngine
// ══════════════════════════════════════════════════════════════
class StackEngine {
  constructor(w, h) { this.w = w; this.h = h; this.reset(); }

  reset() {
    const sz = this.w * this.h;
    this.weightedSum = new Float32Array(sz);
    this.weightSum   = new Float32Array(sz);
    this.runMean     = new Float32Array(sz);
    this.runM2       = new Float32Array(sz);
    this.count = 0;
    this.zoneTargets = null;
    this.zoneCounts  = null;
    this.zoneMeta    = null;
  }

  targetDepth(meanLuma) {
    const t = Math.min(1, meanLuma / C.DARK_THRESHOLD);
    return Math.round(C.STACK_MAX - t * (C.STACK_MAX - C.STACK_MIN));
  }

  _computeZoneTargets(luma, globalDepth) {
    const COLS = 4, ROWS = 4;
    const zw = Math.floor(this.w / COLS);
    const zh = Math.floor(this.h / ROWS);
    const targets = new Int32Array(COLS * ROWS);
    const counts  = new Int32Array(COLS * ROWS);
    for (let zy = 0; zy < ROWS; zy++) {
      for (let zx = 0; zx < COLS; zx++) {
        let sum = 0;
        for (let y = zy * zh; y < (zy + 1) * zh; y++)
          for (let x = zx * zw; x < (zx + 1) * zw; x++)
            sum += luma[y * this.w + x];
        const t = Math.min(1, (sum / (zw * zh)) / 200);
        targets[zy * COLS + zx] = Math.min(globalDepth,
          Math.round(C.STACK_MAX - t * (C.STACK_MAX - C.STACK_MIN)));
      }
    }
    return { targets, counts, COLS, ROWS, zw, zh };
  }

  accumulate(luma, motionMag, meanLuma) {
    const n = this.count + 1;
    if (this.count === 0) {
      const z = this._computeZoneTargets(luma, this.targetDepth(meanLuma));
      this.zoneTargets = z.targets;
      this.zoneCounts  = z.counts;
      this.zoneMeta    = z;
    }
    const w = 1 / (1 + motionMag / C.STABLE_MOT_THRESHOLD);
    for (let i = 0; i < luma.length; i++) {
      const pf      = luma[i];
      const oldMean = this.runMean[i];
      const delta   = pf - oldMean;
      this.runMean[i] += delta / n;
      this.runM2[i]   += delta * (pf - this.runMean[i]);
      const sigma     = n > 1 ? Math.sqrt(this.runM2[i] / (n - 1)) : Infinity;
      if (!(n > 2 && Math.abs(pf - oldMean) > C.SIGMA_CLIP * sigma)) {
        this.weightedSum[i] += pf * w;
        this.weightSum[i]   += w;
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
        ? Math.max(0, Math.min(255, Math.round(this.weightedSum[i] / this.weightSum[i])))
        : 0;
    this.reset();
    return out;
  }
}

// ══════════════════════════════════════════════════════════════
// §05  ClaheEngine
// ══════════════════════════════════════════════════════════════
class ClaheEngine {
  equalise(src, width, height) {
    const TX = C.CLAHE_TILES, TY = C.CLAHE_TILES;
    const tW = Math.floor(width / TX), tH = Math.floor(height / TY);
    const tSz = tW * tH;
    const clip = Math.max(1, Math.floor(C.CLAHE_CLIP * tSz / 256));

    const luts = Array.from({ length: TY }, () =>
      Array.from({ length: TX }, () => new Uint8Array(256)));

    for (let ty = 0; ty < TY; ty++) {
      for (let tx = 0; tx < TX; tx++) {
        const hist = new Int32Array(256);
        for (let y = ty * tH; y < (ty + 1) * tH; y++)
          for (let x = tx * tW; x < (tx + 1) * tW; x++)
            hist[src[y * width + x]]++;
        let excess = 0;
        for (let b = 0; b < 256; b++) {
          if (hist[b] > clip) { excess += hist[b] - clip; hist[b] = clip; }
        }
        const spread = Math.floor(excess / 256);
        for (let b = 0; b < 256; b++) hist[b] += spread;
        let cdf = 0, cdfMin = -1;
        const lut = luts[ty][tx];
        for (let b = 0; b < 256; b++) {
          cdf += hist[b];
          if (cdfMin < 0 && cdf > 0) cdfMin = cdf;
          lut[b] = tSz - cdfMin > 0
            ? Math.max(0, Math.min(255, Math.round((cdf - cdfMin) * 255 / (tSz - cdfMin))))
            : 0;
        }
      }
    }

    const dst = new Uint8Array(src.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const v   = src[y * width + x];
        const txF = x / tW - 0.5, tyF = y / tH - 0.5;
        const tx0 = Math.max(0, Math.min(TX - 1, Math.floor(txF)));
        const ty0 = Math.max(0, Math.min(TY - 1, Math.floor(tyF)));
        const tx1 = Math.min(TX - 1, tx0 + 1);
        const ty1 = Math.min(TY - 1, ty0 + 1);
        const wx  = Math.max(0, Math.min(1, txF - tx0));
        const wy  = Math.max(0, Math.min(1, tyF - ty0));
        dst[y * width + x] = Math.round(
          luts[ty0][tx0][v] * (1 - wx) * (1 - wy) +
          luts[ty0][tx1][v] *      wx  * (1 - wy) +
          luts[ty1][tx0][v] * (1 - wx) *      wy  +
          luts[ty1][tx1][v] *      wx  *      wy
        );
      }
    }
    return dst;
  }
}

// ══════════════════════════════════════════════════════════════
// §06  HELPERS
// ══════════════════════════════════════════════════════════════
function extractLuma(data, width, height) {
  const luma = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++)
    luma[i] = Math.round(0.299 * data[i*4] + 0.587 * data[i*4+1] + 0.114 * data[i*4+2]);
  return luma;
}

function meanOf(luma) {
  let s = 0;
  for (let i = 0; i < luma.length; i++) s += luma[i];
  return s / luma.length;
}

function applyGain(luma, meanL) {
  const t    = Math.min(1, meanL / C.DARK_THRESHOLD);
  const gain = C.GAIN_MAX - t * (C.GAIN_MAX - C.GAIN_MIN);
  const out  = new Uint8Array(luma.length);
  for (let i = 0; i < luma.length; i++)
    out[i] = Math.min(255, Math.round(luma[i] * gain));
  return { gained: out, gain };
}

function lumaToImageData(luma, width, height, ctx) {
  const id = ctx.createImageData(width, height);
  for (let i = 0; i < luma.length; i++) {
    const v = luma[i];
    id.data[i*4] = v; id.data[i*4+1] = v; id.data[i*4+2] = v; id.data[i*4+3] = 255;
  }
  return id;
}

// ══════════════════════════════════════════════════════════════
// §07  COMPONENT
// ══════════════════════════════════════════════════════════════
export default function DarkVision() {
  const videoRef   = useRef(null);
  const srcRef     = useRef(null);
  const outRef     = useRef(null);
  const motionRef  = useRef(new MotionEstimator());
  const stackRef   = useRef(null);
  const claheRef   = useRef(new ClaheEngine());
  const rafRef     = useRef(null);
  const streamRef  = useRef(null);
  const showRawRef = useRef(false);
  const hwExpRef   = useRef(false);

  const [status,  setStatus]  = useState("idle");
  const [hud,     setHud]     = useState(null);
  const [showRaw, setShowRaw] = useState(false);
  const [hwExp,   setHwExp]   = useState(false);

  useEffect(() => { showRawRef.current = showRaw; }, [showRaw]);

  const startCamera = useCallback(async () => {
    setStatus("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width:      { ideal: C.WIDTH },
          height:     { ideal: C.HEIGHT },
          facingMode: "environment",
        },
        audio: false,
      });
      streamRef.current = stream;

      // Attempt hardware exposure control (Chrome Android)
      const track = stream.getVideoTracks()[0];
      const gotHw = await applyHardwareExposure(track);
      hwExpRef.current = gotHw;
      setHwExp(gotHw);

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
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    setStatus("idle");
    setHud(null);
  }, []);

  const beginPipeline = useCallback(() => {
    const video  = videoRef.current;
    const srcCtx = srcRef.current.getContext("2d", { willReadFrequently: true });
    const outCtx = outRef.current.getContext("2d");
    const motion = motionRef.current;
    const clahe  = claheRef.current;
    const stack  = stackRef.current;
    let fCount = 0, t0 = performance.now();

    function tick() {
      if (!video || video.readyState < 2) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }
      const W = C.WIDTH, H = C.HEIGHT;
      srcCtx.drawImage(video, 0, 0, W, H);
      const imgData = srcCtx.getImageData(0, 0, W, H);
      const raw     = extractLuma(imgData.data, W, H);
      const mean    = meanOf(raw);

      if (stack.count === 0) motion.lockReference(raw);

      const { dx, dy, magnitude } = motion.estimate(raw, W, H);
      const aligned = motion.align(raw, W, H, dx, dy);
      stack.accumulate(aligned, magnitude, mean);
      fCount++;

      if (stack.allZonesDone()) {
        const t1  = performance.now();
        const avg = stack.exportAndReset();
        const { gained, gain } = applyGain(avg, mean);
        const eq  = clahe.equalise(gained, W, H);

        if (!showRawRef.current) {
          outCtx.putImageData(lumaToImageData(eq, W, H, outCtx), 0, 0);
        } else {
          outCtx.putImageData(imgData, 0, 0);
        }

        setHud({
          mean:   mean.toFixed(1),
          stack:  stack.targetDepth(mean),
          gain:   gain.toFixed(2),
          dx:     dx.toFixed(1),
          dy:     dy.toFixed(1),
          mag:    magnitude.toFixed(1),
          procMs: Math.round(t1 - t0),
          fps:    fCount,
        });
        fCount = 0; t0 = t1;
      }
      rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  useEffect(() => () => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
  }, []);

  const toggleRaw = useCallback(() => setShowRaw(v => !v), []);

  return (
    <div style={s.root}>
      <video ref={videoRef} style={{ display: "none" }} playsInline muted />
      <canvas ref={srcRef} width={C.WIDTH} height={C.HEIGHT} style={{ display: "none" }} />

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
                Manual exposure · Frame stacking<br />
                Motion alignment · CLAHE contrast
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
              <div style={{ ...s.splashTitle, color: "#ff4444" }}>CAMERA UNAVAILABLE</div>
              <div style={{ ...s.splashDesc, color: "#884444" }}>Check browser permissions.</div>
              <button style={s.btn} onClick={startCamera}>↺ RETRY</button>
            </div>
          </div>
        )}

        {hud && status === "live" && (
          <div style={s.hud}>
            <div style={s.hudH}>▌ DARK VISION  BIG THREE</div>
            <R label="BRIGHTNESS" v={`${hud.mean} luma`}    dim />
            <R label="STACK"      v={`${hud.stack} fr  ×${hud.gain}`} />
            <R label="MOTION"     v={`Δ${hud.dx},${hud.dy}  m${hud.mag}`} dim />
            <R label="PROC"       v={`${hud.procMs}ms  ${hud.fps}f`}   dim />
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
        <Leg n="1" t="Manual Exposure"  d="Max ISO + shutter via ImageCapture API. Digital gain ×6 in darkness." />
        <Leg n="2" t="Frame Stacking"   d="Motion-aligned accumulation. Welford σ-clip removes transients. SNR ∝ √N." />
        <Leg n="3" t="CLAHE"            d="8×8 tile adaptive histogram equalisation with bilinear LUT interpolation." />
      </footer>
    </div>
  );
}

const R = ({ label, v, dim }) => (
  <div style={{ display: "flex", gap: 8, marginBottom: 2 }}>
    <span style={{ ...s.hudL, opacity: dim ? 0.55 : 1 }}>{label}</span>
    <span style={{ ...s.hudV, opacity: dim ? 0.7  : 1 }}>{v}</span>
  </div>
);

const Leg = ({ n, t, d }) => (
  <div style={s.legRow}>
    <span style={s.legN}>#{n}</span>
    <div><div style={s.legT}>{t}</div><div style={s.legD}>{d}</div></div>
  </div>
);

const G    = "#39ff14";
const DIM  = "#1a7a09";
const BG   = "#020702";
const PNL  = "rgba(2,10,2,0.90)";
const MONO = "'Courier New', Courier, monospace";

const s = {
  root:       { background: BG, minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", fontFamily: MONO, color: G, padding: "12px 8px 28px" },
  header:     { width: "100%", maxWidth: 680, display: "flex", alignItems: "baseline", gap: 10, paddingBottom: 8, borderBottom: `1px solid ${DIM}`, marginBottom: 10 },
  hTitle:     { fontSize: 18, fontWeight: "bold", letterSpacing: 3 },
  hSub:       { fontSize: 10, color: DIM, letterSpacing: 4 },
  hwBadge:    { marginLeft: "auto", fontSize: 9, color: G, letterSpacing: 1, border: `1px solid ${DIM}`, padding: "2px 6px" },
  viewport:   { position: "relative", width: "100%", maxWidth: 680, aspectRatio: "4/3", background: "#000", border: `1px solid ${DIM}`, overflow: "hidden" },
  canvas:     { width: "100%", height: "100%", display: "block", imageRendering: "pixelated" },
  overlay:    { position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.86)" },
  splash:     { textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", gap: 10 },
  splashTitle:{ fontSize: 30, fontWeight: "bold", letterSpacing: 6, textShadow: `0 0 24px ${G}` },
  splashSub:  { fontSize: 10, letterSpacing: 6, color: DIM, marginTop: -6 },
  splashDesc: { fontSize: 12, color: DIM, lineHeight: 1.8, marginTop: 4 },
  dimText:    { fontSize: 12, letterSpacing: 3, color: DIM },
  btn:        { marginTop: 8, background: "transparent", border: `1px solid ${G}`, color: G, fontFamily: MONO, fontSize: 13, letterSpacing: 2, padding: "8px 22px", cursor: "pointer" },
  stopBtn:    { background: "transparent", border: "1px solid #ff4444", color: "#ff4444", fontFamily: MONO, fontSize: 11, letterSpacing: 2, padding: "5px 16px", cursor: "pointer", marginTop: 8 },
  ctrl:       { width: "100%", maxWidth: 680, display: "flex", justifyContent: "flex-end" },
  hud:        { position: "absolute", top: 10, left: 10, background: PNL, border: `1px solid ${DIM}`, padding: "9px 13px", fontSize: 11, lineHeight: 1.6, minWidth: 240, backdropFilter: "blur(4px)" },
  hudH:       { fontSize: 9, letterSpacing: 2, color: G, marginBottom: 5, fontWeight: "bold" },
  hudL:       { color: DIM, minWidth: 74, display: "inline-block", fontSize: 9, letterSpacing: 1 },
  hudV:       { color: G, fontSize: 10 },
  rawBtn:     { position: "absolute", bottom: 10, right: 10, border: `1px solid ${G}`, fontFamily: MONO, fontSize: 9, letterSpacing: 2, padding: "4px 9px", cursor: "pointer", transition: "background 0.1s,color 0.1s" },
  footer:     { width: "100%", maxWidth: 680, borderTop: `1px solid ${DIM}`, marginTop: 16, paddingTop: 14, display: "flex", flexDirection: "column", gap: 10 },
  legRow:     { display: "flex", gap: 14, alignItems: "flex-start" },
  legN:       { color: G, fontSize: 13, fontWeight: "bold", minWidth: 20, paddingTop: 1 },
  legT:       { color: G, fontSize: 10, letterSpacing: 1, marginBottom: 1 },
  legD:       { color: DIM, fontSize: 9, lineHeight: 1.5 },
};
