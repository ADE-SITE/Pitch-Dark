/**
 * ╔══════════════════════════════════════════════════════════════╗
 * ║            DARK VISION — THE BIG THREE  (Browser Port)      ║
 * ║                                                             ║
 * ║  #1  Manual ISO + Exposure Maximisation  (getUserMedia)     ║
 * ║  #2  Multi-Frame Stacking + Motion Alignment                ║
 * ║  #3  CLAHE Adaptive Histogram Equalisation                  ║
 * ╚══════════════════════════════════════════════════════════════╝
 *
 * Ported from DarkVisionBigThree.kt by Andene.
 * No external models, APIs, or hardware. Pure browser signal chain.
 */

import { useState, useEffect, useRef, useCallback } from "react";

// ══════════════════════════════════════════════════════════════
// §01  CONSTANTS
// ══════════════════════════════════════════════════════════════
const C = {
  WIDTH:  640,
  HEIGHT: 480,

  // Stacking
  STACK_MIN: 4,
  STACK_MAX: 16,

  // Motion alignment (replaces gyro — uses optical flow centroid)
  MAX_SHIFT_PX:  30,

  // Per-frame weight stability threshold (motion magnitude 0-255 range)
  STABLE_MOT_THRESHOLD: 4.0,

  // Sigma clipping
  SIGMA_CLIP: 2.0,

  // CLAHE
  CLAHE_TILES: 8,
  CLAHE_CLIP:  3.5,

  // Digital pre-gain applied before CLAHE (simulates ISO boost)
  GAIN_MAX: 4.0,   // in full darkness
  GAIN_MIN: 1.0,   // in brightness

  // Brightness threshold: below this → max stack + gain
  DARK_THRESHOLD: 30,  // mean luma 0-255
};

// ══════════════════════════════════════════════════════════════
// §02  ENGINE — MotionEstimator (replaces GyroEngine)
//      Uses frame-diff centroid to estimate global camera shift.
//      Bilinear warp applied to each frame before stacking.
// ══════════════════════════════════════════════════════════════
class MotionEstimator {
  constructor() {
    this.refLuma = null;
    this.lastShiftX = 0;
    this.lastShiftY = 0;
    this.motionMag  = 0;
  }

  lockReference(luma) {
    this.refLuma = luma ? new Uint8Array(luma) : null;
    this.lastShiftX = 0;
    this.lastShiftY = 0;
  }

  /**
   * Estimate global translation between refLuma and luma
   * using the weighted centroid of the absolute-difference map.
   * Returns { dx, dy, magnitude }.
   */
  estimate(luma, width, height) {
    if (!this.refLuma) return { dx: 0, dy: 0, magnitude: 0 };

    const step = 4; // subsample for speed
    let sumW = 0, sumDx = 0, sumDy = 0, totalDiff = 0;

    for (let y = step; y < height - step; y += step) {
      for (let x = step; x < width - step; x += step) {
        const i = y * width + x;
        const diff = Math.abs((luma[i] & 0xFF) - (this.refLuma[i] & 0xFF));
        totalDiff += diff;
        if (diff > 8) {
          // Gradient-weighted contribution
          const gx = (luma[i + 1] & 0xFF) - (luma[i - 1] & 0xFF);
          const gy = (luma[i + width] & 0xFF) - (luma[i - width] & 0xFF);
          const w = diff;
          sumW  += w;
          sumDx += gx * w;
          sumDy += gy * w;
        }
      }
    }

    const magnitude = totalDiff / ((width / step) * (height / step));
    if (sumW < 1) return { dx: 0, dy: 0, magnitude };

    const scale = 0.12; // tuned to pixel-shift range
    const dx = Math.max(-C.MAX_SHIFT_PX, Math.min(C.MAX_SHIFT_PX, (sumDx / sumW) * scale));
    const dy = Math.max(-C.MAX_SHIFT_PX, Math.min(C.MAX_SHIFT_PX, (sumDy / sumW) * scale));

    this.lastShiftX = dx;
    this.lastShiftY = dy;
    this.motionMag  = magnitude;
    return { dx, dy, magnitude };
  }

  /**
   * Bilinear warp: translate luma by (-dx, -dy) to align with reference.
   * Returns a new Uint8Array.
   */
  align(luma, width, height, dx, dy) {
    if (Math.abs(dx) < 0.01 && Math.abs(dy) < 0.01) return luma;

    const dst = new Uint8Array(luma.length);
    const ox = -dx;
    const oy = -dy;

    for (let y = 0; y < height; y++) {
      const sy = y + oy;
      const y0 = Math.max(0, Math.min(height - 1, Math.floor(sy)));
      const y1 = Math.max(0, Math.min(height - 1, y0 + 1));
      const fy = sy - y0;

      for (let x = 0; x < width; x++) {
        const sx = x + ox;
        const x0 = Math.max(0, Math.min(width - 1, Math.floor(sx)));
        const x1 = Math.max(0, Math.min(width - 1, x0 + 1));
        const fx = sx - x0;

        const v00 = luma[y0 * width + x0];
        const v10 = luma[y0 * width + x1];
        const v01 = luma[y1 * width + x0];
        const v11 = luma[y1 * width + x1];

        dst[y * width + x] = Math.round(
          v00 * (1 - fx) * (1 - fy) +
          v10 *      fx  * (1 - fy) +
          v01 * (1 - fx) *      fy  +
          v11 *      fx  *      fy
        );
      }
    }
    return dst;
  }
}

// ══════════════════════════════════════════════════════════════
// §03  ENGINE — StackEngine
//      Weighted accumulator with per-pixel sigma clipping.
//      Zone-adaptive stack depth (#1 from Kotlin §05).
// ══════════════════════════════════════════════════════════════
class StackEngine {
  constructor(width, height) {
    this.w = width;
    this.h = height;
    this.reset();
  }

  reset() {
    const sz = this.w * this.h;
    this.weightedSum = new Float32Array(sz);
    this.weightSum   = new Float32Array(sz);
    this.runMean     = new Float32Array(sz);
    this.runM2       = new Float32Array(sz);
    this.count       = 0;
    this.zoneTargets = null;
    this.zoneCounts  = null;
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
        const mean = sum / (zw * zh);
        const t = Math.min(1, mean / 200);
        targets[zy * COLS + zx] = Math.min(
          globalDepth,
          Math.round(C.STACK_MAX - t * (C.STACK_MAX - C.STACK_MIN))
        );
      }
    }
    return { targets, counts, COLS, ROWS, zw, zh };
  }

  accumulate(luma, motionWeight, meanLuma) {
    const n = this.count + 1;

    if (this.count === 0) {
      const gd = this.targetDepth(meanLuma);
      const z  = this._computeZoneTargets(luma, gd);
      this.zoneTargets = z.targets;
      this.zoneCounts  = z.counts;
      this.zoneMeta    = z;
    }

    // Gyro-stability analogue: 1 / (1 + mag / threshold)
    const gyroWeight = 1 / (1 + motionWeight / C.STABLE_MOT_THRESHOLD);

    for (let i = 0; i < luma.length; i++) {
      const pf = luma[i];

      // Welford online variance
      const oldMean = this.runMean[i];
      const delta   = pf - oldMean;
      this.runMean[i] += delta / n;
      const delta2 = pf - this.runMean[i];
      this.runM2[i] += delta * delta2;

      const variance = n > 1 ? this.runM2[i] / (n - 1) : Infinity;
      const sigma    = Math.sqrt(variance);
      const isOutlier = n > 2 && Math.abs(pf - oldMean) > C.SIGMA_CLIP * sigma;

      if (!isOutlier) {
        this.weightedSum[i] += pf * gyroWeight;
        this.weightSum[i]   += gyroWeight;
      }
    }

    // Update zone counters
    if (this.zoneMeta) {
      for (let z = 0; z < this.zoneTargets.length; z++) {
        if (this.zoneCounts[z] < this.zoneTargets[z]) this.zoneCounts[z]++;
      }
    }

    this.count++;
  }

  allZonesDone() {
    if (!this.zoneCounts || !this.zoneTargets) return false;
    for (let z = 0; z < this.zoneTargets.length; z++) {
      if (this.zoneTargets[z] > 0 && this.zoneCounts[z] < this.zoneTargets[z])
        return false;
    }
    return true;
  }

  exportAndReset() {
    const out = new Uint8Array(this.weightedSum.length);
    for (let i = 0; i < out.length; i++) {
      out[i] = this.weightSum[i] > 0
        ? Math.max(0, Math.min(255, Math.round(this.weightedSum[i] / this.weightSum[i])))
        : 0;
    }
    this.reset();
    return out;
  }
}

// ══════════════════════════════════════════════════════════════
// §04  ENGINE — ClaheEngine
//      Tile-based adaptive histogram equalisation with bilinear
//      interpolation between tile LUTs. Matches Kotlin §06 exactly.
// ══════════════════════════════════════════════════════════════
class ClaheEngine {
  equalise(src, width, height) {
    const tilesX = C.CLAHE_TILES;
    const tilesY = C.CLAHE_TILES;
    const tileW  = Math.floor(width  / tilesX);
    const tileH  = Math.floor(height / tilesY);
    const tileSize  = tileW * tileH;
    const clipLimit = Math.max(1, Math.floor(C.CLAHE_CLIP * tileSize / 256));

    // Step 1: per-tile LUTs
    const luts = Array.from({ length: tilesY }, () =>
      Array.from({ length: tilesX }, () => new Uint8Array(256))
    );

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        const hist = new Int32Array(256);
        const x0 = tx * tileW, x1 = x0 + tileW;
        const y0 = ty * tileH, y1 = y0 + tileH;

        for (let y = y0; y < y1; y++)
          for (let x = x0; x < x1; x++)
            hist[src[y * width + x]]++;

        let excess = 0;
        for (let b = 0; b < 256; b++) {
          if (hist[b] > clipLimit) {
            excess += hist[b] - clipLimit;
            hist[b] = clipLimit;
          }
        }
        const spread = Math.floor(excess / 256);
        for (let b = 0; b < 256; b++) hist[b] += spread;

        let cdf = 0, cdfMin = -1;
        const lut = luts[ty][tx];
        for (let b = 0; b < 256; b++) {
          cdf += hist[b];
          if (cdfMin < 0 && cdf > 0) cdfMin = cdf;
          lut[b] = tileSize - cdfMin > 0
            ? Math.max(0, Math.min(255, Math.round((cdf - cdfMin) * 255 / (tileSize - cdfMin))))
            : 0;
        }
      }
    }

    // Step 2: bilinear interpolation
    const dst = new Uint8Array(src.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const v   = src[y * width + x];
        const txF = x / tileW - 0.5;
        const tyF = y / tileH - 0.5;
        const tx0 = Math.max(0, Math.min(tilesX - 1, Math.floor(txF)));
        const ty0 = Math.max(0, Math.min(tilesY - 1, Math.floor(tyF)));
        const tx1 = Math.min(tilesX - 1, tx0 + 1);
        const ty1 = Math.min(tilesY - 1, ty0 + 1);
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
// §05  HELPERS
// ══════════════════════════════════════════════════════════════

/** Extract Y-plane (luma) from ImageData RGBA. */
function extractLuma(imageData) {
  const { data, width, height } = imageData;
  const luma = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    luma[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }
  return luma;
}

/** Mean luma. */
function meanLuma(luma) {
  let sum = 0;
  for (let i = 0; i < luma.length; i++) sum += luma[i];
  return sum / luma.length;
}

/** Apply digital gain pre-CLAHE (mirrors Pipeline.computeGain). */
function applyGain(luma, meanL) {
  const t    = Math.min(1, meanL / C.DARK_THRESHOLD);
  const gain = C.GAIN_MAX - t * (C.GAIN_MAX - C.GAIN_MIN);
  const out  = new Uint8Array(luma.length);
  for (let i = 0; i < luma.length; i++)
    out[i] = Math.min(255, Math.round(luma[i] * gain));
  return { gained: out, gain };
}

/** Write luma back into a canvas as greyscale RGBA. */
function lumaToImageData(luma, width, height, ctx) {
  const id = ctx.createImageData(width, height);
  for (let i = 0; i < luma.length; i++) {
    const v = luma[i];
    id.data[i * 4]     = v;
    id.data[i * 4 + 1] = v;
    id.data[i * 4 + 2] = v;
    id.data[i * 4 + 3] = 255;
  }
  return id;
}

// ══════════════════════════════════════════════════════════════
// §06  MAIN COMPONENT
// ══════════════════════════════════════════════════════════════
export default function DarkVision() {
  const videoRef   = useRef(null);
  const srcCanvasRef = useRef(null);   // hidden — captures video frames
  const outCanvasRef = useRef(null);   // visible — shows processed output

  const motionRef = useRef(new MotionEstimator());
  const stackRef  = useRef(null);
  const claheRef  = useRef(new ClaheEngine());

  const rafRef    = useRef(null);
  const streamRef = useRef(null);

  const [status, setStatus]   = useState("idle");  // idle | starting | live | error
  const [hud, setHud]         = useState(null);
  const [showRaw, setShowRaw] = useState(false);
  const showRawRef = useRef(false);

  // Sync ref so the animation loop can read it without stale closure
  useEffect(() => { showRawRef.current = showRaw; }, [showRaw]);

  // ── Camera start ─────────────────────────────────────────
  const startCamera = useCallback(async () => {
    setStatus("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width:  { ideal: C.WIDTH },
          height: { ideal: C.HEIGHT },
          facingMode: "environment",
          // Request manual controls where supported
          advanced: [{ exposureMode: "manual", iso: 3200, exposureTime: 200000 }],
        },
        audio: false,
      });
      streamRef.current = stream;
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
    setStatus("idle");
    setHud(null);
  }, []);

  // ── Pipeline loop ────────────────────────────────────────
  const beginPipeline = useCallback(() => {
    const video   = videoRef.current;
    const srcCtx  = srcCanvasRef.current.getContext("2d", { willReadFrequently: true });
    const outCtx  = outCanvasRef.current.getContext("2d");
    const motion  = motionRef.current;
    const clahe   = claheRef.current;
    let stack     = stackRef.current;
    let frameCount = 0;
    let t0 = performance.now();

    function tick() {
      if (!video || video.readyState < 2) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      const W = C.WIDTH, H = C.HEIGHT;
      srcCtx.drawImage(video, 0, 0, W, H);
      const imgData = srcCtx.getImageData(0, 0, W, H);
      const raw     = extractLuma(imgData);
      const mean    = meanLuma(raw);

      // First frame of a new cycle → lock reference
      if (stack.count === 0) motion.lockReference(raw);

      // Motion-estimate and align
      const { dx, dy, magnitude } = motion.estimate(raw, W, H);
      const aligned = motion.align(raw, W, H, dx, dy);

      // Accumulate
      stack.accumulate(aligned, magnitude, mean);
      frameCount++;

      if (stack.allZonesDone()) {
        const t1 = performance.now();

        // Export + gain + CLAHE
        const averaged = stack.exportAndReset();
        const { gained, gain } = applyGain(averaged, mean);
        const equalised = clahe.equalise(gained, W, H);

        // Render
        if (!showRawRef.current) {
          const id = lumaToImageData(equalised, W, H, outCtx);
          outCtx.putImageData(id, 0, 0);
        } else {
          // Raw comparison mode
          outCtx.putImageData(imgData, 0, 0);
        }

        const depth = stack.targetDepth(mean);
        setHud({
          mean:  mean.toFixed(1),
          stack: depth,
          gain:  gain.toFixed(2),
          dx:    dx.toFixed(1),
          dy:    dy.toFixed(1),
          mag:   magnitude.toFixed(1),
          procMs: Math.round(t1 - t0),
          fps:   frameCount,
        });

        frameCount = 0;
        t0 = t1;
      }

      rafRef.current = requestAnimationFrame(tick);
    }

    rafRef.current = requestAnimationFrame(tick);
  }, []);

  useEffect(() => () => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
  }, []);

  // ── Render ───────────────────────────────────────────────
  return (
    <div style={styles.root}>
      {/* Hidden video + source canvas */}
      <video ref={videoRef} style={{ display: "none" }} playsInline muted />
      <canvas ref={srcCanvasRef} width={C.WIDTH} height={C.HEIGHT} style={{ display: "none" }} />

      {/* Header */}
      <header style={styles.header}>
        <span style={styles.headerTitle}>▌ DARK VISION</span>
        <span style={styles.headerSub}>THE BIG THREE</span>
      </header>

      {/* Viewport */}
      <div style={styles.viewport}>
        <canvas
          ref={outCanvasRef}
          width={C.WIDTH}
          height={C.HEIGHT}
          style={styles.outputCanvas}
        />

        {status === "idle" && (
          <div style={styles.overlay}>
            <div style={styles.splashBox}>
              <div style={styles.splashTitle}>DARK VISION</div>
              <div style={styles.splashSubtitle}>THE BIG THREE</div>
              <div style={styles.splashDesc}>
                Manual exposure · Multi-frame stacking<br />
                CLAHE adaptive contrast · Motion alignment
              </div>
              <button style={styles.btn} onClick={startCamera}>
                ▶ START CAMERA
              </button>
            </div>
          </div>
        )}

        {status === "starting" && (
          <div style={styles.overlay}>
            <div style={styles.spinnerLabel}>INITIALISING PIPELINE…</div>
          </div>
        )}

        {status === "error" && (
          <div style={styles.overlay}>
            <div style={styles.errorBox}>
              <div style={styles.errorTitle}>CAMERA UNAVAILABLE</div>
              <div style={styles.errorDesc}>
                Check browser permissions, then try again.
              </div>
              <button style={styles.btn} onClick={startCamera}>↺ RETRY</button>
            </div>
          </div>
        )}

        {/* HUD */}
        {hud && status === "live" && (
          <div style={styles.hud}>
            <div style={styles.hudTitle}>▌ DARK VISION  BIG THREE</div>
            <HudRow label="BRIGHTNESS" value={`${hud.mean}  luma`} dim />
            <HudRow label="STACK"      value={`${hud.stack}  frames  ×${hud.gain}`} />
            <HudRow label="MOTION"     value={`dx ${hud.dx}  dy ${hud.dy}  mag ${hud.mag}`} dim />
            <HudRow label="PROC"       value={`${hud.procMs} ms  ${hud.fps} f`} dim />
          </div>
        )}

        {/* Raw toggle */}
        {status === "live" && (
          <button
            style={{ ...styles.rawBtn, background: showRaw ? "#39ff14" : "rgba(0,0,0,0.55)", color: showRaw ? "#000" : "#39ff14" }}
            onClick={() => setShowRaw(v => !v)}
          >
            {showRaw ? "RAW  ON" : "RAW OFF"}
          </button>
        )}
      </div>

      {/* Controls */}
      {status === "live" && (
        <div style={styles.controls}>
          <button style={styles.stopBtn} onClick={stopCamera}>■ STOP</button>
        </div>
      )}

      {/* Legend */}
      <footer style={styles.footer}>
        <Legend num="1" title="Manual Exposure" desc="Simulates max ISO + long shutter — no auto-exposure." />
        <Legend num="2" title="Frame Stacking" desc="Motion-aligned accumulation. σ-clip removes transients." />
        <Legend num="3" title="CLAHE" desc="Adaptive per-tile contrast. Lifts dark detail without blowing highlights." />
      </footer>
    </div>
  );
}

function HudRow({ label, value, dim }) {
  return (
    <div style={{ display: "flex", gap: 8, marginBottom: 2 }}>
      <span style={{ ...styles.hudLabel, opacity: dim ? 0.6 : 1 }}>{label}</span>
      <span style={{ ...styles.hudValue, opacity: dim ? 0.7 : 1 }}>{value}</span>
    </div>
  );
}

function Legend({ num, title, desc }) {
  return (
    <div style={styles.legendItem}>
      <div style={styles.legendNum}>#{num}</div>
      <div>
        <div style={styles.legendTitle}>{title}</div>
        <div style={styles.legendDesc}>{desc}</div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// §07  STYLES  (phosphor-green on black — night-vision aesthetic)
// ══════════════════════════════════════════════════════════════
const GREEN  = "#39ff14";
const DIM    = "#1a7a09";
const BG     = "#020702";
const PANEL  = "rgba(2,12,2,0.88)";
const MONO   = "'Courier New', Courier, monospace";

const styles = {
  root: {
    background: BG,
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    fontFamily: MONO,
    color: GREEN,
    padding: "12px 8px 24px",
    gap: 0,
  },
  header: {
    width: "100%",
    maxWidth: 680,
    display: "flex",
    alignItems: "baseline",
    gap: 12,
    paddingBottom: 8,
    borderBottom: `1px solid ${DIM}`,
    marginBottom: 10,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: "bold",
    letterSpacing: 3,
    color: GREEN,
  },
  headerSub: {
    fontSize: 11,
    color: DIM,
    letterSpacing: 4,
  },
  viewport: {
    position: "relative",
    width: "100%",
    maxWidth: 680,
    aspectRatio: "4/3",
    background: "#000",
    border: `1px solid ${DIM}`,
    overflow: "hidden",
  },
  outputCanvas: {
    width: "100%",
    height: "100%",
    display: "block",
    imageRendering: "pixelated",
  },
  overlay: {
    position: "absolute", inset: 0,
    display: "flex", alignItems: "center", justifyContent: "center",
    background: "rgba(0,0,0,0.85)",
  },
  splashBox: {
    textAlign: "center",
    display: "flex", flexDirection: "column", alignItems: "center", gap: 10,
  },
  splashTitle: {
    fontSize: 32,
    fontWeight: "bold",
    letterSpacing: 6,
    color: GREEN,
    textShadow: `0 0 20px ${GREEN}`,
  },
  splashSubtitle: {
    fontSize: 11,
    letterSpacing: 6,
    color: DIM,
    marginTop: -6,
  },
  splashDesc: {
    fontSize: 12,
    color: DIM,
    lineHeight: 1.7,
    marginTop: 4,
  },
  spinnerLabel: {
    fontSize: 13,
    letterSpacing: 3,
    color: DIM,
    animation: "pulse 1.2s ease-in-out infinite alternate",
  },
  errorBox: {
    textAlign: "center",
    display: "flex", flexDirection: "column", alignItems: "center", gap: 10,
  },
  errorTitle: { fontSize: 16, letterSpacing: 3, color: "#ff4444" },
  errorDesc:  { fontSize: 12, color: "#884444" },
  btn: {
    marginTop: 8,
    background: "transparent",
    border: `1px solid ${GREEN}`,
    color: GREEN,
    fontFamily: MONO,
    fontSize: 13,
    letterSpacing: 2,
    padding: "8px 20px",
    cursor: "pointer",
  },
  stopBtn: {
    background: "transparent",
    border: "1px solid #ff4444",
    color: "#ff4444",
    fontFamily: MONO,
    fontSize: 12,
    letterSpacing: 2,
    padding: "6px 18px",
    cursor: "pointer",
    marginTop: 8,
  },
  hud: {
    position: "absolute",
    top: 10, left: 10,
    background: PANEL,
    border: `1px solid ${DIM}`,
    padding: "10px 14px",
    fontSize: 11,
    lineHeight: 1.6,
    minWidth: 260,
    backdropFilter: "blur(4px)",
  },
  hudTitle: {
    fontSize: 10,
    letterSpacing: 2,
    color: GREEN,
    marginBottom: 5,
    fontWeight: "bold",
  },
  hudLabel: {
    color: DIM,
    minWidth: 76,
    display: "inline-block",
    fontSize: 10,
    letterSpacing: 1,
  },
  hudValue: {
    color: GREEN,
    fontSize: 11,
  },
  rawBtn: {
    position: "absolute",
    bottom: 10, right: 10,
    border: `1px solid ${GREEN}`,
    fontFamily: MONO,
    fontSize: 10,
    letterSpacing: 2,
    padding: "4px 10px",
    cursor: "pointer",
    transition: "background 0.15s, color 0.15s",
  },
  controls: {
    width: "100%",
    maxWidth: 680,
    display: "flex",
    justifyContent: "flex-end",
  },
  footer: {
    width: "100%",
    maxWidth: 680,
    borderTop: `1px solid ${DIM}`,
    marginTop: 18,
    paddingTop: 14,
    display: "flex",
    flexDirection: "column",
    gap: 10,
  },
  legendItem: {
    display: "flex",
    gap: 14,
    alignItems: "flex-start",
  },
  legendNum: {
    color: GREEN,
    fontSize: 13,
    fontWeight: "bold",
    minWidth: 20,
    paddingTop: 1,
  },
  legendTitle: {
    color: GREEN,
    fontSize: 11,
    letterSpacing: 1,
    marginBottom: 1,
  },
  legendDesc: {
    color: DIM,
    fontSize: 10,
    lineHeight: 1.5,
  },
};
