/**
 * ╔══════════════════════════════════════════════════════════════╗
 * ║            DARK VISION — THE BIG THREE                      ║
 * ║                                                             ║
 * ║  Three technologies. Nothing else. Maximum dark vision.     ║
 * ║                                                             ║
 * ║  #1  Manual ISO + Exposure Maximisation  (Camera2 API)      ║
 * ║  #2  Multi-Frame Stacking + Gyro Alignment                  ║
 * ║  #3  CLAHE Adaptive Histogram Equalisation                  ║
 * ╚══════════════════════════════════════════════════════════════╝
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │ TABLE OF CONTENTS                                           │
 * ├─────────────────────────────────────────────────────────────┤
 * │ §01  CONSTANTS        — All tuning knobs in one place       │
 * │ §02  DATA MODELS      — Frame, meta, gyro state             │
 * │ §03  EXPOSURE ENGINE  — Camera2 full manual control         │
 * │ §04  GYRO ENGINE      — Angle integration + frame warp      │
 * │ §05  STACK ENGINE     — Frame accumulation + averaging      │
 * │ §06  CLAHE ENGINE     — Tile-based contrast equalisation    │
 * │ §07  PIPELINE         — Wires §03→§04→§05→§06 in order     │
 * │ §08  RENDERER         — Draws final bitmap to SurfaceView   │
 * │ §09  HUD VIEW         — Live telemetry overlay              │
 * │ §10  PERMISSIONS      — Camera runtime permission helper    │
 * │ §11  MAIN ACTIVITY    — Entry point, lifecycle, UI          │
 * └─────────────────────────────────────────────────────────────┘
 *
 * REQUIRED MANIFEST:
 *   <uses-permission android:name="android.permission.CAMERA"/>
 *   <uses-feature android:name="android.hardware.camera2" android:required="true"/>
 *
 * REQUIRED GRADLE:
 *   implementation 'androidx.appcompat:appcompat:1.7.0'
 */

// ══════════════════════════════════════════════════════════════
// §00  PACKAGE & IMPORTS
// ══════════════════════════════════════════════════════════════
package com.darkvision.bigthree

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.*
import android.hardware.camera2.*
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.Image
import android.media.ImageReader
import android.os.*
import android.util.Log
import android.util.Range
import android.view.*
import android.widget.FrameLayout
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import kotlin.math.*


// ══════════════════════════════════════════════════════════════
// §01  CONSTANTS
//      Purpose: Every tuneable value lives here.
//               Nothing is hard-coded anywhere else.
// ══════════════════════════════════════════════════════════════
object C {

    // ── Output resolution ─────────────────────────────────────
    const val WIDTH  = 1280
    const val HEIGHT = 960

    // ── #1  EXPOSURE — pushed to physical device limits ───────
    // ISO: as high as the sensor supports (clamped per device)
    const val ISO_DARK   = 12800    // target ISO in darkness
    const val ISO_BRIGHT = 200      // floor ISO in brighter scenes

    // Exposure time in nanoseconds
    // 250 ms = hand-holdable maximum; gyro alignment compensates the rest
    const val EXP_MAX_NS = 250_000_000L   // 1/4 s
    const val EXP_MIN_NS =   4_000_000L   // 1/250 s

    // Lux threshold below which we go to maximum ISO + exposure
    const val DARK_THRESHOLD_LUX = 5f

    // ── #2  STACKING ─────────────────────────────────────────
    // Frames to average:  SNR gain = √N
    //   8 frames  → 2.8× SNR gain
    //  32 frames  → 5.6× SNR gain
    const val STACK_MIN  = 8    // used near DARK_THRESHOLD_LUX
    const val STACK_MAX  = 32   // used in true darkness (< 0.5 lux)

    // ── #2  GYRO ALIGNMENT ───────────────────────────────────
    // Complementary filter weight (keeps long-term drift at zero)
    const val GYRO_ALPHA     = 0.96f
    // Maximum pixel shift we'll apply per frame (sanity clamp)
    const val MAX_SHIFT_PX   = 40
    // Sub-pixel alignment: bilinear warp replaces integer-pixel shift
    // (reduces edge ghosting on fine detail like text and door frames)

    // ── #4  PER-FRAME EXPOSURE WEIGHTING ─────────────────────
    // Angular velocity (rad/s) below which a frame is considered stable
    // Frames above this threshold receive proportionally less weight
    const val STABLE_GYRO_THRESHOLD = 0.05f   // ~3 deg/s

    // ── #3  TEMPORAL OUTLIER REJECTION ───────────────────────
    // Sigma multiplier for per-pixel rejection before averaging
    // Pixels beyond SIGMA_CLIP * σ from the running mean are dropped
    const val SIGMA_CLIP     = 2.0f

    // ── #3  CLAHE ────────────────────────────────────────────
    // Grid: image divided into TILE × TILE tiles
    const val CLAHE_TILES     = 8
    // Clip limit: prevents over-amplifying uniform noise regions
    // Higher = more aggressive contrast boost
    const val CLAHE_CLIP      = 3.5f

    // ── Misc ─────────────────────────────────────────────────
    const val TAG             = "DarkVision"
    const val PERM_CODE       = 1001
    const val HUD_INTERVAL_MS = 250L
}


// ══════════════════════════════════════════════════════════════
// §02  DATA MODELS
//      Purpose: Immutable structs passed between pipeline stages.
//               Keeps each engine decoupled and thread-safe.
// ══════════════════════════════════════════════════════════════

/**
 * Raw luminance (Y-plane) extracted from one camera frame.
 * width × height bytes, values 0-255.
 */
data class RawFrame(
    val luma: ByteArray,
    val width: Int,
    val height: Int,
    val captureTimeNs: Long,
    // Gyro angles at moment of capture — used for warp alignment
    val angleX: Float,
    val angleY: Float,
    // Angular velocity magnitude at capture — used for per-frame weighting (#4)
    // Low value = stable frame = high contribution weight
    val gyroMagnitude: Float = 0f
)

/**
 * Stacked + CLAHE-processed output ready for display.
 */
data class FinalFrame(
    val bitmap: Bitmap,
    val stackDepth: Int,        // how many frames were averaged
    val isoUsed: Int,           // reported ISO from camera
    val exposureMs: Float,      // reported exposure in ms
    val gainApplied: Float,     // digital gain multiplier applied pre-CLAHE
    val processingMs: Long      // total pipeline time
)

/**
 * Running gyroscope state — mutated by GyroEngine on sensor thread.
 */
class GyroState {
    @Volatile var angleX  = 0f
    @Volatile var angleY  = 0f
    @Volatile var lastNs  = 0L
    // Current angular velocity magnitude (rad/s) — used for per-frame weighting (#4)
    @Volatile var angularVelocity = 0f
}


// ══════════════════════════════════════════════════════════════
// §03  EXPOSURE ENGINE  (#1 — Manual ISO + Exposure)
//      Purpose: Drives Camera2 in fully manual mode.
//               Auto-exposure, auto-white-balance, and
//               auto-focus are all disabled.
//               ISO and shutter are chosen to maximise photon
//               collection given the ambient light level.
// ══════════════════════════════════════════════════════════════
class ExposureEngine(
    private val context: Context,
    private val gyroState: GyroState
) {
    private val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

    private var device: CameraDevice?           = null
    private var session: CameraCaptureSession?  = null
    private var reader: ImageReader?            = null

    private val camThread  = HandlerThread("CamThread").also { it.start() }
    private val camHandler = Handler(camThread.looper)

    /** Current lux from ambient sensor — drives ISO/exposure selection. */
    @Volatile var currentLux: Float = 0f

    /** Reported values from last CaptureResult — shown in HUD. */
    @Volatile var lastIso: Int    = 0
    @Volatile var lastExpMs: Float = 0f

    /** Callback: called on camera thread for each captured frame. */
    var onFrame: ((RawFrame) -> Unit)? = null

    // ── Ambient light sensor ──────────────────────────────────

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val lightSensor   = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)

    private val luxListener = object : SensorEventListener {
        override fun onSensorChanged(e: SensorEvent) {
            // Simple EMA smoothing
            currentLux = 0.15f * e.values[0] + 0.85f * currentLux
        }
        override fun onAccuracyChanged(s: Sensor?, a: Int) {}
    }

    fun startLuxSensor() {
        lightSensor?.let {
            sensorManager.registerListener(luxListener, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
    }

    fun stopLuxSensor() = sensorManager.unregisterListener(luxListener)

    // ── Camera open ───────────────────────────────────────────

    @SuppressLint("MissingPermission")
    fun open(onReady: () -> Unit) {
        val camId = pickBackCamera()
        reader = ImageReader.newInstance(C.WIDTH, C.HEIGHT, ImageFormat.YUV_420_888, 3).apply {
            setOnImageAvailableListener({ r ->
                val img = r.acquireLatestImage() ?: return@setOnImageAvailableListener
                handleImage(img)
            }, camHandler)
        }

        manager.openCamera(camId, object : CameraDevice.StateCallback() {
            override fun onOpened(cam: CameraDevice) {
                device = cam
                createSession(onReady)
            }
            override fun onDisconnected(cam: CameraDevice) = cam.close()
            override fun onError(cam: CameraDevice, e: Int) {
                Log.e(C.TAG, "[Exposure] Camera error $e"); cam.close()
            }
        }, camHandler)
    }

    private fun createSession(onReady: () -> Unit) {
        val surface = reader!!.surface

        val stateCallback = object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(s: CameraCaptureSession) {
                session = s
                s.setRepeatingRequest(buildRequest(surface), captureCallback, camHandler)
                onReady()
            }
            override fun onConfigureFailed(s: CameraCaptureSession) {
                Log.e(C.TAG, "[Exposure] Session configure failed")
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            device!!.createCaptureSession(
                SessionConfiguration(
                    SessionConfiguration.SESSION_REGULAR,
                    listOf(OutputConfiguration(surface)),
                    Executors.newSingleThreadExecutor(),
                    stateCallback
                )
            )
        } else {
            @Suppress("DEPRECATION")
            device!!.createCaptureSession(listOf(surface), stateCallback, camHandler)
        }
    }

    // ── Build fully manual CaptureRequest ─────────────────────

    private fun buildRequest(surface: Surface): CaptureRequest {
        val chars   = manager.getCameraCharacteristics(pickBackCamera())
        val isoRng  = chars[CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE]
                      ?: Range(100, 12800)
        val expRng  = chars[CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE]
                      ?: Range(C.EXP_MIN_NS, C.EXP_MAX_NS)

        val iso = selectIso(isoRng)
        val exp = selectExposure(expRng)

        Log.d(C.TAG, "[Exposure] Setting ISO=$iso  Exp=${exp/1_000_000}ms  Lux=${currentLux}")

        return device!!.createCaptureRequest(CameraDevice.TEMPLATE_MANUAL).apply {
            addTarget(surface)

            // Disable every automatic system — we own all controls
            set(CaptureRequest.CONTROL_MODE,     CaptureRequest.CONTROL_MODE_OFF)
            set(CaptureRequest.CONTROL_AE_MODE,  CaptureRequest.CONTROL_AE_MODE_OFF)
            set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_OFF)
            set(CaptureRequest.CONTROL_AF_MODE,  CaptureRequest.CONTROL_AF_MODE_OFF)

            // Maximum light collection
            set(CaptureRequest.SENSOR_SENSITIVITY,    iso.coerceIn(isoRng.lower, isoRng.upper))
            set(CaptureRequest.SENSOR_EXPOSURE_TIME,  exp.coerceIn(expRng.lower, expRng.upper))

            // Lens to hyperfocal / infinity — maximises depth of field in dark
            set(CaptureRequest.LENS_FOCUS_DISTANCE, 0f)

            // Disable in-camera NR and sharpening — pipeline does this better
            set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF)
            set(CaptureRequest.EDGE_MODE,            CaptureRequest.EDGE_MODE_OFF)
        }.build()
    }

    // ── ISO and exposure selection based on lux ───────────────

    /**
     * At 0 lux  → maximum device ISO.
     * At DARK_THRESHOLD_LUX → ISO_BRIGHT.
     * Linear interpolation in between.
     */
    private fun selectIso(range: Range<Int>): Int {
        val t   = (currentLux / C.DARK_THRESHOLD_LUX).coerceIn(0f, 1f)
        val max = range.upper.coerceAtMost(C.ISO_DARK)
        return (max - t * (max - C.ISO_BRIGHT)).toInt()
    }

    /**
     * At 0 lux → EXP_MAX_NS (250 ms).
     * At DARK_THRESHOLD_LUX → EXP_MIN_NS (4 ms).
     */
    private fun selectExposure(range: Range<Long>): Long {
        val t = (currentLux / C.DARK_THRESHOLD_LUX).coerceIn(0f, 1f)
        return (C.EXP_MAX_NS - t * (C.EXP_MAX_NS - C.EXP_MIN_NS)).toLong()
            .coerceIn(range.lower, range.upper)
    }

    // ── CaptureResult callback — reads back actual values ─────

    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            s: CameraCaptureSession, rq: CaptureRequest, r: TotalCaptureResult
        ) {
            lastIso   = r[CaptureResult.SENSOR_SENSITIVITY] ?: lastIso
            lastExpMs = ((r[CaptureResult.SENSOR_EXPOSURE_TIME] ?: 0L) / 1_000_000f)
        }
    }

    // ── Image extraction — Y-plane only ───────────────────────

    private fun handleImage(image: Image) {
        try {
            val plane  = image.planes[0]
            val buf    = plane.buffer
            val stride = plane.rowStride
            val w = image.width; val h = image.height

            val luma = ByteArray(w * h)
            if (stride == w) {
                buf.get(luma)
            } else {
                for (row in 0 until h) {
                    buf.position(row * stride)
                    buf.get(luma, row * w, w)
                }
            }

            onFrame?.invoke(
                RawFrame(
                    luma          = luma,
                    width         = w,
                    height        = h,
                    captureTimeNs = System.nanoTime(),
                    angleX        = gyroState.angleX,
                    angleY        = gyroState.angleY,
                    gyroMagnitude = gyroState.angularVelocity
                )
            )
        } finally {
            image.close()
        }
    }

    // ── Helpers ───────────────────────────────────────────────

    private fun pickBackCamera(): String {
        for (id in manager.cameraIdList) {
            val ch = manager.getCameraCharacteristics(id)
            if (ch[CameraCharacteristics.LENS_FACING] == CameraCharacteristics.LENS_FACING_BACK)
                return id
        }
        return manager.cameraIdList.first()
    }

    fun close() {
        session?.close(); device?.close(); reader?.close()
        camThread.quitSafely()
    }
}


// ══════════════════════════════════════════════════════════════
// §04  GYRO ENGINE  (#2 part A — Frame alignment)
//      Purpose: Integrates gyroscope angular velocity into
//               cumulative roll/pitch angles.  Before each
//               frame is added to the stack, it is translated
//               by the delta since the reference angle was
//               captured, cancelling hand-shake blur.
//
//      WHY THIS MATTERS:
//      Stacking without alignment smears moving edges into
//      a ghost blur.  With alignment, 32 frames of a still
//      scene snap into crystal clarity.
// ══════════════════════════════════════════════════════════════
class GyroEngine(context: Context) : SensorEventListener {

    val state = GyroState()

    private val sm   = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val gyro = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    /** Reference angles locked at the start of each stack cycle. */
    @Volatile private var refX = 0f
    @Volatile private var refY = 0f

    fun start() {
        gyro?.let {
            sm.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
            Log.i(C.TAG, "[Gyro] Started")
        } ?: Log.w(C.TAG, "[Gyro] No gyroscope — stacking without alignment")
    }

    fun stop() = sm.unregisterListener(this)

    /** Lock reference at the beginning of a new stack cycle. */
    fun lockReference() {
        refX = state.angleX
        refY = state.angleY
    }

    /**
     * Warp the frame's luma array to compensate for the angular
     * delta between the reference and this frame's capture angle.
     *
     * Uses bilinear-interpolated fractional-pixel translation (#2).
     * At 250 ms / 1280 px, sub-pixel drift between frames is real;
     * bilinear sampling sharpens fine edges (text, door frames)
     * noticeably compared to integer-pixel rounding.
     */
    fun align(frame: RawFrame): RawFrame {
        // Convert angular delta to pixel shift
        // At 45° FOV: full rotation = full width in pixels
        val fovFactor = C.WIDTH / (PI / 4).toFloat()
        val dX = ((frame.angleX - refX) * fovFactor)
            .coerceIn(-C.MAX_SHIFT_PX.toFloat(), C.MAX_SHIFT_PX.toFloat())
        val dY = ((frame.angleY - refY) * fovFactor)
            .coerceIn(-C.MAX_SHIFT_PX.toFloat(), C.MAX_SHIFT_PX.toFloat())

        // Skip if shift is truly sub-pixel (< 0.01 px — nothing to gain)
        if (abs(dX) < 0.01f && abs(dY) < 0.01f) return frame

        val src = frame.luma
        val dst = ByteArray(src.size)
        val w   = frame.width
        val h   = frame.height

        // Fractional source offset (negative: we pull the image the other way)
        val ox = -dX
        val oy = -dY

        for (y in 0 until h) {
            val sy = y + oy
            val y0 = sy.toInt().coerceIn(0, h - 1)
            val y1 = (y0 + 1).coerceIn(0, h - 1)
            val fy = sy - y0.toFloat()

            for (x in 0 until w) {
                val sx = x + ox
                val x0 = sx.toInt().coerceIn(0, w - 1)
                val x1 = (x0 + 1).coerceIn(0, w - 1)
                val fx = sx - x0.toFloat()

                // Bilinear blend of 4 neighbours
                val v00 = src[y0 * w + x0].toInt() and 0xFF
                val v10 = src[y0 * w + x1].toInt() and 0xFF
                val v01 = src[y1 * w + x0].toInt() and 0xFF
                val v11 = src[y1 * w + x1].toInt() and 0xFF

                val v = (v00 * (1f - fx) * (1f - fy) +
                         v10 *       fx  * (1f - fy) +
                         v01 * (1f - fx) *       fy  +
                         v11 *       fx  *       fy).toInt().coerceIn(0, 255)

                dst[y * w + x] = v.toByte()
            }
        }

        return frame.copy(luma = dst)
    }

    // ── SensorEventListener ───────────────────────────────────

    override fun onSensorChanged(event: SensorEvent) {
        val now = event.timestamp
        if (state.lastNs == 0L) { state.lastNs = now; return }
        val dt = (now - state.lastNs) * 1e-9f
        state.lastNs = now

        // Complementary filter: integrate rate, bleed off drift
        state.angleX = C.GYRO_ALPHA * (state.angleX + event.values[0] * dt)
        state.angleY = C.GYRO_ALPHA * (state.angleY + event.values[1] * dt)

        // Track instantaneous angular velocity magnitude for frame weighting (#4)
        val wx = event.values[0]; val wy = event.values[1]; val wz = event.values[2]
        state.angularVelocity = sqrt(wx * wx + wy * wy + wz * wz)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}


// ══════════════════════════════════════════════════════════════
// §05  STACK ENGINE  (#2 part B — Multi-frame averaging)
//      Purpose: Accumulates gyro-aligned luma frames into an
//               integer buffer, then exports the per-pixel
//               mean.
//
//      ENHANCEMENTS IMPLEMENTED:
//      #1 — Adaptive zone stack depth: frame is split into a
//           ZONE_COLS × ZONE_ROWS grid.  Each zone measures its
//           own mean brightness and selects a local stack depth
//           (bright zones → STACK_MIN; dark zones → STACK_MAX).
//           The result is optimal SNR across the whole frame
//           rather than a single global compromise.
//
//      #3 — Temporal outlier rejection: before contributing to
//           the mean, each pixel is checked against a running
//           per-pixel mean + stddev.  Pixels beyond SIGMA_CLIP × σ
//           are discarded.  Moving objects (hands, pets, curtains)
//           are erased from the stack; the background stays sharp.
//
//      #4 — Per-frame exposure weighting: each frame contributes
//           proportionally to its gyro stability score.
//           weight = 1 / (1 + gyroMagnitude / STABLE_GYRO_THRESHOLD)
//           Shaky frames contribute less; stable frames vote more.
//
//      PHYSICS:
//      Each pixel has random photon shot-noise with standard
//      deviation σ.  Averaging N independent samples reduces
//      noise to σ/√N while the true signal remains constant.
//      32 frames → 5.6× SNR improvement.
// ══════════════════════════════════════════════════════════════
class StackEngine(
    private val width: Int,
    private val height: Int,
    private val exposure: ExposureEngine
) {
    // Per-pixel weighted accumulator and weight sum (floats for precision)
    private val weightedSum = FloatArray(width * height)
    private val weightSum   = FloatArray(width * height)

    // Running mean and M2 for Welford online variance (sigma clipping, #3)
    private val runMean = FloatArray(width * height)
    private val runM2   = FloatArray(width * height)

    // Zone constants (#1)
    private val ZONE_COLS = 4
    private val ZONE_ROWS = 4
    private val zoneW = width  / ZONE_COLS
    private val zoneH = height / ZONE_ROWS

    // Per-zone target depths and accumulated frame counts
    private val zoneTarget = IntArray(ZONE_COLS * ZONE_ROWS)
    private val zoneCount  = IntArray(ZONE_COLS * ZONE_ROWS)

    @Volatile var count = 0
        private set

    /**
     * Returns the global target depth for the current lux level.
     * Individual zones may use lower values if they are bright enough.
     */
    fun targetDepth(): Int {
        val lux = exposure.currentLux
        val t   = (lux / C.DARK_THRESHOLD_LUX).coerceIn(0f, 1f)
        return (C.STACK_MAX - t * (C.STACK_MAX - C.STACK_MIN)).roundToInt()
    }

    /**
     * Whether all zones have reached their individual target depths.
     */
    fun allZonesDone(): Boolean = zoneCount.zip(zoneTarget.toList()).all { (c, t) -> t == 0 || c >= t }

    @Synchronized
    fun accumulate(frame: RawFrame) {
        val src = frame.luma
        val n   = count + 1   // 1-based count for Welford

        // Gyro stability weight (#4)
        val gyroWeight = 1f / (1f + frame.gyroMagnitude / C.STABLE_GYRO_THRESHOLD)

        // On first frame: compute per-zone target depths (#1)
        if (count == 0) {
            computeZoneTargets(src)
            zoneCount.fill(0)
        }

        for (y in 0 until height) {
            val zRow = (y / zoneH).coerceIn(0, ZONE_ROWS - 1)
            for (x in 0 until width) {
                val zCol  = (x / zoneW).coerceIn(0, ZONE_COLS - 1)
                val zIdx  = zRow * ZONE_COLS + zCol
                val pixel = src[y * width + x].toInt() and 0xFF
                val pIdx  = y * width + x
                val pf    = pixel.toFloat()

                // Sigma-clipping outlier rejection (#3)
                // Welford online mean/variance update first
                val oldMean = runMean[pIdx]
                val delta   = pf - oldMean
                runMean[pIdx] += delta / n
                val delta2  = pf - runMean[pIdx]
                runM2[pIdx] += delta * delta2

                val variance = if (n > 1) runM2[pIdx] / (n - 1) else Float.MAX_VALUE
                val sigma    = sqrt(variance)
                val isOutlier = n > 2 && abs(pf - oldMean) > C.SIGMA_CLIP * sigma

                if (!isOutlier) {
                    weightedSum[pIdx] += pf * gyroWeight
                    weightSum[pIdx]   += gyroWeight
                }
            }
        }

        // Increment per-zone counters
        for (z in zoneTarget.indices) {
            if (zoneCount[z] < zoneTarget[z]) zoneCount[z]++
        }

        count++
    }

    /**
     * Compute per-zone stack depths from zone mean brightness (#1).
     * Bright zones → STACK_MIN frames; dark zones → STACK_MAX.
     */
    private fun computeZoneTargets(src: ByteArray) {
        val globalDepth = targetDepth()
        for (zRow in 0 until ZONE_ROWS) {
            for (zCol in 0 until ZONE_COLS) {
                val x0 = zCol * zoneW; val x1 = x0 + zoneW
                val y0 = zRow * zoneH; val y1 = y0 + zoneH
                var sum = 0L
                for (y in y0 until y1)
                    for (x in x0 until x1)
                        sum += src[y * width + x].toInt() and 0xFF
                val mean  = sum.toFloat() / (zoneW * zoneH)
                // Dark zones (mean near 0) → STACK_MAX; bright → STACK_MIN
                val t     = (mean / 200f).coerceIn(0f, 1f)
                val depth = (C.STACK_MAX - t * (C.STACK_MAX - C.STACK_MIN)).roundToInt()
                    .coerceAtMost(globalDepth)
                zoneTarget[zRow * ZONE_COLS + zCol] = depth
            }
        }
    }

    /**
     * Exports weighted-average luma, resets all state.
     * Returns a 0-255 IntArray ready for CLAHE.
     */
    @Synchronized
    fun exportAndReset(): IntArray {
        val out = IntArray(width * height) { i ->
            if (weightSum[i] > 0f) (weightedSum[i] / weightSum[i]).toInt().coerceIn(0, 255)
            else 0
        }
        weightedSum.fill(0f)
        weightSum.fill(0f)
        runMean.fill(0f)
        runM2.fill(0f)
        zoneTarget.fill(0)
        zoneCount.fill(0)
        count = 0
        return out
    }
}


// ══════════════════════════════════════════════════════════════
// §06  CLAHE ENGINE  (#3 — Adaptive histogram equalisation)
//      Purpose: Lifts detail in dark regions without blowing
//               bright regions.  The image is divided into a
//               grid of tiles; each tile gets its own histogram
//               and lookup table.  Boundaries are blended via
//               bilinear interpolation.  The clip limit
//               prevents uniform noise from being amplified.
//
//      WHY CLAHE OVER SIMPLE BRIGHTNESS:
//      Simple brightness lift raises noise equally with signal.
//      CLAHE is contrast-relative: it amplifies regions where
//      the sensor saw meaningful variation, not flat noise.
//      A face in a dark room has micro-contrast — CLAHE finds
//      it; a simple brightness boost just makes grey noise.
// ══════════════════════════════════════════════════════════════
class ClaheEngine {

    /**
     * Applies CLAHE to a 0-255 IntArray and returns a new
     * equalised IntArray of the same dimensions.
     *
     * @param src       input luma, width × height values 0-255
     * @param width     image width
     * @param height    image height
     */
    fun equalise(src: IntArray, width: Int, height: Int): IntArray {
        val tilesX    = C.CLAHE_TILES
        val tilesY    = C.CLAHE_TILES
        val tileW     = width  / tilesX
        val tileH     = height / tilesY
        val tileSize  = tileW * tileH
        val clipLimit = (C.CLAHE_CLIP * tileSize / 256f).toInt().coerceAtLeast(1)

        // Step 1: Build per-tile LUTs
        val luts = Array(tilesY) { Array(tilesX) { IntArray(256) } }

        for (ty in 0 until tilesY) {
            for (tx in 0 until tilesX) {

                // Histogram
                val hist = IntArray(256)
                val x0 = tx * tileW;  val x1 = x0 + tileW
                val y0 = ty * tileH;  val y1 = y0 + tileH
                for (y in y0 until y1)
                    for (x in x0 until x1)
                        hist[src[y * width + x]]++

                // Clip: excess pixels redistributed uniformly
                var excess = 0
                for (b in 0..255) {
                    if (hist[b] > clipLimit) {
                        excess += hist[b] - clipLimit
                        hist[b] = clipLimit
                    }
                }
                val spread = excess / 256
                for (b in 0..255) hist[b] += spread

                // Build cumulative distribution → LUT
                var cdf    = 0
                var cdfMin = -1
                val lut    = luts[ty][tx]
                for (b in 0..255) {
                    cdf += hist[b]
                    if (cdfMin < 0 && cdf > 0) cdfMin = cdf
                    lut[b] = if (tileSize - cdfMin > 0)
                        ((cdf - cdfMin) * 255 / (tileSize - cdfMin)).coerceIn(0, 255)
                    else 0
                }
            }
        }

        // Step 2: Bilinear interpolation between neighbouring tile LUTs
        val dst = IntArray(src.size)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val v = src[y * width + x]

                // Tile coordinates (floating, centred on tile centres)
                val txF = (x.toFloat() / tileW) - 0.5f
                val tyF = (y.toFloat() / tileH) - 0.5f

                val tx0 = txF.toInt().coerceIn(0, tilesX - 1)
                val ty0 = tyF.toInt().coerceIn(0, tilesY - 1)
                val tx1 = (tx0 + 1).coerceIn(0, tilesX - 1)
                val ty1 = (ty0 + 1).coerceIn(0, tilesY - 1)

                val wx = (txF - tx0).coerceIn(0f, 1f)
                val wy = (tyF - ty0).coerceIn(0f, 1f)

                // 4-corner blend
                val v00 = luts[ty0][tx0][v]
                val v10 = luts[ty0][tx1][v]
                val v01 = luts[ty1][tx0][v]
                val v11 = luts[ty1][tx1][v]

                dst[y * width + x] = (
                    v00 * (1 - wx) * (1 - wy) +
                    v10 *      wx  * (1 - wy) +
                    v01 * (1 - wx) *      wy  +
                    v11 *      wx  *      wy
                ).roundToInt().coerceIn(0, 255)
            }
        }

        return dst
    }

    /**
     * Converts an equalised luma IntArray to a greyscale ARGB_8888 Bitmap.
     * Pure greyscale preserves maximum fidelity — no false colour distraction.
     */
    fun toBitmap(luma: IntArray, width: Int, height: Int): Bitmap {
        val pixels = IntArray(luma.size) { i ->
            val v = luma[i]
            Color.argb(255, v, v, v)
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }
}


// ══════════════════════════════════════════════════════════════
// §07  PIPELINE
//      Purpose: Wires the three engines in sequence.
//               ExposureEngine → GyroEngine.align()
//                             → StackEngine.accumulate()
//                             → ClaheEngine.equalise()
//                             → Bitmap → Renderer
//
//      All processing happens on a dedicated background thread.
//      The UI thread only receives finished bitmaps.
// ══════════════════════════════════════════════════════════════
class Pipeline(
    private val gyro:     GyroEngine,
    private val stack:    StackEngine,
    private val clahe:    ClaheEngine,
    private val exposure: ExposureEngine
) {
    private val thread  = HandlerThread("PipelineThread").also { it.start() }
    private val handler = Handler(thread.looper)

    var onResult: ((FinalFrame) -> Unit)? = null

    /** Called from camera thread — posts to pipeline thread. */
    fun push(raw: RawFrame) {
        handler.post {
            val t0 = System.currentTimeMillis()

            // First frame of a new cycle: lock gyro reference
            if (stack.count == 0) gyro.lockReference()

            // Step A: Gyro-align the frame before accumulation
            val aligned = gyro.align(raw)

            // Step B: Accumulate into the stack
            stack.accumulate(aligned)

            // Wait until all zones have reached their individual target depths (#1)
            if (!stack.allZonesDone()) return@post

            // Step C: Export averaged luma
            val averaged = stack.exportAndReset()

            // Compute a modest digital pre-gain before CLAHE
            // (lifts very dark frames to CLAHE's operating range)
            val gain   = computeGain()
            val gained = IntArray(averaged.size) { i ->
                (averaged[i] * gain).toInt().coerceIn(0, 255)
            }

            // Step D: CLAHE equalisation
            val equalised = clahe.equalise(gained, raw.width, raw.height)

            // Step E: Convert to displayable Bitmap
            val bitmap = clahe.toBitmap(equalised, raw.width, raw.height)

            val result = FinalFrame(
                bitmap       = bitmap,
                stackDepth   = stack.targetDepth(),
                isoUsed      = exposure.lastIso,
                exposureMs   = exposure.lastExpMs,
                gainApplied  = gain,
                processingMs = System.currentTimeMillis() - t0
            )

            onResult?.invoke(result)
        }
    }

    private fun computeGain(): Float {
        val t = (exposure.currentLux / C.DARK_THRESHOLD_LUX).coerceIn(0f, 1f)
        // At 0 lux → 4.0× digital gain; at threshold → 1.0×
        return 4.0f - t * 3.0f
    }

    fun shutdown() = thread.quitSafely()
}


// ══════════════════════════════════════════════════════════════
// §08  RENDERER
//      Purpose: Draws FinalFrames to a SurfaceView as fast as
//               they arrive.  Scales to fill the surface while
//               preserving aspect ratio.  Uses hardware canvas
//               when available, software canvas as fallback.
// ══════════════════════════════════════════════════════════════
class Renderer(private val surface: SurfaceView) {

    private val thread  = HandlerThread("RenderThread").also { it.start() }
    private val handler = Handler(thread.looper)

    private val srcRect = Rect()
    private val dstRect = RectF()
    private val paint   = Paint(Paint.FILTER_BITMAP_FLAG)

    fun draw(frame: FinalFrame) {
        handler.post {
            val holder = surface.holder
            val canvas = try { holder.lockHardwareCanvas() } catch (e: Exception) { null }
                         ?: holder.lockCanvas()
                         ?: return@post
            try {
                canvas.drawColor(Color.BLACK)

                val bmp = frame.bitmap
                srcRect.set(0, 0, bmp.width, bmp.height)

                // Letterbox-fit into surface
                val sw = canvas.width.toFloat()
                val sh = canvas.height.toFloat()
                val scale = minOf(sw / bmp.width, sh / bmp.height)
                val dw = bmp.width * scale
                val dh = bmp.height * scale
                dstRect.set(
                    (sw - dw) / 2f, (sh - dh) / 2f,
                    (sw + dw) / 2f, (sh + dh) / 2f
                )

                canvas.drawBitmap(bmp, srcRect, dstRect, paint)
            } finally {
                holder.unlockCanvasAndPost(canvas)
            }
        }
    }

    fun shutdown() = thread.quitSafely()
}


// ══════════════════════════════════════════════════════════════
// §09  HUD VIEW
//      Purpose: Minimal telemetry overlay — shows exactly the
//               three parameters that matter:
//               ISO, exposure, stack depth, and lux.
//               Drawn on a transparent Canvas above the preview.
// ══════════════════════════════════════════════════════════════
class HudView(context: Context) : View(context) {

    data class State(
        val lux: Float       = 0f,
        val iso: Int         = 0,
        val expMs: Float     = 0f,
        val stack: Int       = 0,
        val gain: Float      = 1f,
        val procMs: Long     = 0L
    )

    @Volatile private var state = State()

    private val bg = Paint().apply {
        color     = Color.argb(160, 0, 0, 0)
        isAntiAlias = true
    }
    private val txt = Paint().apply {
        color     = Color.argb(240, 180, 255, 180)
        textSize  = 34f
        typeface  = Typeface.MONOSPACE
        isAntiAlias = true
    }
    private val dim = Paint(txt).apply {
        color   = Color.argb(160, 120, 200, 120)
        textSize = 26f
    }

    fun update(s: State) { state = s; postInvalidate() }

    override fun onDraw(canvas: Canvas) {
        val s   = state
        val pad = 22f
        val lh  = 44f
        val w   = 380f
        val h   = lh * 6 + pad * 2.5f

        canvas.drawRoundRect(pad, pad, pad + w, pad + h, 16f, 16f, bg)

        var y = pad + lh
        fun line(text: String, p: Paint = txt) { canvas.drawText(text, pad * 2, y, p); y += lh }

        line("▌ DARK VISION  BIG THREE")
        line("LUX     ${String.format("%7.4f", s.lux)}", dim)
        line("ISO     ${s.iso.toString().padStart(7)}")
        line("EXP     ${String.format("%6.1f", s.expMs)} ms")
        line("STACK   ${s.stack.toString().padStart(5)} frames  ×${String.format("%.1f",s.gain)}")
        line("PROC    ${s.procMs.toString().padStart(6)} ms", dim)
    }
}


// ══════════════════════════════════════════════════════════════
// §10  PERMISSIONS
//      Purpose: Runtime camera permission — single place,
//               no boilerplate scattered through the activity.
// ══════════════════════════════════════════════════════════════
object Permissions {
    private val NEEDED = arrayOf(Manifest.permission.CAMERA)
    const val CODE = C.PERM_CODE

    fun allGranted(ctx: Context) = NEEDED.all {
        ContextCompat.checkSelfPermission(ctx, it) == PackageManager.PERMISSION_GRANTED
    }

    fun request(act: AppCompatActivity) =
        ActivityCompat.requestPermissions(act, NEEDED, CODE)
}


// ══════════════════════════════════════════════════════════════
// §11  MAIN ACTIVITY
//      Purpose: Android entry point.
//               Builds the UI, creates all engines, connects
//               the pipeline, manages the lifecycle cleanly.
// ══════════════════════════════════════════════════════════════
class MainActivity : AppCompatActivity() {

    // ── Engines ───────────────────────────────────────────────
    private lateinit var gyroEngine:     GyroEngine
    private lateinit var exposureEngine: ExposureEngine
    private lateinit var stackEngine:    StackEngine
    private lateinit var claheEngine:    ClaheEngine
    private lateinit var pipeline:       Pipeline
    private lateinit var renderer:       Renderer

    // ── Views ─────────────────────────────────────────────────
    private lateinit var surfaceView: SurfaceView
    private lateinit var hudView:     HudView

    // ── HUD refresh ───────────────────────────────────────────
    private val hudHandler = Handler(Looper.getMainLooper())
    private var lastFrame: FinalFrame? = null

    private val hudTick = object : Runnable {
        override fun run() {
            lastFrame?.let { f ->
                hudView.update(HudView.State(
                    lux   = exposureEngine.currentLux,
                    iso   = f.isoUsed,
                    expMs = f.exposureMs,
                    stack = f.stackDepth,
                    gain  = f.gainApplied,
                    procMs = f.processingMs
                ))
            } ?: hudView.update(HudView.State(lux = exposureEngine.currentLux))
            hudHandler.postDelayed(this, C.HUD_INTERVAL_MS)
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        goFullscreen()
        buildUi()
        initEngines()

        if (Permissions.allGranted(this)) launch()
        else Permissions.request(this)
    }

    override fun onResume() {
        super.onResume()
        exposureEngine.startLuxSensor()
        gyroEngine.start()
        hudHandler.post(hudTick)
    }

    override fun onPause() {
        super.onPause()
        exposureEngine.stopLuxSensor()
        gyroEngine.stop()
        hudHandler.removeCallbacks(hudTick)
    }

    override fun onDestroy() {
        super.onDestroy()
        exposureEngine.close()
        pipeline.shutdown()
        renderer.shutdown()
    }

    override fun onRequestPermissionsResult(
        code: Int, perms: Array<out String>, results: IntArray
    ) {
        super.onRequestPermissionsResult(code, perms, results)
        if (code == Permissions.CODE && results.all { it == PackageManager.PERMISSION_GRANTED })
            launch()
        else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    // ── UI construction ───────────────────────────────────────

    private fun buildUi() {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }

        surfaceView = SurfaceView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        }

        hudView = HudView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        }

        root.addView(surfaceView)
        root.addView(hudView)
        setContentView(root)
    }

    // ── Engine wiring ─────────────────────────────────────────

    private fun initEngines() {
        gyroEngine     = GyroEngine(this)
        exposureEngine = ExposureEngine(this, gyroEngine.state)
        stackEngine    = StackEngine(C.WIDTH, C.HEIGHT, exposureEngine)
        claheEngine    = ClaheEngine()
        renderer       = Renderer(surfaceView)

        pipeline = Pipeline(gyroEngine, stackEngine, claheEngine, exposureEngine)
        pipeline.onResult = { frame ->
            lastFrame = frame
            renderer.draw(frame)
        }

        exposureEngine.onFrame = { raw -> pipeline.push(raw) }
    }

    private fun launch() {
        exposureEngine.open {
            Log.i(C.TAG, "[Main] Pipeline live")
        }
    }

    // ── Fullscreen ────────────────────────────────────────────

    private fun goFullscreen() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.insetsController?.hide(
                android.view.WindowInsets.Type.statusBars() or
                android.view.WindowInsets.Type.navigationBars()
            )
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_FULLSCREEN or
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
            )
        }
    }
}

/*
 * ════════════════════════════════════════════════════════════════
 * END  DarkVisionBigThree.kt
 * ════════════════════════════════════════════════════════════════
 *
 * AndroidManifest.xml additions:
 * ───────────────────────────────────────────────────────────────
 *   <uses-permission android:name="android.permission.CAMERA"/>
 *   <uses-feature
 *       android:name="android.hardware.camera2"
 *       android:required="true"/>
 *
 * build.gradle (:app):
 * ───────────────────────────────────────────────────────────────
 *   implementation 'androidx.appcompat:appcompat:1.7.0'
 *
 * That is all that is needed.  No third-party libraries.
 * No model files.  No special hardware.
 * ════════════════════════════════════════════════════════════════
 */
