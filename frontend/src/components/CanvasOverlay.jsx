import { useEffect, useRef } from 'react'

const VEHICLE_COLOR = '#45d3e0'
const NEAREST_FRAME_TOLERANCE_SEC = 2

// Same real Indian RTO plate-background colors as the plate chips elsewhere
// in the UI (FinalPlatesTable, DetectionTable) — keeps a detection's plate
// reading visually identifiable by category directly on the live feed, not
// just in the tables below it.
const PLATE_CATEGORY_COLORS = {
  private: { bg: '#f4f6f8', fg: '#14161a' },
  commercial: { bg: '#f2b705', fg: '#14161a' },
  ev: { bg: '#1f9d55', fg: '#f2fbf5' },
  government: { bg: '#1c5fd1', fg: '#f2f6fd' },
}
// A plate box with no category call yet (crop too small/washed-out) or no
// OCR text yet (still pending) — neutral, not one of the four real colors.
const PLATE_PENDING_COLOR = { bg: '#3a4450', fg: '#c7cdd2' }

export default function CanvasOverlay({ frames, videoRef, frameWidth, frameHeight }) {
  const canvasRef = useRef(null)
  // Read via ref inside the draw loop instead of restarting the effect on
  // every `frames` update — avoids tearing down/recreating the rAF loop
  // every time a new segment's detections arrive.
  const framesRef = useRef(frames)
  framesRef.current = frames
  // Same reasoning as framesRef: read via ref inside the draw loop so a
  // later-arriving frameWidth/frameHeight (populated only once the backend
  // has processed its first frame, after this effect already started)
  // doesn't require tearing down/recreating the rAF loop.
  const sourceSizeRef = useRef({ frameWidth, frameHeight })
  sourceSizeRef.current = { frameWidth, frameHeight }

  useEffect(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    let rafId
    const draw = () => {
      const rect = video.getBoundingClientRect()
      if (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height)) {
        canvas.width = Math.round(rect.width)
        canvas.height = Math.round(rect.height)
      }
      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const vidW = video.videoWidth
      const vidH = video.videoHeight
      const currentFrames = framesRef.current

      if (currentFrames?.length && vidW > 0) {
        let best = null
        let bestDiff = Infinity
        for (const f of currentFrames) {
          const d = Math.abs(f.video_timestamp_sec - video.currentTime)
          if (d < bestDiff) {
            bestDiff = d
            best = f
          }
        }

        if (best && bestDiff < NEAREST_FRAME_TOLERANCE_SEC) {
          const { frameWidth, frameHeight } = sourceSizeRef.current
          // Detection bboxes are in the raw source frame's pixel space,
          // which can differ from the decoded HLS preview's resolution
          // (downscaled for faster ffmpeg encoding) — fall back to the
          // decoded video's own size only until the real source size has
          // been fetched, so boxes aren't wildly misplaced meanwhile.
          drawFrame(ctx, canvas, vidW, vidH, frameWidth || vidW, frameHeight || vidH, best.detections ?? [])
        }
      }
      rafId = requestAnimationFrame(draw)
    }
    rafId = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(rafId)
  }, [videoRef])

  return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
}

function drawFrame(ctx, canvas, vidW, vidH, frameWidth, frameHeight, detections) {
  // object-contain letterbox/pillarbox offset — the decoded video may not
  // exactly fill the canvas (e.g. fullscreen with a different aspect ratio
  // than the source), so boxes must be positioned relative to the actual
  // rendered video area, not the raw canvas/container size. Based on the
  // *decoded* video's aspect ratio (vidW/vidH), which is what the browser
  // actually letterboxes against.
  const videoAspect = vidW / vidH
  const canvasAspect = canvas.width / canvas.height
  let renderW, renderH, offsetX, offsetY
  if (videoAspect > canvasAspect) {
    renderW = canvas.width
    renderH = canvas.width / videoAspect
    offsetX = 0
    offsetY = (canvas.height - renderH) / 2
  } else {
    renderH = canvas.height
    renderW = canvas.height * videoAspect
    offsetX = (canvas.width - renderW) / 2
    offsetY = 0
  }
  // Detection bboxes are in the raw source frame's pixel space (frameWidth/
  // frameHeight), which can be a different resolution than the decoded HLS
  // preview (vidW/vidH) — the preview is downscaled independently for
  // faster ffmpeg encoding. Scaling by the source size here, not vidW/vidH,
  // is what keeps boxes aligned with the video regardless of that mismatch.
  const sx = renderW / frameWidth
  const sy = renderH / frameHeight

  const corner = Math.max(8, canvas.width / 60)
  ctx.lineWidth = Math.max(2, canvas.width / 500)
  ctx.font = `600 ${Math.max(11, canvas.width / 85)}px ui-monospace, "SF Mono", Consolas, monospace`
  ctx.textBaseline = 'bottom'

  for (const d of detections) {
    // Every vehicle is tracked now (VehicleTracker assigns a track_id to
    // every vehicle box regardless of whether a plate was found inside it
    // this frame) — track_id is never null.
    if (d.vehicle_bbox) {
      const label = `${d.vehicle_type} #${d.track_id}${
        d.vehicle_confidence != null ? ` · ${Math.round(d.vehicle_confidence * 100)}%` : ''
      }`
      drawReticle(ctx, d.vehicle_bbox, VEHICLE_COLOR, label, offsetX, offsetY, sx, sy, corner)
    }
    if (d.plate_bbox) {
      const colors = PLATE_CATEGORY_COLORS[d.plate_category] ?? PLATE_PENDING_COLOR
      drawPlateBox(ctx, d.plate_bbox, d.plate, colors, offsetX, offsetY, sx, sy, corner * 0.6)
    }
  }
}

// Bracket-corner "targeting reticle" instead of a full rectangle outline —
// reads as a live detection system marking a subject, not a plain drawn box.
function drawReticle(ctx, bbox, color, label, offsetX, offsetY, sx, sy, corner) {
  const [x1, y1, x2, y2] = bbox
  const rx = offsetX + x1 * sx
  const ry = offsetY + y1 * sy
  const rw = (x2 - x1) * sx
  const rh = (y2 - y1) * sy
  const c = Math.min(corner, rw / 2, rh / 2)

  ctx.strokeStyle = color
  ctx.beginPath()
  // top-left
  ctx.moveTo(rx, ry + c); ctx.lineTo(rx, ry); ctx.lineTo(rx + c, ry)
  // top-right
  ctx.moveTo(rx + rw - c, ry); ctx.lineTo(rx + rw, ry); ctx.lineTo(rx + rw, ry + c)
  // bottom-right
  ctx.moveTo(rx + rw, ry + rh - c); ctx.lineTo(rx + rw, ry + rh); ctx.lineTo(rx + rw - c, ry + rh)
  // bottom-left
  ctx.moveTo(rx + c, ry + rh); ctx.lineTo(rx, ry + rh); ctx.lineTo(rx, ry + rh - c)
  ctx.stroke()

  const textWidth = ctx.measureText(label).width
  const tagH = Math.max(14, corner * 1.1)
  ctx.fillStyle = 'rgba(10,14,20,0.75)'
  ctx.fillRect(rx, ry - tagH - 2, textWidth + 8, tagH)
  ctx.fillStyle = color
  ctx.fillText(label, rx + 4, ry - 4)
}

// A thin reticle around the plate box itself, plus — once OCR has actually
// accepted a reading — a filled plate-shaped chip rendered in its real RTO
// category color, matching the plate chips in FinalPlatesTable/
// DetectionTable rather than a generic colored label.
function drawPlateBox(ctx, bbox, plateText, colors, offsetX, offsetY, sx, sy, corner) {
  const [x1, y1, x2, y2] = bbox
  const rx = offsetX + x1 * sx
  const ry = offsetY + y1 * sy
  const rw = (x2 - x1) * sx
  const rh = (y2 - y1) * sy
  const c = Math.min(corner, rw / 2, rh / 2)

  ctx.strokeStyle = colors.bg
  ctx.beginPath()
  ctx.moveTo(rx, ry + c); ctx.lineTo(rx, ry); ctx.lineTo(rx + c, ry)
  ctx.moveTo(rx + rw - c, ry); ctx.lineTo(rx + rw, ry); ctx.lineTo(rx + rw, ry + c)
  ctx.moveTo(rx + rw, ry + rh - c); ctx.lineTo(rx + rw, ry + rh); ctx.lineTo(rx + rw - c, ry + rh)
  ctx.moveTo(rx + c, ry + rh); ctx.lineTo(rx, ry + rh); ctx.lineTo(rx, ry + rh - c)
  ctx.stroke()

  if (!plateText) return

  const chipH = Math.max(16, corner * 1.6)
  const textWidth = ctx.measureText(plateText).width
  const chipW = textWidth + 12
  const chipX = rx
  const chipY = ry + rh + 4

  ctx.fillStyle = colors.bg
  ctx.fillRect(chipX, chipY, chipW, chipH)
  ctx.strokeStyle = 'rgba(0,0,0,0.35)'
  ctx.lineWidth = 1
  ctx.strokeRect(chipX, chipY, chipW, chipH)
  ctx.fillStyle = colors.fg
  ctx.textBaseline = 'middle'
  ctx.fillText(plateText, chipX + 6, chipY + chipH / 2 + 1)
  ctx.textBaseline = 'bottom'
}
