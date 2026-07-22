import { useEffect, useMemo, useState } from 'react'
import { getCamera, getPlateResults, startCamera } from '../services/api'
import LivePlayer from '../components/LivePlayer'
import DetectionTable from '../components/DetectionTable'
import FinalPlatesTable from '../components/FinalPlatesTable'
import SummaryStats from '../components/SummaryStats'
import VehicleTypeBreakdown from '../components/VehicleTypeBreakdown'

const PLATE_POLL_MS = 2000

export default function CameraView({ cameraId }) {
  const [camera, setCamera] = useState(null)
  const [error, setError] = useState(null)
  const [started, setStarted] = useState(false)
  // Every OCR attempt (accepted, rejected, or no readable text) — the
  // backend already returns the full bounded, most-recent-first list, so
  // this is just replaced wholesale on each poll rather than merged.
  const [plateResults, setPlateResults] = useState([])
  // One row per vehicle, not one row per OCR attempt — keeps only the
  // highest-confidence "accepted" reading per track_id, since a track can
  // rack up several accepted attempts (see pipeline.py's confidence-gated
  // OCR-stop) before settling on its best read.
  //
  // The backend's plateResults list is a bounded, rolling window
  // (segment_store.py's _MAX_OCR_ATTEMPTS=300, oldest-dropped) — it exists
  // for short-term debugging, not as a permanent record. If this table only
  // ever derived from the latest poll, an early vehicle's accepted read
  // would silently vanish once enough later OCR attempts (including noisy
  // rejected/no_text ones) pushed it out of that window. So best-per-track
  // is accumulated here across polls and only ever added to/upgraded, never
  // removed just because the backend's snapshot moved on without it.
  const [bestByTrack, setBestByTrack] = useState(new Map())
  // Every track_id that's had at least one OCR attempt (accepted, rejected,
  // or no_text) — a running total accumulated the same way as bestByTrack,
  // for the same reason: plateResults is a bounded rolling snapshot, so a
  // count derived fresh from it each poll could shrink as older attempts
  // age out, instead of only ever growing like "vehicles attempted" should.
  const [attemptedTrackIds, setAttemptedTrackIds] = useState(new Set())

  useEffect(() => {
    getCamera(cameraId)
      .then(setCamera)
      .catch((err) => setError(err.message))
  }, [cameraId])

  // frame_width/frame_height (the source resolution detection bboxes are in)
  // are only known once the backend's pipeline has processed its first frame
  // — null at the moment Play is clicked. Poll briefly until populated so
  // CanvasOverlay can scale boxes correctly instead of assuming they're in
  // the HLS preview stream's (possibly downscaled) decoded resolution.
  useEffect(() => {
    if (!started || !camera || camera.frame_width) return
    const interval = setInterval(() => {
      getCamera(cameraId)
        .then((data) => {
          if (data.frame_width) {
            setCamera(data)
            clearInterval(interval)
          }
        })
        .catch(() => {})
    }, 500)
    return () => clearInterval(interval)
  }, [started, camera, cameraId])

  // OCR runs asynchronously on the backend and can resolve well after its
  // triggering segment was already fetched for the live overlay, so results
  // are polled from a separate endpoint instead of riding along with
  // per-segment detections.
  useEffect(() => {
    if (!started || !camera) return
    const poll = () => {
      getPlateResults(camera.detections_url)
        .then(({ results }) => setPlateResults(results))
        .catch(() => {})
    }
    poll()
    const interval = setInterval(poll, PLATE_POLL_MS)
    return () => clearInterval(interval)
  }, [started, camera])

  // vehicle_count climbs as VehicleTracker mints new track ids — same poll
  // cadence as plate results, just re-fetching the camera endpoint since
  // that's where the backend surfaces it (no dedicated count endpoint).
  useEffect(() => {
    if (!started) return
    const poll = () => {
      getCamera(cameraId)
        .then((data) =>
          setCamera((prev) =>
            prev
              ? { ...prev, vehicle_count: data.vehicle_count, vehicle_count_by_type: data.vehicle_count_by_type }
              : prev
          )
        )
        .catch(() => {})
    }
    poll()
    const interval = setInterval(poll, PLATE_POLL_MS)
    return () => clearInterval(interval)
  }, [started, cameraId])

  useEffect(() => {
    setBestByTrack((prev) => {
      let changed = false
      const next = new Map(prev)
      for (const r of plateResults) {
        if (r.status !== 'accepted') continue
        const existing = next.get(r.track_id)
        if (!existing || r.ocr_confidence > existing.ocr_confidence) {
          next.set(r.track_id, r)
          changed = true
        }
      }
      return changed ? next : prev
    })

    setAttemptedTrackIds((prev) => {
      let changed = false
      const next = new Set(prev)
      for (const r of plateResults) {
        if (!next.has(r.track_id)) {
          next.add(r.track_id)
          changed = true
        }
      }
      return changed ? next : prev
    })
  }, [plateResults])

  const finalPlates = useMemo(() => {
    return Array.from(bestByTrack.values()).sort((a, b) => b.track_id - a.track_id)
  }, [bestByTrack])

  if (error) {
    return (
      <main className="flex items-center justify-center p-8">
        <p className="text-sm text-bad">Failed to load camera: {error}</p>
      </main>
    )
  }
  if (!camera) {
    return (
      <main className="flex items-center justify-center p-8">
        <p className="font-mono text-sm text-muted">Loading camera {cameraId}…</p>
      </main>
    )
  }

  return (
    <main className="flex max-w-[1180px] flex-col gap-5 p-5 pb-12">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-[1.05rem] font-semibold tracking-wide">Camera — {camera.camera_id}</h1>
          <div className="font-mono text-xs text-muted">generation {camera.generation ?? 0}</div>
        </div>
        {started && (
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold ${
              camera.camera_connected
                ? 'border-good/30 text-good'
                : 'border-warn/30 text-warn'
            }`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${camera.camera_connected ? 'bg-good' : 'bg-warn'}`} />
            {camera.camera_connected ? 'Connected' : 'Reconnecting'}
          </span>
        )}
      </div>

      {started ? (
        <>
          <LivePlayer
            cameraId={cameraId}
            hlsUrl={camera.hls_url}
            detectionsUrl={camera.detections_url}
            frameWidth={camera.frame_width}
            frameHeight={camera.frame_height}
            onEnded={() => {
              setStarted(false)
              setPlateResults([])
              setBestByTrack(new Map())
              setAttemptedTrackIds(new Set())
            }}
          />

          <SummaryStats
            crossed={camera.vehicle_count ?? 0}
            attempted={attemptedTrackIds.size}
            platesFound={bestByTrack.size}
          />
          <VehicleTypeBreakdown countByType={camera.vehicle_count_by_type ?? {}} />

          <section className="max-w-3xl overflow-hidden rounded-xl border border-border bg-panel">
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <h2 className="text-sm font-semibold">Final Plates</h2>
              <span className="font-mono text-xs tabular-nums text-muted">{finalPlates.length} vehicles</span>
            </div>
            <FinalPlatesTable plates={finalPlates} />
          </section>

          <section className="max-w-3xl overflow-hidden rounded-xl border border-border bg-panel">
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <h2 className="text-sm font-semibold">All OCR Attempts</h2>
              <span className="font-mono text-xs tabular-nums text-muted">{plateResults.length} logged</span>
            </div>
            <DetectionTable results={plateResults} />
          </section>
        </>
      ) : (
        <div className="flex aspect-video w-full max-w-3xl items-center justify-center rounded-xl border border-border bg-panel">
          <button
            onClick={async () => {
              try {
                // Tells the backend to actually begin reading/processing the
                // source — before this, its models are loaded and idle, not
                // running the detection loop regardless of the process
                // having been launched.
                await startCamera(cameraId)
                setStarted(true)
              } catch (err) {
                setError(err.message)
              }
            }}
            className="rounded-lg bg-accent px-8 py-3.5 text-base font-bold text-[#06181a] hover:brightness-110"
          >
            ▶ Play
          </button>
        </div>
      )}
    </main>
  )
}
