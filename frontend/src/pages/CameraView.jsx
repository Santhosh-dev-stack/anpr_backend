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

  if (error) return <p className="text-red-400">Failed to load camera: {error}</p>
  if (!camera) return <p className="text-gray-400">Loading camera {cameraId}...</p>

  return (
    <div className="p-4">
      <h1 className="text-xl font-semibold mb-3">Camera: {camera.camera_id}</h1>
      {started ? (
        <>
          <SummaryStats
            crossed={camera.vehicle_count ?? 0}
            attempted={attemptedTrackIds.size}
            platesFound={bestByTrack.size}
          />
          <VehicleTypeBreakdown countByType={camera.vehicle_count_by_type ?? {}} />
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
          <h2 className="mt-4 text-lg font-semibold">Final Plates</h2>
          <FinalPlatesTable plates={finalPlates} />
          <h2 className="mt-6 text-lg font-semibold">All OCR Attempts</h2>
          <DetectionTable results={plateResults} />
        </>
      ) : (
        <div className="flex h-96 w-full max-w-3xl items-center justify-center rounded bg-black/40">
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
            className="rounded bg-green-600 px-8 py-4 text-lg font-medium text-white hover:bg-green-500"
          >
            ▶ Play
          </button>
        </div>
      )}
    </div>
  )
}
