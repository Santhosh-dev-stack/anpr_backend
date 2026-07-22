import { useEffect, useRef, useState } from 'react'
import Hls from 'hls.js'
import { getCamera, getDetections } from '../services/api'
import CanvasOverlay from './CanvasOverlay'

const RETRY_DELAY_MS = 1200
const MAX_ATTEMPTS = 5

// How often to poll GET /api/cameras/{id} for hls_manifest_ready, and how
// long to wait before giving up. ffmpeg needs to actually write a real
// segment reference into index.m3u8 (not just create the file — a
// freshly-created manifest is just an empty header) before hls.js can
// safely request it; requesting too early can exhaust hls.js's own retry
// budget and never recover for this page load without a refresh. The
// timeout is kept comfortably above the backend's own wait timeout
// (_MANIFEST_WAIT_TIMEOUT in hls_service.py, 20s) so a genuine backend
// failure surfaces as a backend log/warning first, not just a frontend
// timeout racing it.
const MANIFEST_POLL_MS = 400
const MANIFEST_WAIT_TIMEOUT_MS = 25000
// After the stream is up, poll less aggressively — this is just watching
// for an RTSP disconnect (camera_connected) or an ffmpeg-watchdog restart
// (generation), not racing a tight startup window anymore.
const STATE_POLL_MS = 1000

export default function LivePlayer({ cameraId, hlsUrl, detectionsUrl, frameWidth, frameHeight, onEnded }) {
  const videoRef = useRef(null)
  const hlsInstanceRef = useRef(null)
  // Tracks the most recently requested segment so a slow retry for an old
  // segment can't clobber a newer one's already-applied result.
  const latestSegmentRef = useRef(-1)

  const [frames, setFrames] = useState([])
  const [buffering, setBuffering] = useState(true)
  const [currentSegment, setCurrentSegment] = useState(null)
  // null: still waiting; true: ready, playback wired up; a string: gave up
  // with this error message (shown with a retry button).
  const [manifestState, setManifestState] = useState(null)
  // Only meaningful for an RTSP source — a file source never flips this
  // false on the backend, so it's always true there and this banner never
  // shows for file playback. Drives the "reconnecting" banner so a frozen
  // video during an RTSP drop reads as "camera dropped", not as a bug.
  const [connected, setConnected] = useState(true)
  const [retryCount, setRetryCount] = useState(0)

  const fetchWithRetry = (segment, attempt = 1) => {
    getDetections(detectionsUrl, segment)
      .then((result) => {
        if (latestSegmentRef.current !== segment) return
        setFrames(result.frames)
      })
      .catch(() => {
        if (latestSegmentRef.current !== segment) return
        if (attempt < MAX_ATTEMPTS) {
          setTimeout(() => fetchWithRetry(segment, attempt + 1), RETRY_DELAY_MS)
        } else {
          setFrames([])
        }
      })
  }

  // (Re)creates the hls.js instance against `url` and wires up playback —
  // called once when the stream first becomes ready, and again whenever
  // generation changes (ffmpeg watchdog restarted it, so segment
  // numbering jumped backward to 0 — hls.js won't handle that on its own,
  // it needs a full teardown/recreate, not just another loadSource on the
  // same instance).
  const setupPlayback = (url) => {
    const video = videoRef.current
    if (!video) return

    hlsInstanceRef.current?.destroy()
    hlsInstanceRef.current = null
    setBuffering(true)
    setCurrentSegment(null)

    let hls
    if (Hls.isSupported()) {
      hls = new Hls()
      hls.loadSource(url)
      hls.attachMedia(video)
      // Fetching a segment's detections right when hls.js moves into that
      // segment is inherently synchronized — no separate timeline-matching
      // protocol needed, unlike the previous WebSocket + video_time approach.
      hls.on(Hls.Events.FRAG_CHANGED, (_event, data) => {
        latestSegmentRef.current = data.frag.sn
        setCurrentSegment(data.frag.sn)
        fetchWithRetry(data.frag.sn)
      })
      hlsInstanceRef.current = hls
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = url
    }
  }

  // Single continuous poll: fast while waiting for the stream to first
  // become ready (racing ffmpeg's first segment write), then slower just to
  // watch camera_connected/hls_generation for the lifetime of the player.
  useEffect(() => {
    let cancelled = false
    let readyOnce = false
    let lastGen = null
    const deadline = Date.now() + MANIFEST_WAIT_TIMEOUT_MS

    const poll = () => {
      if (cancelled) return
      getCamera(cameraId)
        .then((data) => {
          if (cancelled) return
          setConnected(data.camera_connected ?? true)

          if (!readyOnce) {
            if (data.hls_manifest_ready) {
              readyOnce = true
              lastGen = data.generation ?? 0
              setManifestState(true)
            } else if (Date.now() >= deadline) {
              setManifestState('Stream failed to start — try again.')
              return
            }
          } else {
            const gen = data.generation ?? 0
            if (gen !== lastGen) {
              lastGen = gen
              if (data.hls_manifest_ready) setupPlayback(data.hls_url)
            }
          }

          setTimeout(poll, readyOnce ? STATE_POLL_MS : MANIFEST_POLL_MS)
        })
        .catch(() => {
          if (cancelled) return
          if (!readyOnce && Date.now() >= deadline) {
            setManifestState('Stream failed to start — try again.')
            return
          }
          setTimeout(poll, readyOnce ? STATE_POLL_MS : MANIFEST_POLL_MS)
        })
    }
    poll()

    return () => {
      cancelled = true
    }
  }, [cameraId, retryCount])

  // Runs the *first* setupPlayback, once the stream is confirmed ready and
  // <video> is actually in the DOM (the poll above only flips manifestState
  // to true — it can't call setupPlayback itself since videoRef isn't
  // mounted yet at that point, this render hasn't happened). Later
  // regenerations call setupPlayback directly from the poll instead, since
  // by then <video> already exists.
  useEffect(() => {
    if (manifestState !== true) return
    const video = videoRef.current
    if (!video) return

    setupPlayback(hlsUrl)

    const onCanPlay = () => {
      setBuffering(false)
      video.play().catch((err) => console.warn('Autoplay blocked:', err))
    }
    video.addEventListener('canplay', onCanPlay)

    const stopAll = () => {
      video.pause()
      setCurrentSegment(null)
      onEnded?.()
    }
    video.addEventListener('ended', stopAll)

    return () => {
      video.removeEventListener('canplay', onCanPlay)
      video.removeEventListener('ended', stopAll)
      hlsInstanceRef.current?.destroy()
      hlsInstanceRef.current = null
    }
  }, [manifestState, hlsUrl, detectionsUrl])

  if (typeof manifestState === 'string') {
    return (
      <div className="flex h-96 w-full max-w-3xl flex-col items-center justify-center gap-3 rounded bg-black/40 text-gray-300">
        <p>{manifestState}</p>
        <button
          onClick={() => setRetryCount((c) => c + 1)}
          className="rounded bg-green-600 px-4 py-2 text-sm font-medium text-white hover:bg-green-500"
        >
          Retry
        </button>
      </div>
    )
  }

  if (manifestState !== true) {
    return (
      <div className="flex h-96 w-full max-w-3xl items-center justify-center rounded bg-black/40 text-gray-400">
        Starting stream...
      </div>
    )
  }

  return (
    <div className="relative inline-block max-w-full">
      <video ref={videoRef} controls muted className="max-w-full" />
      <CanvasOverlay
        frames={frames}
        videoRef={videoRef}
        frameWidth={frameWidth}
        frameHeight={frameHeight}
      />
      {currentSegment !== null && (
        <div className="absolute left-2 top-2 rounded bg-black/60 px-2 py-1 text-xs text-white">
          Segment #{currentSegment}
        </div>
      )}
      {!connected && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-lg text-white">
          Camera disconnected — reconnecting...
        </div>
      )}
      {buffering && connected && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-lg text-white">
          Buffering video and detection data...
        </div>
      )}
    </div>
  )
}
