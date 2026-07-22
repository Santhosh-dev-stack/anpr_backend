import { useState } from 'react'
import CameraView from './pages/CameraView'

// No backend endpoint enumerates running cameras — each backend process is
// started with one fixed --camera-id (see app/static/main.py / app/live/
// main.py), so the frontend has to be told which one to point at rather
// than discovering it. Read from ?camera= so a specific camera is
// bookmarkable/shareable; default to cam01 to match the previous
// hardcoded behavior when no param is given.
function initialCameraId() {
  const fromUrl = new URLSearchParams(window.location.search).get('camera')
  return fromUrl || 'cam01'
}

export default function App() {
  const [cameraId, setCameraId] = useState(initialCameraId)
  const [input, setInput] = useState(cameraId)

  const applyCameraId = () => {
    const trimmed = input.trim()
    if (!trimmed || trimmed === cameraId) return
    setCameraId(trimmed)
    const url = new URL(window.location.href)
    url.searchParams.set('camera', trimmed)
    window.history.replaceState({}, '', url)
  }

  return (
    <div className="min-h-screen bg-[#0b0d12] text-gray-100">
      <div className="flex items-center gap-2 border-b border-gray-800 p-3">
        <label className="text-sm text-gray-400" htmlFor="camera-id-input">
          Camera ID:
        </label>
        <input
          id="camera-id-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && applyCameraId()}
          className="rounded border border-gray-700 bg-black/40 px-2 py-1 text-sm text-white"
          placeholder="cam01"
        />
        <button
          onClick={applyCameraId}
          className="rounded bg-gray-700 px-3 py-1 text-sm font-medium text-white hover:bg-gray-600"
        >
          Load
        </button>
      </div>
      {/* key={cameraId} forces a full remount on camera switch — simpler and
          safer than relying on CameraView's internal effects to reset every
          piece of per-camera state (plateResults, bestByTrack, started...)
          on prop change alone. */}
      <CameraView key={cameraId} cameraId={cameraId} />
    </div>
  )
}
