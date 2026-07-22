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
    <div className="grid min-h-screen grid-cols-[248px_1fr] bg-bg text-gray-100 max-[880px]:grid-cols-1">
      <aside className="flex flex-col gap-6 border-r border-border bg-panel p-5 max-[880px]:border-r-0 max-[880px]:border-b">
        <div className="flex items-center gap-2.5 border-b border-border pb-4">
          <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-md bg-gradient-to-br from-accent to-[#1f8a94] font-mono text-sm font-bold text-[#06181a]">
            AN
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold tracking-wide">ANPR Control</div>
            <div className="text-xs text-muted">detection node</div>
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <label className="px-1 text-[0.68rem] uppercase tracking-wider text-muted-2" htmlFor="camera-id-input">
            Camera
          </label>
          <div className="flex items-center gap-1.5 rounded-lg border border-border bg-panel-2 py-1 pl-3 pr-1.5">
            <input
              id="camera-id-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && applyCameraId()}
              className="w-full min-w-0 bg-transparent font-mono text-sm outline-none"
              placeholder="cam01"
            />
            <button
              onClick={applyCameraId}
              className="flex-shrink-0 rounded-md bg-accent px-3 py-1 text-xs font-bold text-[#06181a] hover:brightness-110"
            >
              Load
            </button>
          </div>
        </div>
      </aside>

      {/* key={cameraId} forces a full remount on camera switch — simpler and
          safer than relying on CameraView's internal effects to reset every
          piece of per-camera state (plateResults, bestByTrack, started...)
          on prop change alone. */}
      <CameraView key={cameraId} cameraId={cameraId} />
    </div>
  )
}
