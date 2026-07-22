// Three top-line pipeline stats: how many vehicles crossed (VehicleTracker's
// track_id count), how many we attempted an OCR read on (at least one
// accepted/rejected/no_text attempt), and how many plates were actually
// validated/read. attempted/platesFound come from client-side running
// totals (see CameraView) rather than the raw plateResults snapshot, since
// that's a bounded rolling window (segment_store.py's _MAX_OCR_ATTEMPTS)
// an older vehicle's attempt could otherwise silently drop out of.
const STATS = [
  { key: 'crossed', label: 'Vehicles Crossed', icon: '▣', iconClass: 'bg-accent/10 text-accent' },
  { key: 'attempted', label: 'OCR Attempted', icon: '◎', iconClass: 'bg-warn/10 text-warn' },
  { key: 'platesFound', label: 'Plates Read', icon: '✓', iconClass: 'bg-good/10 text-good' },
]

export default function SummaryStats({ crossed, attempted, platesFound }) {
  const values = { crossed, attempted, platesFound }

  return (
    <div className="grid max-w-3xl grid-cols-3 gap-3">
      {STATS.map((s) => (
        <div key={s.key} className="flex flex-col gap-1.5 rounded-xl border border-border bg-panel px-4 py-4">
          <div className="flex items-center justify-between">
            <span className="text-[0.76rem] text-muted">{s.label}</span>
            <span className={`flex h-7 w-7 items-center justify-center rounded-lg text-sm ${s.iconClass}`}>
              {s.icon}
            </span>
          </div>
          <span className="font-mono text-3xl leading-none font-semibold tabular-nums">{values[s.key]}</span>
        </div>
      ))}
    </div>
  )
}
