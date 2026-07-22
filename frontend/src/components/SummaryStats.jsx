// Three top-line pipeline stats: how many vehicles crossed (VehicleTracker's
// track_id count), how many we attempted an OCR read on (at least one
// accepted/rejected/no_text attempt), and how many plates were actually
// validated/read. attempted/platesFound come from client-side running
// totals (see CameraView) rather than the raw plateResults snapshot, since
// that's a bounded rolling window (segment_store.py's _MAX_OCR_ATTEMPTS)
// an older vehicle's attempt could otherwise silently drop out of.
export default function SummaryStats({ crossed, attempted, platesFound }) {
  const stats = [
    { label: 'Vehicles Crossed', value: crossed, className: 'text-white' },
    { label: 'OCR Attempted', value: attempted, className: 'text-yellow-400' },
    { label: 'Plates Read', value: platesFound, className: 'text-green-400' },
  ]

  return (
    <div className="mb-4 grid max-w-3xl grid-cols-3 gap-3">
      {stats.map((s) => (
        <div key={s.label} className="rounded bg-black/40 px-4 py-3 text-center">
          <div className={`text-2xl font-semibold ${s.className}`}>{s.value}</div>
          <div className="mt-1 text-xs text-gray-400">{s.label}</div>
        </div>
      ))}
    </div>
  )
}
