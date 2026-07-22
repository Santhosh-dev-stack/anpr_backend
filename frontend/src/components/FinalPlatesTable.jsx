import { plateCategoryStyle } from '../lib/plateCategory'

// One card per vehicle (track_id), not one per OCR attempt like
// DetectionTable — shows only the best accepted reading for each track, so
// a vehicle that took several attempts to validate still only shows up
// once here, as its final/best plate. The plate number itself renders as a
// mini plate badge in its real RTO category color (see plateCategory.js),
// same as the live-feed overlay.
export default function FinalPlatesTable({ plates }) {
  if (plates.length === 0) {
    return <p className="p-4 text-sm text-muted">No validated plates yet.</p>
  }

  return (
    <div className="grid grid-cols-1 gap-px bg-border sm:grid-cols-2 lg:grid-cols-3">
      {plates.map((r) => {
        const category = plateCategoryStyle(r.plate_category)
        return (
          <div key={r.track_id} className="flex items-center gap-3 bg-panel px-3.5 py-3">
            {r.vehicle_image ? (
              <img
                src={r.vehicle_image}
                alt={`Vehicle crop for track ${r.track_id}`}
                className="h-10 w-12 flex-shrink-0 rounded-md bg-panel-3 object-cover"
              />
            ) : (
              <div className="h-10 w-12 flex-shrink-0 rounded-md bg-panel-3" />
            )}
            <div className="min-w-0 flex-1 flex flex-col gap-1.5">
              <span
                className={`inline-flex w-fit rounded font-mono text-[0.82rem] font-bold tracking-wide px-2 py-0.5 border border-black/25 ${
                  category ? category.chip : 'bg-panel-3 text-muted'
                }`}
              >
                {r.plate}
              </span>
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="capitalize text-muted">
                  {r.vehicle_type}
                  {category ? ` · ${category.label}` : ''}
                </span>
                <span className="font-mono tabular-nums text-good">
                  {r.ocr_confidence != null ? `${(r.ocr_confidence * 100).toFixed(0)}%` : '—'}
                </span>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
