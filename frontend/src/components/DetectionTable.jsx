// Every OCR attempt is shown, not just successfully-validated plates — the
// status column/color makes it clear which is which so failures stay
// visible for debugging instead of silently disappearing.
const STATUS_STYLE = {
  accepted: { label: 'OK', className: 'text-green-400' },
  rejected: { label: 'Rejected', className: 'text-yellow-500' },
  no_text: { label: 'No text', className: 'text-gray-500' },
}

// plate_category comes from the backend's HSV plate-background-color
// classifier (ocr/plate_category.py, GPU-PC copy): white/achromatic →
// private, yellow-orange → commercial, teal-green → ev, blue/red →
// government. null for a plateless (untracked) box; "unknown" when the
// crop's background didn't clearly match any of the above.
const CATEGORY_STYLE = {
  private: { label: 'Private', className: 'text-gray-300' },
  commercial: { label: 'Commercial', className: 'text-yellow-400' },
  ev: { label: 'EV', className: 'text-emerald-400' },
  government: { label: 'Government', className: 'text-blue-400' },
  unknown: { label: 'Unknown', className: 'text-gray-500' },
}

export default function DetectionTable({ results }) {
  const rows = results

  if (rows.length === 0) {
    return <p className="mt-3 text-sm text-gray-400">No plate reads yet.</p>
  }

  return (
    // Fixed height + scroll rather than pagination: this table is a live
    // feed (polled every PLATE_POLL_MS), so a page number would shift
    // under the user as new attempts arrive — scroll keeps whatever row
    // they're looking at in place instead.
    <div className="mt-3 max-h-[28rem] max-w-3xl overflow-y-auto rounded border border-gray-800">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="sticky top-0 border-b border-gray-700 bg-[#0b0d12] text-left text-gray-400">
            <th className="py-1.5 pr-4">Track</th>
            <th className="py-1.5 pr-4">Vehicle Image</th>
            <th className="py-1.5 pr-4">Plate Image</th>
            <th className="py-1.5 pr-4">Vehicle</th>
            <th className="py-1.5 pr-4">Category</th>
            <th className="py-1.5 pr-4">Plate</th>
            <th className="py-1.5 pr-4">Confidence</th>
            <th className="py-1.5 pr-4">Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const status = STATUS_STYLE[r.status] ?? STATUS_STYLE.rejected
            const category = CATEGORY_STYLE[r.plate_category]
            return (
              <tr key={r.id} className="border-b border-gray-800">
                <td className="py-1.5 pr-4 text-gray-300">#{r.track_id}</td>
                <td className="py-1.5 pr-4">
                  {r.vehicle_image ? (
                    <img
                      src={r.vehicle_image}
                      alt={`Vehicle crop for track ${r.track_id}`}
                      className="h-12 rounded"
                    />
                  ) : (
                    <span className="text-gray-600">—</span>
                  )}
                </td>
                <td className="py-1.5 pr-4">
                  {r.image ? (
                    <img src={r.image} alt={`Plate crop for track ${r.track_id}`} className="h-8 rounded" />
                  ) : (
                    <span className="text-gray-600">—</span>
                  )}
                </td>
                <td className="py-1.5 pr-4 text-gray-300 capitalize">{r.vehicle_type}</td>
                <td className={`py-1.5 pr-4 ${category ? category.className : 'text-gray-600'}`}>
                  {category ? category.label : '—'}
                </td>
                <td className={`py-1.5 pr-4 font-mono font-semibold ${status.className}`}>{r.plate ?? '—'}</td>
                <td className="py-1.5 pr-4 text-gray-400">
                  {r.ocr_confidence != null ? `${(r.ocr_confidence * 100).toFixed(0)}%` : '—'}
                </td>
                <td className={`py-1.5 pr-4 ${status.className}`}>{status.label}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
