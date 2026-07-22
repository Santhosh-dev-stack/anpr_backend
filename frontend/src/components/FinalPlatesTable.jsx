// plate_category comes from the backend's HSV plate-background-color
// classifier (ocr/plate_category.py, GPU-PC copy): white/achromatic →
// private, yellow-orange → commercial, teal-green → ev, blue/red →
// government; "unknown" when the crop's background didn't clearly match
// any of the above.
const CATEGORY_STYLE = {
  private: { label: 'Private', className: 'text-gray-300' },
  commercial: { label: 'Commercial', className: 'text-yellow-400' },
  ev: { label: 'EV', className: 'text-emerald-400' },
  government: { label: 'Government', className: 'text-blue-400' },
  unknown: { label: 'Unknown', className: 'text-gray-500' },
}

// One row per vehicle (track_id), not one row per OCR attempt like
// DetectionTable — shows only the best accepted reading for each track,
// so a vehicle that took several attempts to validate still only shows up
// once here, as its final/best plate.
export default function FinalPlatesTable({ plates }) {
  if (plates.length === 0) {
    return <p className="mt-3 text-sm text-gray-400">No validated plates yet.</p>
  }

  return (
    <table className="mt-3 w-full max-w-3xl border-collapse text-sm">
      <thead>
        <tr className="border-b border-gray-700 text-left text-gray-400">
          <th className="py-1.5 pr-4">Track</th>
          <th className="py-1.5 pr-4">Vehicle Image</th>
          <th className="py-1.5 pr-4">Plate Image</th>
          <th className="py-1.5 pr-4">Vehicle</th>
          <th className="py-1.5 pr-4">Category</th>
          <th className="py-1.5 pr-4">Plate</th>
          <th className="py-1.5 pr-4">Confidence</th>
        </tr>
      </thead>
      <tbody>
        {plates.map((r) => {
          const category = CATEGORY_STYLE[r.plate_category]
          return (
          <tr key={r.track_id} className="border-b border-gray-800">
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
            <td className="py-1.5 pr-4 font-mono font-semibold text-green-400">{r.plate}</td>
            <td className="py-1.5 pr-4 text-gray-400">
              {r.ocr_confidence != null ? `${(r.ocr_confidence * 100).toFixed(0)}%` : '—'}
            </td>
          </tr>
          )
        })}
      </tbody>
    </table>
  )
}
