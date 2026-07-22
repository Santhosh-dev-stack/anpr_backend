import { plateCategoryStyle } from '../lib/plateCategory'

// Every OCR attempt is shown, not just successfully-validated plates — the
// status column/dot makes it clear which is which so failures stay visible
// for debugging instead of silently disappearing.
const STATUS_STYLE = {
  accepted: { label: 'OK', className: 'text-good' },
  rejected: { label: 'Rejected', className: 'text-warn' },
  no_text: { label: 'No text', className: 'text-muted-2' },
}

export default function DetectionTable({ results }) {
  const rows = results

  if (rows.length === 0) {
    return <p className="p-4 text-sm text-muted">No plate reads yet.</p>
  }

  return (
    // Fixed height + scroll rather than pagination: this table is a live
    // feed (polled every PLATE_POLL_MS), so a page number would shift
    // under the user as new attempts arrive — scroll keeps whatever row
    // they're looking at in place instead.
    <div className="max-h-[28rem] overflow-y-auto">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="sticky top-0 z-10 bg-panel-2 text-left">
            {['Track', 'Vehicle', 'Plate', 'Vehicle', 'Category', 'Plate No.', 'Conf.', 'Status'].map((h) => (
              <th
                key={h}
                className="border-b border-border px-3 py-2 text-[0.68rem] font-semibold uppercase tracking-wider text-muted"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const status = STATUS_STYLE[r.status] ?? STATUS_STYLE.rejected
            const category = plateCategoryStyle(r.plate_category)
            return (
              <tr key={r.id} className="border-b border-border last:border-none hover:bg-panel-2">
                <td className="px-3 py-1.5 font-mono tabular-nums text-muted">#{r.track_id}</td>
                <td className="px-3 py-1.5">
                  {r.vehicle_image ? (
                    <img
                      src={r.vehicle_image}
                      alt={`Vehicle crop for track ${r.track_id}`}
                      className="h-8 w-11 rounded-md bg-panel-3 object-cover"
                    />
                  ) : (
                    <span className="text-muted-2">—</span>
                  )}
                </td>
                <td className="px-3 py-1.5">
                  {r.image ? (
                    <img
                      src={r.image}
                      alt={`Plate crop for track ${r.track_id}`}
                      className="h-8 w-11 rounded-md bg-panel-3 object-cover"
                    />
                  ) : (
                    <span className="text-muted-2">—</span>
                  )}
                </td>
                <td className="px-3 py-1.5 capitalize text-gray-300">{r.vehicle_type}</td>
                <td className={`px-3 py-1.5 text-xs capitalize ${category ? category.text : 'text-muted-2'}`}>
                  {category ? category.label : '—'}
                </td>
                <td className="px-3 py-1.5 font-mono font-semibold">{r.plate ?? <span className="text-muted-2">—</span>}</td>
                <td className="px-3 py-1.5 font-mono tabular-nums text-muted">
                  {r.ocr_confidence != null ? `${(r.ocr_confidence * 100).toFixed(0)}%` : '—'}
                </td>
                <td className={`px-3 py-1.5 ${status.className}`}>
                  <span className="inline-flex items-center gap-1.5 text-xs font-semibold">
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${
                        r.status === 'accepted' ? 'bg-good' : r.status === 'rejected' ? 'bg-warn' : 'bg-muted-2'
                      }`}
                    />
                    {status.label}
                  </span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
