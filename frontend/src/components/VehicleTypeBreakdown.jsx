// Same "vehicles crossed" number as SummaryStats, split by vehicle_type
// (car/motorcycle/truck/...) — comes straight from the backend's
// vehicle_count_by_type (app/services/pipeline.py), already corrected for
// PlateIdentity-folded duplicates the same way the total is.
export default function VehicleTypeBreakdown({ countByType }) {
  const entries = Object.entries(countByType).sort((a, b) => b[1] - a[1])

  if (entries.length === 0) {
    return null
  }

  return (
    <div className="mb-4 flex max-w-3xl flex-wrap gap-2">
      {entries.map(([type, count]) => (
        <div
          key={type}
          className="flex items-center gap-1.5 rounded bg-black/40 px-3 py-1.5 text-sm capitalize"
        >
          <span className="text-gray-400">{type}</span>
          <span className="font-semibold text-white">{count}</span>
        </div>
      ))}
    </div>
  )
}
