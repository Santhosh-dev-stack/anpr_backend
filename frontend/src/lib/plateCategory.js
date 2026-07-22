// India RTO plate-background color convention (see app/ocr/plate_category.py):
// private=white, commercial=yellow, ev=green, government=blue/red. Shared
// here so the live-feed overlay, Final Plates grid, and OCR Attempts log
// all render a category in the same real plate color instead of each
// picking its own palette.
export const PLATE_CATEGORY_STYLE = {
  private: { label: 'Private', chip: 'bg-plate-private text-plate-private-fg', text: 'text-gray-300' },
  commercial: { label: 'Commercial', chip: 'bg-plate-commercial text-plate-commercial-fg', text: 'text-plate-commercial' },
  ev: { label: 'EV', chip: 'bg-plate-ev text-plate-ev-fg', text: 'text-plate-ev' },
  government: { label: 'Government', chip: 'bg-plate-gov text-plate-gov-fg', text: 'text-[#5b8ff0]' },
  unknown: { label: 'Unknown', chip: 'bg-panel-3 text-muted', text: 'text-muted' },
}

export function plateCategoryStyle(category) {
  return PLATE_CATEGORY_STYLE[category] ?? null
}
