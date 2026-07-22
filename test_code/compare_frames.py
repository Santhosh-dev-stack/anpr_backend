"""Compare two run_pipeline_offline.py / colab notebook *_frames.csv files
row-by-row -- unlike compare_pipeline_runs.py (which only compares the final
validated-plate summary), this checks every detected plate on every frame:
same bbox/confidence, same OCR text, and the per-stage inference-time delta.

Rows are matched by (frame_id, nearest plate-bbox center) rather than exact
bbox coordinates or track_id: track IDs are assigned independently in each
run, and even the SAME physical plate can land at very slightly different
pixel coordinates between two runs -- always true when comparing two
different model weights (e.g. yolov8s vs yolov8m), and even possible
between two runs of the identical model/weights due to floating-point
non-determinism. Exact-coordinate matching only works when comparing the
literal same model on the same frames (e.g. CPU vs GPU) where detections
are bit-for-bit reproducible; nearest-center matching (within
--match-radius pixels) works for both that case and cross-model comparisons.

Usage:
    python compare_frames.py without_ocr/colab_frames.csv with_ocr/colab_frames.csv
    python compare_frames.py model_s_5fps/frames.csv model_m_5fps/frames.csv --match-radius 30
"""

import argparse
import csv
import math
from collections import defaultdict
from statistics import mean


def _center(row: dict, prefix: str) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(row[f"{prefix}_{f}"]) for f in ("x1", "y1", "x2", "y2"))
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def load_rows_by_frame(path: str) -> dict[str, list[dict]]:
    by_frame: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_frame[row["frame_id"]].append(row)
    return by_frame


def match_frame_rows(
    rows_a: list[dict], rows_b: list[dict], match_radius: float
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Greedy nearest-center matching within one frame_id. Returns
    (matched_pairs, unmatched_a, unmatched_b)."""
    remaining_b = list(rows_b)
    matched: list[tuple[dict, dict]] = []
    unmatched_a: list[dict] = []

    for ra in rows_a:
        ca = _center(ra, "plate_bbox")
        best_idx, best_dist = None, match_radius
        for i, rb in enumerate(remaining_b):
            dist = math.dist(ca, _center(rb, "plate_bbox"))
            if dist < best_dist:
                best_idx, best_dist = i, dist
        if best_idx is not None:
            matched.append((ra, remaining_b.pop(best_idx)))
        else:
            unmatched_a.append(ra)

    return matched, unmatched_a, remaining_b


def _print_stats(label: str, values: list[float]) -> None:
    if not values:
        print(f"{label}: no data")
        return
    print(f"{label}: avg={mean(values):.1f}  min={min(values):.1f}  max={max(values):.1f}  (n={len(values)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two per-frame pipeline CSVs")
    parser.add_argument("run_a", help="e.g. without_ocr/colab_frames.csv")
    parser.add_argument("run_b", help="e.g. with_ocr/colab_frames.csv")
    parser.add_argument(
        "--match-radius", type=float, default=30.0,
        help="Max pixel distance between plate-bbox centers to count as the same "
             "detection (default 30 -- same physical plate, two model weights/runs). "
             "Use a small value like 2-5 when comparing bit-identical runs.",
    )
    args = parser.parse_args()

    frames_a = load_rows_by_frame(args.run_a)
    frames_b = load_rows_by_frame(args.run_b)

    total_a = sum(len(rows) for rows in frames_a.values())
    total_b = sum(len(rows) for rows in frames_b.values())
    print(f"{args.run_a}: {total_a} detected plate-frames")
    print(f"{args.run_b}: {total_b} detected plate-frames")

    all_frame_ids = sorted(set(frames_a) | set(frames_b), key=lambda x: int(x))
    matched_pairs: list[tuple[dict, dict]] = []
    only_a_count = 0
    only_b_count = 0

    for frame_id in all_frame_ids:
        rows_a = frames_a.get(frame_id, [])
        rows_b = frames_b.get(frame_id, [])
        pairs, unmatched_a, unmatched_b = match_frame_rows(rows_a, rows_b, args.match_radius)
        matched_pairs.extend(pairs)
        only_a_count += len(unmatched_a)
        only_b_count += len(unmatched_b)

    print(f"Matched (same frame, plate-bbox centers within {args.match_radius:.0f}px): {len(matched_pairs)}")
    if only_a_count:
        print(f"Only in {args.run_a}: {only_a_count}")
    if only_b_count:
        print(f"Only in {args.run_b}: {only_b_count}")

    if not matched_pairs:
        print("\nNo matched rows -- nothing further to compare. Try a larger --match-radius "
              "if these are genuinely the same plates at slightly different coordinates.")
        return

    conf_mismatches = 0
    ocr_status_mismatches = 0
    ocr_text_mismatches = 0
    detect_ms_a, detect_ms_b = [], []
    ocr_ms_a, ocr_ms_b = [], []
    text_mismatch_details: list[tuple[dict, dict]] = []

    for ra, rb in matched_pairs:
        if ra["plate_confidence"] != rb["plate_confidence"]:
            conf_mismatches += 1

        detect_ms_a.append(float(ra["detect_inference_ms"]))
        detect_ms_b.append(float(rb["detect_inference_ms"]))

        if ra["ocr_ran"] == "True" and rb["ocr_ran"] == "True":
            if ra["ocr_status"] != rb["ocr_status"]:
                ocr_status_mismatches += 1
            if ra["ocr_validated_plate"] != rb["ocr_validated_plate"]:
                ocr_text_mismatches += 1
                text_mismatch_details.append((ra, rb))
            if ra["ocr_inference_ms"]:
                ocr_ms_a.append(float(ra["ocr_inference_ms"]))
            if rb["ocr_inference_ms"]:
                ocr_ms_b.append(float(rb["ocr_inference_ms"]))

    print(f"\nPlate-confidence mismatches: {conf_mismatches}/{len(matched_pairs)}")
    print(f"OCR status (accepted/rejected/no_text) mismatches: {ocr_status_mismatches}")
    print(f"OCR validated-plate text mismatches: {ocr_text_mismatches}")

    if text_mismatch_details:
        print(f"\n=== OCR text mismatch detail (frame_id: {args.run_a} -> {args.run_b}) ===")
        for ra, rb in text_mismatch_details:
            a_text = ra["ocr_validated_plate"] or f"[{ra['ocr_status']}] {ra['ocr_raw_text']}"
            b_text = rb["ocr_validated_plate"] or f"[{rb['ocr_status']}] {rb['ocr_raw_text']}"
            a_conf = ra["ocr_confidence"] or "-"
            b_conf = rb["ocr_confidence"] or "-"
            print(f"  frame {ra['frame_id']:>5}: {a_text!r:22} (conf={a_conf})  vs  {b_text!r:22} (conf={b_conf})")

    print(f"\n=== Detection timing (ms) ===")
    _print_stats(args.run_a, detect_ms_a)
    _print_stats(args.run_b, detect_ms_b)

    print(f"\n=== OCR timing (ms), rows where both ran OCR ===")
    _print_stats(args.run_a, ocr_ms_a)
    _print_stats(args.run_b, ocr_ms_b)
    if ocr_ms_a and ocr_ms_b:
        print(f"Speedup ({args.run_b} vs {args.run_a}): {mean(ocr_ms_a)/mean(ocr_ms_b):.2f}x")


if __name__ == "__main__":
    main()
