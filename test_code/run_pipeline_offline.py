"""Run the real detection+tracking+OCR+validation pipeline over a video file
and dump a per-track results CSV — no FastAPI/DB/websocket/RTSP involved.

Uses the exact same classes as the production pipeline (PlateDetector,
VehicleTracker, PlateReader, normalize_plate — see app/services/pipeline.py)
so this is not a re-implementation that could drift from production
behavior, just the same logic run synchronously against a file instead of
a live camera. OCR runs on every tracked vehicle with a plate found in it
every frame, unthrottled, matching production (see app/services/pipeline.py
/ ocr_worker.py — the OcrGate throttle was removed).

Purpose: produce a baseline CSV on this (CPU) machine, then run the paired
Colab notebook (colab_full_pipeline_test.ipynb) on the same video with the
same models and diff the two CSVs with compare_pipeline_runs.py to check
whether GPU inference changes *what* gets detected/read, not just how fast.

Writes two CSVs:
  --out         one row per track: best/validated plate reading summary.
  --frames-out  one row per tracked vehicle *per frame*: bbox coordinates,
                detection confidence/inference time, and (on frames where
                a plate was found and OCR ran) the raw OCR text/confidence/
                inference time and whether it passed validation.

By default decimates the source to --target-fps (5, matching production's
PROCESSING_FPS in app/config.py) using the exact same fixed-stride formula as
VideoReader (app/camera/video_reader.py): stride = round(native_fps /
target_fps). Pass --target-fps 0 to process every frame instead (no
decimation) — useful for stress-testing tracking independent of what
production's frame rate happens to skip.

Known, intentional difference from production: this script reports
VehicleTracker's raw track_ids as-is, so a vehicle VehicleTracker fragmented
into 2-3 track_ids (see app/tracking/vehicle_tracker.py's docstring for
measured fragmentation rates) still shows up as that many rows here.
Production additionally runs each *accepted* reading through
app.services.plate_identity.PlateIdentity to fold such fragments back into
a single reported vehicle — left out here on purpose, since this script's
value is showing the tracker's actual raw behavior for diagnosis.

Run from the backend/ directory:
    python run_pipeline_offline.py --source /path/to/vid1.mp4 --out local_results.csv --frames-out local_frames.csv
"""

import argparse
import csv
import time

import cv2

from app.detection.plate_detector import PlateDetector
from app.ocr.plate_reader import PlateReader
from app.ocr.plate_validator import normalize_plate
from app.tracking.vehicle_tracker import VehicleTracker


def _crop(frame, bbox):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


class TrackRecord:
    def __init__(self, vehicle_type: str, first_frame: int):
        self.vehicle_type = vehicle_type
        self.first_frame = first_frame
        self.last_frame = first_frame
        self.ocr_attempts = 0
        self.best_text: str | None = None
        self.best_confidence = 0.0
        self.readings: set[str] = set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline full-pipeline run (detect+track+OCR+validate)")
    parser.add_argument("--source", required=True, help="Video file path")
    parser.add_argument("--out", default="pipeline_results.csv", help="Track-summary output CSV path")
    parser.add_argument(
        "--frames-out", default=None,
        help="Per-frame detection CSV path (default: derived from --out, e.g. pipeline_results_frames.csv)",
    )
    parser.add_argument(
        "--target-fps", type=float, default=10.0,
        help="Decimate to this fps, matching production's PROCESSING_FPS (default 5). 0 = every frame.",
    )
    args = parser.parse_args()
    frames_out = args.frames_out or args.out.rsplit(".", 1)[0] + "_frames.csv"

    detector = PlateDetector()
    tracker = VehicleTracker(detector)
    ocr_reader = PlateReader()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    # Same fixed-stride formula as VideoReader.__init__ (app/camera/video_reader.py).
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_stride = max(1, round(native_fps / args.target_fps)) if args.target_fps and native_fps else 1
    print(f"Native fps={native_fps:.2f}, target_fps={args.target_fps}, stride={frame_stride}")

    records: dict[int, TrackRecord] = {}
    frame_id = 0
    processed_count = 0
    t_start = time.perf_counter()

    frames_file = open(frames_out, "w", newline="")
    frames_writer = csv.writer(frames_file)
    frames_writer.writerow([
        "frame_id", "track_id", "vehicle_type",
        "vehicle_bbox_x1", "vehicle_bbox_y1", "vehicle_bbox_x2", "vehicle_bbox_y2",
        "plate_bbox_x1", "plate_bbox_y1", "plate_bbox_x2", "plate_bbox_y2",
        "plate_confidence", "detect_inference_ms",
        "ocr_ran", "ocr_raw_text", "ocr_confidence", "ocr_validated_plate",
        "ocr_status", "ocr_inference_ms",
    ])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        if frame_id % frame_stride != 0:
            continue  # decimated frame, matching VideoReader's stride exactly
        processed_count += 1

        # Matches VideoReader's video_time computation so the tracker's
        # missed-time tolerance means the same thing here as in production,
        # regardless of how fast this offline script processes frames.
        video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        t_detect = time.perf_counter()
        tracked = tracker.track(frame, video_time)
        detect_ms = (time.perf_counter() - t_detect) * 1000

        for vehicle in tracked:
            record = records.get(vehicle.track_id)
            if record is None:
                record = TrackRecord(vehicle.vehicle_type, frame_id)
                records[vehicle.track_id] = record
            record.last_frame = frame_id

            # vehicle.bbox IS the tracked entity's own box now — always
            # present, unlike the old plate-centric TrackedPlate.vehicle_bbox
            # (which could be None if no containing vehicle box was found).
            vx1, vy1, vx2, vy2 = vehicle.bbox
            px1, py1, px2, py2 = vehicle.plate_bbox if vehicle.plate_bbox else ("", "", "", "")
            plate_conf_str = f"{vehicle.plate_confidence:.3f}" if vehicle.plate_confidence is not None else ""

            ocr_ran = False
            ocr_raw_text = ocr_confidence = ocr_validated = ocr_status = ""
            ocr_ms = ""

            # OCR only runs when a plate was actually found inside this
            # vehicle's box this frame — matching production
            # (Pipeline._process_vehicle), no throttling gate otherwise.
            if vehicle.plate_bbox is not None:
                crop = _crop(frame, vehicle.plate_bbox)
                ocr_ran = crop.size > 0

                if ocr_ran:
                    t_ocr = time.perf_counter()
                    result = ocr_reader.read(crop)
                    ocr_ms = f"{(time.perf_counter() - t_ocr) * 1000:.1f}"

                    if result is None:
                        ocr_status = "no_text"
                    else:
                        record.ocr_attempts += 1
                        ocr_raw_text = result.text
                        ocr_confidence = f"{result.confidence:.3f}"

                        normalized = normalize_plate(result.text)
                        if normalized is not None:
                            ocr_status = "accepted"
                            ocr_validated = normalized
                            record.readings.add(normalized)
                            if result.confidence > record.best_confidence:
                                record.best_text = normalized
                                record.best_confidence = result.confidence
                        else:
                            ocr_status = "rejected"

            frames_writer.writerow([
                frame_id, vehicle.track_id, vehicle.vehicle_type,
                vx1, vy1, vx2, vy2,
                px1, py1, px2, py2,
                plate_conf_str, f"{detect_ms:.1f}",
                ocr_ran, ocr_raw_text, ocr_confidence, ocr_validated,
                ocr_status, ocr_ms,
            ])

    frames_file.close()
    cap.release()
    elapsed = time.perf_counter() - t_start
    print(f"Source frames: {frame_id}, processed (after decimation): {processed_count} "
          f"in {elapsed:.1f}s ({processed_count/elapsed:.1f} processed-fps)")
    print(f"Tracks seen: {len(records)}")
    print(f"Wrote {frames_out}")

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "track_id", "vehicle_type", "first_frame", "last_frame",
            "ocr_attempts", "best_plate", "best_confidence", "all_readings",
        ])
        for track_id, r in sorted(records.items()):
            writer.writerow([
                track_id, r.vehicle_type, r.first_frame, r.last_frame,
                r.ocr_attempts, r.best_text or "", f"{r.best_confidence:.3f}",
                ";".join(sorted(r.readings)),
            ])
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()


