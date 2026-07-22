"""Live preview window: plate detection + tracking + OCR, drawn on the video
as it plays. Uses the same PlateTracker/PlateReader/plate_validator as the
real pipeline (app/services/pipeline.py) — this is a visual debugging tool,
not part of the app itself.

Run from the backend/ directory:
    python live_preview.py --source /path/to/video.mp4
    python live_preview.py --source rtsp://...

Press ESC to quit.
"""

import argparse
import time

import cv2

from app.detection.plate_detector import PlateDetector
from app.ocr.plate_reader import PlateReader
from app.ocr.plate_validator import normalize_plate
from app.tracking.plate_tracker import PlateTracker

PLATE_BOX_COLOR = (0, 255, 255)  # yellow
TEXT_COLOR = (0, 255, 0)  # green
INFO_COLOR = (0, 255, 255)

# Re-run OCR on a track every N frames it's visible, until it has a validated
# reading — keeps this simple/synchronous (no background worker/gate) since
# it's just a visual demo, not the production async pipeline.
OCR_RETRY_EVERY_FRAMES = 15


def main() -> None:
    parser = argparse.ArgumentParser(description="Live plate detection + tracking + OCR preview")
    parser.add_argument("--source", required=True, help="Video file path or rtsp:// URL")
    parser.add_argument("--conf", type=float, default=0.25, help="Plate detector confidence threshold")
    args = parser.parse_args()

    detector = PlateDetector()
    tracker = PlateTracker(detector)
    ocr_reader = PlateReader()

    # track_id -> best validated plate text seen so far
    plate_text_by_track: dict[int, str] = {}
    # track_id -> frame_id it was last OCR'd, so we don't OCR every single frame
    last_ocr_frame_by_track: dict[int, int] = {}

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Could not open source: {args.source}")
        return

    cv2.namedWindow("ANPR Live Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ANPR Live Preview", 1280, 720)

    frame_id = 0
    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        # Matches VideoReader's video_time computation (app/camera/video_reader.py)
        # so the tracker's missed-time tolerance means the same thing here as
        # it does in production, regardless of how fast this script decodes.
        video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        t0 = time.perf_counter()
        tracked = tracker.track(frame, video_time)
        detect_ms = (time.perf_counter() - t0) * 1000

        for plate in tracked:
            x1, y1, x2, y2 = plate.bbox

            # OCR a track on first sighting, then retry periodically until validated
            needs_ocr = plate.track_id not in plate_text_by_track and (
                plate.track_id not in last_ocr_frame_by_track
                or frame_id - last_ocr_frame_by_track[plate.track_id] >= OCR_RETRY_EVERY_FRAMES
            )
            if needs_ocr:
                last_ocr_frame_by_track[plate.track_id] = frame_id
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size > 0:
                    result = ocr_reader.read(crop)
                    if result is not None:
                        normalized = normalize_plate(result.text)
                        if normalized is not None:
                            plate_text_by_track[plate.track_id] = normalized

            label = plate_text_by_track.get(plate.track_id, f"#{plate.track_id}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), PLATE_BOX_COLOR, 2)
            cv2.putText(
                frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2,
            )

        now = time.time()
        fps = 1 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INFO_COLOR, 2)
        cv2.putText(frame, f"Detect: {detect_ms:.0f}ms", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INFO_COLOR, 2)
        cv2.putText(frame, f"Plates: {len(tracked)}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INFO_COLOR, 2)

        cv2.imshow("ANPR Live Preview", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
