"""Quick benchmark: model.predict() (old approach) vs model.track() (ByteTrack,
current PlateTracker approach) -- same model, same video, same settings --
to quantify exactly how much overhead ByteTrack adds per call on this CPU.

FOR DIAGNOSIS ONLY. Doesn't touch app/tracking/plate_tracker.py or any
production code -- loads the same weight directly via ultralytics.YOLO.

Run from the backend/ directory:
    python benchmark_track_vs_predict.py --source /path/to/vid1.mp4
"""

import argparse
import time

import cv2
from ultralytics import YOLO

from app.config import DEVICE, PLATE_CONF_THRESHOLD, PLATE_DETECTION_IMGSZ, PLATE_MODEL_PATH


def bench_predict(video_path: str) -> list[float]:
    model = YOLO(PLATE_MODEL_PATH)
    model.to(DEVICE)
    cap = cv2.VideoCapture(video_path)

    ok, warm = cap.read()
    if ok:
        model.predict(warm, conf=PLATE_CONF_THRESHOLD, imgsz=PLATE_DETECTION_IMGSZ, verbose=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    times = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        model.predict(frame, conf=PLATE_CONF_THRESHOLD, imgsz=PLATE_DETECTION_IMGSZ, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    cap.release()
    return times


def bench_track(video_path: str) -> list[float]:
    model = YOLO(PLATE_MODEL_PATH)
    model.to(DEVICE)
    cap = cv2.VideoCapture(video_path)

    ok, warm = cap.read()
    if ok:
        model.track(
            warm, conf=PLATE_CONF_THRESHOLD, imgsz=PLATE_DETECTION_IMGSZ,
            persist=True, tracker="bytetrack.yaml", verbose=False,
        )
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    times = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        model.track(
            frame, conf=PLATE_CONF_THRESHOLD, imgsz=PLATE_DETECTION_IMGSZ,
            persist=True, tracker="bytetrack.yaml", verbose=False,
        )
        times.append((time.perf_counter() - t0) * 1000)
    cap.release()
    return times


def summarize(label: str, times: list[float]) -> None:
    avg = sum(times) / len(times)
    print(f"{label}: n={len(times)}  avg={avg:.1f}ms  min={min(times):.1f}ms  max={max(times):.1f}ms  "
          f"total={sum(times)/1000:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model.predict() vs model.track() (ByteTrack)")
    parser.add_argument("--source", required=True, help="Video file path")
    args = parser.parse_args()

    print(f"Device: {DEVICE}\n")

    print("=== Running model.predict() (old, no tracking) ===")
    predict_times = bench_predict(args.source)
    summarize("predict()", predict_times)

    print("\n=== Running model.track() (ByteTrack, current PlateTracker) ===")
    track_times = bench_track(args.source)
    summarize("track()  ", track_times)

    avg_predict = sum(predict_times) / len(predict_times)
    avg_track = sum(track_times) / len(track_times)
    print(f"\nByteTrack overhead: {avg_track - avg_predict:+.1f}ms/call avg "
          f"({avg_track / avg_predict:.2f}x {'slower' if avg_track > avg_predict else 'faster'})")


if __name__ == "__main__":
    main()
