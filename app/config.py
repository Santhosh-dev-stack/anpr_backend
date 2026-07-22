import os
from pathlib import Path

import torch
from dotenv import load_dotenv

# backend/app/config.py -> backend/.env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Avoid oversubscribing the host: on CPU, torch defaults to using every core,
# which fights the concurrently-running ffmpeg HLS transcode (and anything
# else on the machine) for CPU time instead of leaving it any headroom.
# Tested relaxing this to (cores - 2): measured dropped-frames-per-cycle was
# the same or slightly worse (594-595 vs baseline 545-559), so reverted —
# on this specific machine (already under heavy external load), more torch
# threads just means more contention, not more real throughput.
if DEVICE == "cpu":
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

# backend/app/config.py -> backend/ -> backend/models
BACKEND_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_DIR / "models"

PLATE_MODEL_PATH = str(MODELS_DIR / "vechile_plate_yolov8s_v2.pt")
PLATE_CONF_THRESHOLD = 0.20
# The yolov8s plate weight is ~4x larger than the nano weight and runs on the
# full frame (no vehicle-crop stage first) — at imgsz=960 that was 1-2.4s per
# frame on the old CPU deployment (vs. 0.2-0.6s for the nano weight),
# dropping most of the already-decimated frames before they could even be
# processed. That's why this was traded down to 640 there. On this GPU-PC
# deployment, imgsz=960 measured at ~31ms/frame (see the full pipeline
# timing benchmark) — GPU headroom makes the higher imgsz affordable again,
# recovering small/distant-plate recall that 640 traded away.
PLATE_DETECTION_IMGSZ = 960
# This weight is a combined vehicle+plate model (12 classes: car, truck,
# bus, motorcycle, bicycle, auto, van, emergency_vehicle, tractor, hcm_eme,
# cart, number_plate) — the pipeline still only wants plates, so detection
# is filtered to this one class name (see vehicle_tracker.py).
PLATE_CLASS_NAME = "number_plate"

# Vehicle-only gate on top of PLATE_CONF_THRESHOLD's raw detection floor (see
# vehicle_tracker.py) — a vehicle box below this confidence never reaches
# the tracker at all, so it can't be assigned a track_id (new or
# continuing) and isn't counted. Plate detections are unaffected; they're
# still gated only by PLATE_CONF_THRESHOLD.
VEHICLE_TRACK_MIN_CONFIDENCE = 0.50

OCR_LANG = "en"
FRAME_QUEUE_MAXSIZE = 10

# Decimate a video file's frame rate down to this before it ever reaches the
# pipeline — a fixed, uniform stride (unlike the earlier load-adaptive
# skipping) that cuts CPU workload substantially while keeping consecutive
# processed frames close enough together for VehicleTracker's ByteTrack-
# based IoU matching to still follow normal traffic motion. None/0 disables
# decimation (process every frame).
PROCESSING_FPS = 5

RTSP_RECONNECT_INITIAL_DELAY = 1.0
RTSP_RECONNECT_MAX_DELAY = 30.0

DATABASE_URL = os.environ.get("DATABASE_URL")
DB_POOL_MIN_CONN = 1
DB_POOL_MAX_CONN = 5

# One reference crop saved per newly-counted track_id (see Pipeline.
# _process_vehicle) — a manual sanity-check folder for "is vehicle_count
# real", not a permanent archive; nothing prunes it automatically.
VEHICLE_CROPS_DIR = BACKEND_DIR / "vehicle_crops"

HLS_OUTPUT_DIR = BACKEND_DIR / "hls_output"
FFMPEG_BINARY = "ffmpeg"
HLS_SEGMENT_SECONDS = 2
# RTSP only (see HlsService) — a live source never ends, so its HLS output
# uses a sliding window instead of the file case's growing VOD playlist:
# only this many most-recent segments are kept, older ones deleted from
# disk, bounding storage for a camera that runs for days.
HLS_LIVE_LIST_SIZE = 6

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

# Host/port the browser should use to reach the FastAPI app (WebSocket + HLS +
# REST all share this one process/port). Distinct from the bind host
# (main.py's --ws-host, e.g. 0.0.0.0), which isn't a valid address to connect
# to from a browser.
PUBLIC_HOST = os.environ.get("PUBLIC_HOST", "localhost")
PUBLIC_PORT = int(os.environ.get("PUBLIC_PORT", "8765"))

