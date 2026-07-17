# ANPR Backend — Reference

Internal reference for the ANPR (license plate recognition) backend — GPU-PC backend-only copy.

- **Path:** `~/Documents/sandy_files/anpr_backend`
- **Host:** `facetag4@10.10.192.120` — NVIDIA T600
- **Updated:** 2026-07-17

Vehicle + license-plate detection, tracking, OCR, and category classification, over recorded video or a live RTSP camera.

---

## 1. Overview & pipeline

One process per camera. Each frame passes through detection, tracking, OCR, and identity resolution before landing in an in-memory store the API and HLS preview both read from.

```
Camera (file or rtsp://)
  -> VideoReader / RTSPReader
  -> FrameQueue
  -> PlateTracker (YOLOv8, combined vehicle+plate model)
  -> OcrWorker (PaddleOCR, background thread)
  -> plate_validator + plate_category (HSV color classifier)
  -> PlateIdentity (dedup fragmented track_ids)
  -> SegmentStore + ResultSink
```

Running alongside: `HlsService` transcodes the same source to HLS via ffmpeg for the browser preview — independent of, and unpaced by, the detection loop.

---

## 2. Folder structure

Live and static pipelines were split into separate entrypoints (2026-07-17) so each can be deployed as its own process. Detection, OCR, tracking, and the API/storage layer stayed untouched and shared.

```
app/
├── live/main.py                 # live RTSP entrypoint — auto-starts
├── static/main.py               # video-file entrypoint — needs POST /start
├── main.py                      # backward-compat dispatcher (routes by URL scheme)
├── pipeline_runner.py           # shared bootstrap(), reader_loop(), build_sink()
├── config.py                    # every tunable — always read live, values drift
├── camera/
│   ├── frame_source.py          # FrameSource ABC · Frame · is_rtsp_source()
│   ├── video_reader.py          # paced to real-time PTS, decimates to PROCESSING_FPS
│   └── rtsp_reader.py           # own reconnect thread, .healthy property
├── services/
│   ├── pipeline.py              # orchestrates detect -> OCR -> identity -> sink
│   ├── pipeline_controller.py   # start-gate (Play button / auto-start)
│   ├── hls_service_base.py      # shared launch/manifest-poll/kill
│   ├── static_hls_service.py    # VOD, one-shot manifest wait
│   ├── live_hls_service.py      # sliding window + crash/stall watchdog
│   ├── segment_store.py         # per-camera detection overlay, bounded|unbounded
│   ├── ocr_worker.py, plate_identity.py    # background OCR · track_id dedup
│   └── result_sink.py, frame_queue.py, api_server.py
├── api/
│   ├── cameras.py               # GET /api/cameras/{id} · POST /start
│   └── detections.py            # GET /api/detections/{id}[?segment=N|/plates]
├── detection/plate_detector.py  # YOLOv8 wrapper
├── tracking/plate_tracker.py    # nearest-center + velocity matcher
├── ocr/
│   ├── plate_reader.py, plate_validator.py  # PaddleOCR · text validation
│   └── plate_category.py        # HSV plate-color classifier
├── database/ (schema.py, db_sink.py)  # optional Postgres persistence
└── websocket/server.py           # FastAPI app, CORS, /hls mount

models/            # YOLO weights (.pt, gitignored)
hls_output/        # ffmpeg HLS segments per camera_id (gitignored)
.git/              # initialized 2026-07-17 — first VCS this project has had
```

---

## 3. Running it

### Live · RTSP
```bash
python -m app.live.main \
  --source 'rtsp://user:pass@host/stream1' \
  --camera-id cam1 --api-port 8765
```
Auto-starts — no `POST /start` needed. A password containing `@` must be URL-encoded (`%40`).

### Static · video file
```bash
python -m app.static.main \
  --source sample_video.mp4 \
  --camera-id cam1 --api-port 8765

# then trigger processing:
curl -X POST http://localhost:8765/api/cameras/cam1/start
```

### Backward-compatible
```bash
python -m app.main --source ... --camera-id ... --api-port ...
```
Auto-detects the source type from the URL scheme and dispatches to one of the two above.

---

## 4. API reference

### `GET /api/cameras/{camera_id}`

| Field | Meaning |
|---|---|
| `hls_url` / `detections_url` | Browser-facing stream + detections endpoints |
| `hls_manifest_ready` | ffmpeg has written a real segment — safe to `hls.loadSource()` |
| `hls_ready` | Entire source consumed (file only — never fires for a live camera) |
| `camera_connected` | Live camera's RTSP connection health (always `true` for a file) |
| `hls_generation` | Bumped on each ffmpeg watchdog restart — frontend should fully reinit hls.js on change |
| `vehicle_count` | Distinct vehicles tracked so far this cycle |
| `started` | Whether the detection loop has been triggered |

### `POST /api/cameras/{camera_id}/start`
Triggers the start-gate. Idempotent — safe to call even if already started (e.g. a live camera that auto-started).

### `GET /api/detections/{camera_id}?segment=N`
Per-frame detections for one HLS segment: `track_id`, `vehicle_type`, `vehicle_bbox`, `plate_bbox`, `plate`, `plate_category`, confidences. 404 if not reached yet; empty `frames: []` if reached but pruned or nothing detected.

### `GET /api/detections/{camera_id}/plates`
Every OCR attempt — accepted, rejected, or no_text — newest first, capped at 300.

---

## 5. Config

All in `app/config.py`. This copy has drifted from other recorded values before — always read it live, don't trust a remembered number.

| Key | Value | Note |
|---|---|---|
| `PLATE_CONF_THRESHOLD` | 0.20 | Detector confidence floor |
| `PLATE_DETECTION_IMGSZ` | 640 | Traded down from 960 for CPU throughput |
| `PROCESSING_FPS` | 10 | Video-file decimation target |
| `FRAME_QUEUE_MAXSIZE` | 5 | Drops oldest under backpressure |
| `HLS_SEGMENT_SECONDS` | 2 | ffmpeg segment length, both flavors |
| `HLS_LIVE_LIST_SIZE` | 6 | Live sliding-window segment count (~12s buffer) |
| `RTSP_RECONNECT_INITIAL_DELAY` / `_MAX_DELAY` | 1.0s / 30.0s | Shared backoff — camera reader & ffmpeg watchdog |
| `DATABASE_URL` | unset by default | Postgres persistence off unless set |

---

## 6. Database

Optional — only active when `DATABASE_URL` is set. Otherwise every result is just printed (stdout JSON).

**`vehicle_events`** — one row per `(camera_id, track_id)`, upserted as better reads arrive:

```
camera_id, track_id, vehicle_type, plate_number, plate_category,
vehicle_confidence, plate_confidence, ocr_confidence,
vehicle_bbox, plate_bbox,
first_seen_frame, last_seen_frame, first_seen_at, last_seen_at
```

A track's row converges to its *best* reading — a later, lower-confidence OCR attempt never overwrites a stronger earlier one.

---

## 7. Key design decisions

**Nearest-center + velocity tracking, not ByteTrack.**
ByteTrack's IoU matching lost most tracks at this pipeline's decimated fps — plate boxes move too far frame-to-frame for IoU to catch.

**No OCR throttle, but stop after one confident accept.**
Every frame gets OCR'd until a track reaches ≥0.95 confidence — throttling earlier traded away real reads for CPU savings.

**Plate category from background color, not text.**
HSV rule-based classifier — white/achromatic → private, yellow-orange (hue 8–38, tuned from real footage) → commercial, teal-green (hue 40–100) → EV, blue/red → government. No crop-size floor: a real small plate and a false-positive detection turned out to be the same size, so filtering by area only hurt legitimate small reads.

**Live RTSP made a first-class source, then split into its own folder.**
Auto-start, bounded `SegmentStore`, HLS sliding window, and an ffmpeg crash/stall watchdog with generation tracking — verified against a real camera including a live `kill -9` restart test. Then reorganized into `app/live/` / `app/static/` for separate-process deployment, shared core left untouched.

---

## 8. Known limitations

| Status | Item |
|---|---|
| 🟠 open | OCR near-miss spelling (`TN10BL7364` vs `TN1OBL7364`) isn't deduped — `PlateIdentity` only exact-matches text. |
| 🟠 frontend | `LivePlayer.jsx` (not in this directory) still needs to gate `hls.loadSource()` on `hls_manifest_ready`, and reinit hls.js on `hls_generation` change. |
| 🟢 verified | HLS manifest race, live-camera disk/memory bounding, and the crash-restart watchdog — all fixed and re-verified after the folder split. |

---

## 9. Deployment topology

This directory is the **GPU-PC backend-only copy** — no `frontend/` here. The full monorepo (`ANPR_web3`, backend + frontend) lives separately on the laptop.

- **Backend** — here, on `facetag4@10.10.192.120` (NVIDIA T600, 4GB VRAM), or locally on the laptop CPU for comparison.
- **Frontend** — laptop only, `npm run dev` on port 5173; `VITE_API_BASE_URL` toggles which backend it points at.
- **Version control** — this directory had none until 2026-07-17, initialized specifically as a safety net before the live/static split.

---

*Generated from working session notes, 2026-07-17.*
