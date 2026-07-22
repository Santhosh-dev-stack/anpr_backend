import threading

from app.services.result_sink import DetectionResult

# Caps memory/response size for a long-running session — old attempts are
# dropped oldest-first once this many have accumulated.
_MAX_OCR_ATTEMPTS = 300

# Only applied when bounded=True (RTSP — see SegmentStore.__init__). Larger
# than HlsService's live HLS_LIVE_LIST_SIZE (the video segments themselves
# roll off sooner) since detection metadata is cheap to keep a bit longer,
# giving a slightly-behind frontend some slack.
_MAX_SEGMENTS_RETAINED = 30


class SegmentStore:
    """In-memory, per-camera store of detection results grouped by HLS
    segment index. Replaces the earlier WebSocket per-frame stream: the
    frontend fetches a segment's detections directly when hls.js starts
    playing that segment (FRAG_CHANGED), which is inherently synchronized —
    no separate timeline-matching protocol needed. Reset fresh each time the
    backend starts a new source-loop cycle (a new SegmentStore is created and
    registered), consistent with everything else in the per-cycle pipeline.
    """

    def __init__(self, bounded: bool = False):
        # True for RTSP: a live camera never "ends" the way a video file
        # does, so _segments must be pruned or it grows forever (a memory
        # leak). False (default) keeps every segment forever, since a
        # finite video file's segment count is naturally bounded and a
        # viewer can legitimately rewind to any earlier segment — pruning
        # unconditionally would silently break that.
        self._bounded = bounded
        self._lock = threading.Lock()
        self._segments: dict[int, list[dict]] = {}
        # Under CPU backpressure the bounded frame queue can drop every frame
        # belonging to a whole segment, so add_frame() is never called for
        # it — tracked separately so get_segment() can tell "genuinely not
        # reached yet" (404) apart from "reached, but the pipeline had
        # nothing for it" (200, empty frames — not an error).
        self._highest_segment_seen = -1
        self.total_segments: int | None = None
        self.duration_s: float | None = None
        self.hls_ready = False
        # Distinct from hls_ready above (which only flips true once the
        # *entire* source has been consumed, via mark_source_ended). This
        # flips true once HlsService confirms ffmpeg has written the first
        # segment into index.m3u8 — the frontend should wait for this
        # before calling hls.loadSource(), otherwise it can request the
        # manifest before the file exists on disk (404) and hls.js
        # sometimes exhausts its retry budget without recovering.
        self.hls_manifest_ready = False
        # Bumped whenever this camera's track_id numbering restarts from
        # scratch — either a live camera's HlsService watchdog restarting
        # ffmpeg (RTSP only; ffmpeg's segment numbering resets to 0 on
        # relaunch) or a static video's Play-button restart (Pipeline.
        # reset_for_new_cycle — track_ids there deliberately restart at 1
        # each cycle). Either way, old track_id numbers are no longer
        # meaningfully comparable to new ones, so the frontend needs an
        # explicit signal to fully reinit (hls.js reload / clear detection
        # tables) rather than assume continuity across the boundary.
        self.generation = 0
        # True unless the live source (RTSPReader) is currently
        # disconnected/reconnecting — always True for a file source, which
        # has no such transient-outage concept. See FrameSource.healthy.
        self.camera_healthy = True
        # Detection bboxes are in the raw source frame's pixel coordinates
        # (VideoReader reads the source directly, independent of the HLS
        # transcode). The frontend needs this to scale boxes correctly since
        # the HLS preview stream is encoded at a *different* resolution
        # (downscaled for faster ffmpeg encoding) than the source it detects
        # on — video.videoWidth alone isn't the coordinate space bboxes live in.
        self.frame_width: int | None = None
        self.frame_height: int | None = None
        # Every OCR attempt (accepted, rejected, or nothing readable) —
        # populated asynchronously by OcrWorker, independent of which
        # segment/frame triggered it. Polled by the frontend's Detection
        # Table, decoupled from segment fetches (an OCR result can resolve
        # well after its triggering segment was already served to the
        # player). Kept as a bounded list, not deduped per track, so failed
        # attempts stay visible instead of being silently overwritten by a
        # later better one.
        self._ocr_attempts: list[dict] = []
        self._next_attempt_id = 1
        # Mirrors VehicleTracker.total_vehicle_count for this camera — updated
        # each frame from the pipeline (see Pipeline.process) rather than
        # computed here, since counting "new track" is the tracker's job.
        self.vehicle_count = 0
        # Same count, broken down by vehicle_type — e.g. {"car": 12,
        # "motorcycle": 8}. Same PlateIdentity-duplicate correction already
        # applied to vehicle_count above, just per-type (see Pipeline.process).
        self.vehicle_count_by_type: dict[str, int] = {}

    @classmethod
    def for_live(cls) -> "SegmentStore":
        return cls(bounded=True)

    @classmethod
    def for_file(cls) -> "SegmentStore":
        return cls(bounded=False)

    def set_vehicle_count(self, count: int, count_by_type: dict[str, int]) -> None:
        with self._lock:
            self.vehicle_count = count
            self.vehicle_count_by_type = count_by_type

    def set_camera_healthy(self, healthy: bool) -> None:
        with self._lock:
            self.camera_healthy = healthy

    def set_source_resolution(self, width: int, height: int) -> None:
        with self._lock:
            self.frame_width = width
            self.frame_height = height

    def add_frame(self, segment_index: int, video_time: float, results: list[DetectionResult]) -> None:
        frame_entry = {
            "video_timestamp_sec": video_time,
            "video_timestamp_ms": round(video_time * 1000),
            "detections": [
                {
                    "track_id": r.track_id,
                    "vehicle_type": r.vehicle_type,
                    "vehicle_bbox": list(r.vehicle_bbox) if r.vehicle_bbox else None,
                    "plate_bbox": list(r.plate_bbox) if r.plate_bbox else None,
                    "plate": r.plate,
                    "plate_category": r.plate_category,
                    "vehicle_confidence": r.vehicle_confidence,
                    "ocr_confidence": r.ocr_confidence,
                }
                for r in results
            ],
        }
        with self._lock:
            self._segments.setdefault(segment_index, []).append(frame_entry)
            self._highest_segment_seen = max(self._highest_segment_seen, segment_index)
            if self._bounded and len(self._segments) > _MAX_SEGMENTS_RETAINED:
                # Dicts preserve insertion order, and segment_index only
                # ever increases within one store's lifetime (a generation
                # bump creates a brand new store instead), so the first key
                # is always the oldest.
                oldest_index = next(iter(self._segments))
                del self._segments[oldest_index]

    def get_segment(self, index: int) -> dict | None:
        with self._lock:
            frames = self._segments.get(index)
            reached = index <= self._highest_segment_seen
        if frames is not None:
            return {"segment": index, "frame_count": len(frames), "frames": frames}
        if reached:
            return {"segment": index, "frame_count": 0, "frames": []}
        return None

    def mark_source_ended(self, total_segments: int, duration_s: float) -> None:
        with self._lock:
            self.total_segments = total_segments
            self.duration_s = duration_s
            self.hls_ready = True

    def mark_hls_manifest_ready(self) -> None:
        with self._lock:
            self.hls_manifest_ready = True

    def record_ocr_attempt(
        self,
        track_id: int,
        vehicle_type: str,
        plate_category: str,
        plate: str | None,
        ocr_confidence: float | None,
        image: str,
        vehicle_image: str,
        status: str,
    ) -> None:
        with self._lock:
            self._ocr_attempts.append(
                {
                    "id": self._next_attempt_id,
                    "track_id": track_id,
                    "vehicle_type": vehicle_type,
                    "plate_category": plate_category,
                    "plate": plate,
                    "ocr_confidence": ocr_confidence,
                    "image": image,
                    "vehicle_image": vehicle_image,
                    "status": status,
                }
            )
            self._next_attempt_id += 1
            if len(self._ocr_attempts) > _MAX_OCR_ATTEMPTS:
                self._ocr_attempts.pop(0)

    def get_plate_results(self) -> list[dict]:
        with self._lock:
            return list(reversed(self._ocr_attempts))


_registry: dict[str, SegmentStore] = {}
_registry_lock = threading.Lock()


def register(camera_id: str, store: SegmentStore) -> None:
    with _registry_lock:
        _registry[camera_id] = store


def get(camera_id: str) -> SegmentStore | None:
    with _registry_lock:
        return _registry.get(camera_id)
