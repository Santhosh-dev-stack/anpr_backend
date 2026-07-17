from fastapi import APIRouter, HTTPException

from app import config
from app.services import pipeline_controller, segment_store

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("/{camera_id}")
def get_camera(camera_id: str) -> dict:
    base = f"{config.PUBLIC_HOST}:{config.PUBLIC_PORT}"
    store = segment_store.get(camera_id)
    controller = pipeline_controller.get(camera_id)
    return {
        "camera_id": camera_id,
        "hls_url": f"http://{base}/hls/{camera_id}/index.m3u8",
        "detections_url": f"http://{base}/api/detections/{camera_id}",
        "total_segments": store.total_segments if store else None,
        "duration_s": store.duration_s if store else None,
        "hls_ready": store.hls_ready if store else False,
        # Distinct from hls_ready above (end-of-source flag) — true once
        # ffmpeg's manifest has its first segment, safe for hls.js to load.
        # See HlsService for why this exists separately from POST /start.
        "hls_manifest_ready": store.hls_manifest_ready if store else False,
        # True unless a live (RTSP) source is currently disconnected/
        # reconnecting — always True for a file source. See FrameSource.healthy.
        "camera_connected": store.camera_healthy if store else False,
        # Bumped whenever HlsService's watchdog restarts ffmpeg (RTSP
        # only) — ffmpeg's segment numbering resets to 0 on relaunch, so a
        # frontend should force a full hls.js reinit when this changes
        # rather than assume continuous playback.
        "hls_generation": store.hls_generation if store else 0,
        "frame_width": store.frame_width if store else None,
        "frame_height": store.frame_height if store else None,
        "vehicle_count": store.vehicle_count if store else 0,
        "started": controller.is_started() if controller else False,
    }


@router.post("/{camera_id}/start")
def start_camera(camera_id: str) -> dict:
    controller = pipeline_controller.get(camera_id)
    if controller is None:
        raise HTTPException(status_code=404, detail=f"No pipeline registered for camera {camera_id}")
    controller.trigger_start()
    return {"camera_id": camera_id, "started": True}
