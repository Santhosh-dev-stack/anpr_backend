from fastapi import APIRouter, HTTPException

from app.services import segment_store

router = APIRouter(prefix="/api/detections", tags=["detections"])


@router.get("/{camera_id}")
def get_detections(camera_id: str, segment: int) -> dict:
    store = segment_store.get(camera_id)
    if store is None:
        raise HTTPException(status_code=404, detail=f"No active stream for camera {camera_id}")
    result = store.get_segment(segment)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Segment {segment} not available yet for camera {camera_id}"
        )
    return result


@router.get("/{camera_id}/plates")
def get_plate_results(camera_id: str) -> dict:
    store = segment_store.get(camera_id)
    if store is None:
        raise HTTPException(status_code=404, detail=f"No active stream for camera {camera_id}")
    return {"results": store.get_plate_results()}
