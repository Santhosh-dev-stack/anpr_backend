from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.cameras import router as cameras_router
from app.api.detections import router as detections_router
from app.config import CORS_ORIGINS, HLS_OUTPUT_DIR

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_hls(request: Request, call_next):
    # Every cycle (new source file, or the same camera restarted with a
    # different --source) reuses the exact same URLs — /hls/{camera_id}/
    # index.m3u8, seg_00000.ts, seg_00001.ts, ... — since the output
    # directory is per-camera, not per-cycle. Without this, the browser's
    # default heuristic caching can serve a *previous* cycle's (or even a
    # previous video file's) cached segment content for the same URL,
    # instead of the new file the server just wrote — seen as stale
    # video/duration after swapping --source and restarting. This also
    # covers the earlier fix for a segment .ts 404 getting cached (hls.js
    # can request one a fraction of a second before ffmpeg finishes writing
    # it) — no-store means that retry always hits the network for real.
    response = await call_next(request)
    if request.url.path.startswith("/hls/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response


HLS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/hls", StaticFiles(directory=HLS_OUTPUT_DIR), name="hls")

app.include_router(cameras_router)
app.include_router(detections_router)
