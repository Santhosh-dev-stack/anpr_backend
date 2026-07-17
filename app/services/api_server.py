import asyncio
import threading

import uvicorn

from app.utils.logger import get_logger
from app.websocket.server import app

logger = get_logger(__name__)


class ApiServer:
    """Runs the FastAPI app (HLS static files + REST endpoints) on its own
    event loop in a background thread. Detection metadata is now delivered
    via REST-per-segment (see app/api/detections.py), fetched by the
    frontend directly when hls.js starts playing that segment — no
    WebSocket broadcast needed.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self._host = host
        self._port = port
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._ready = threading.Event()

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(timeout=5)
        logger.info("API server listening on http://%s:%d", self._host, self._port)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        config = uvicorn.Config(app, host=self._host, port=self._port, log_level="warning", loop="asyncio")
        server = uvicorn.Server(config)

        async def _serve():
            self._ready.set()
            await server.serve()

        self._loop.run_until_complete(_serve())
