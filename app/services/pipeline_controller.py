import threading


class PipelineController:
    """Gates when a camera's video-reading/detection loop actually begins.

    Models load eagerly at process startup (that's slow — several seconds —
    and paying it upfront means no cold-start delay once the frontend
    actually asks to start watching). The frame-processing loop itself,
    however, blocks on `wait_for_start()` so it never runs unless something
    has explicitly asked it to, instead of always running the moment the
    process launches regardless of whether any frontend is even open.
    """

    def __init__(self):
        self._start_event = threading.Event()

    def trigger_start(self) -> None:
        self._start_event.set()

    def wait_for_start(self, timeout: float | None = None) -> bool:
        return self._start_event.wait(timeout)

    def is_started(self) -> bool:
        return self._start_event.is_set()

    def reset(self) -> None:
        # Called once a video-file cycle finishes, so a later Play click
        # (POST /start) can trigger a fresh cycle instead of the process
        # having already permanently "started" from the first click.
        self._start_event.clear()


_registry: dict[str, PipelineController] = {}
_registry_lock = threading.Lock()


def register(camera_id: str, controller: PipelineController) -> None:
    with _registry_lock:
        _registry[camera_id] = controller


def get(camera_id: str) -> PipelineController | None:
    with _registry_lock:
        return _registry.get(camera_id)
