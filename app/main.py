import argparse

from app.camera.frame_source import is_rtsp_source
from app.live.main import run as run_live
from app.static.main import run as run_static


def main() -> None:
    # Thin backward-compatible dispatcher — prefer invoking app.live.main or
    # app.static.main directly for a new deployment (see those modules for
    # the actual per-source-type logic); kept so anything still launching
    # this module by its old path (systemd unit, supervisor config, shell
    # wrapper — none of which live in this repo to check) keeps working.
    parser = argparse.ArgumentParser(description="Run the ANPR detection pipeline")
    parser.add_argument("--source", required=True, help="Path to a video file or an rtsp:// URL")
    parser.add_argument("--camera-id", required=True, help="Identifier for this camera")
    parser.add_argument("--api-host", default="0.0.0.0", help="Host to bind the API server to")
    parser.add_argument("--api-port", type=int, default=8765, help="Port for the API server")
    args = parser.parse_args()

    run = run_live if is_rtsp_source(args.source) else run_static
    run(args.source, args.camera_id, args.api_host, args.api_port)


if __name__ == "__main__":
    main()
