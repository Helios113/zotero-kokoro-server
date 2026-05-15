from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="zotero-kokoro-server",
        description="Local Kokoro TTS server — OpenAI-compatible API + browser UI for Zotero Read Aloud.",
    )
    parser.add_argument("--host", default=os.environ.get("KOKORO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("KOKORO_PORT", "8880")))
    parser.add_argument("--default-voice", default=os.environ.get("KOKORO_DEFAULT_VOICE", "af_heart"))
    parser.add_argument("--repo-id", default=os.environ.get("KOKORO_REPO_ID", "hexgrad/Kokoro-82M"))
    parser.add_argument("--cache-size", type=int, default=256, help="Max in-memory audio cache entries")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (development)")
    args = parser.parse_args()

    from .engine import KokoroEngine
    from .server import create_app
    import uvicorn

    engine = KokoroEngine(
        default_voice=args.default_voice,
        repo_id=args.repo_id,
        cache_max_entries=args.cache_size,
    )
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", reload=args.reload)
