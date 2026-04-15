"""Run the RoboTaskManipulator Lambda.ai annotation service."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robotask_manipulator.api import create_api_app
from robotask_manipulator.config import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional YAML settings file.")
    parser.add_argument("--host", default=None, help="Override the bind host.")
    parser.add_argument("--port", type=int, default=None, help="Override the bind port.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    service_settings = settings.service
    if args.host or args.port:
        service_settings = replace(
            service_settings,
            host=args.host or service_settings.host,
            port=args.port or service_settings.port,
        )
        settings = replace(settings, service=service_settings)

    app = create_api_app(settings=settings)
    uvicorn.run(app, host=settings.service.host, port=settings.service.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
