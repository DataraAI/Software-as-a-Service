"""FastAPI service entrypoints for RoboTaskManipulator."""

from robotask_manipulator.api.app import create_api_app

__all__ = ["create_api_app"]
