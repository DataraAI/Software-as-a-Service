"""Authentication helpers for the annotation API."""

from __future__ import annotations

from fastapi import Header, HTTPException, Request, status


def verify_internal_bearer_token(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """Validate the internal DaaS -> Lambda.ai bearer token when configured."""
    expected = request.app.state.settings.service.auth_token
    if not expected:
        return

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "missing_auth", "message": "Missing Authorization bearer token."},
        )
    scheme, _, provided_token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not provided_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_auth", "message": "Authorization header must use Bearer token format."},
        )
    if provided_token != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "invalid_auth", "message": "Provided bearer token is not authorized."},
        )
