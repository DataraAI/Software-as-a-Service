"""FastAPI application for Lambda.ai VM deployment."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from robotask_manipulator.api.auth import verify_internal_bearer_token
from robotask_manipulator.api.models import (
    AnnotationErrorDetail,
    AnnotationErrorResponse,
    AnnotationImageRequest,
    AnnotationJobAcceptedResponse,
    AnnotationResponse,
)
from robotask_manipulator.config import AppSettings, load_settings
from robotask_manipulator.service import AnnotationProcessingError, AnnotationService, AnnotationServiceRequestError
from robotask_manipulator.storage import AzureBlobStorageError, SourceBlobNotFoundError
from robotask_manipulator.utils.validation import RoboTaskManipulatorError

LOGGER = logging.getLogger(__name__)


def create_api_app(
    *,
    settings: AppSettings | None = None,
    annotation_service: AnnotationService | None = None,
) -> FastAPI:
    """Create the Lambda.ai VM-hosted annotation API."""
    resolved_settings = settings or load_settings()
    resolved_service = annotation_service or AnnotationService(resolved_settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.settings.service.warm_load:
            application.state.annotation_service.ensure_ready()
        yield

    app = FastAPI(
        title="RoboTaskManipulator Annotation Service",
        version="1.0.0",
        description=(
            "Lambda.ai VM-hosted sync image annotation service for DaaS. Milestone 1 supports single-image "
            "annotation requests and Milestone 2 reserves async image/video job routes."
        ),
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.annotation_service = resolved_service

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        message = "; ".join(error.get("msg", "Invalid request.") for error in exc.errors()) or "Invalid request."
        payload = AnnotationErrorResponse(
            status="failed",
            error=AnnotationErrorDetail(code="validation_error", message=message),
        )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload.model_dump(mode="json"))

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": str(exc.detail)}
        payload = AnnotationErrorResponse(
            status="failed" if exc.status_code != status.HTTP_501_NOT_IMPLEMENTED else "not_implemented",
            error=AnnotationErrorDetail(
                code=str(detail.get("code", "http_error")),
                message=str(detail.get("message", "Request failed.")),
            ),
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump(mode="json"))

    @app.exception_handler(SourceBlobNotFoundError)
    async def handle_missing_source(_: Request, exc: SourceBlobNotFoundError) -> JSONResponse:
        payload = AnnotationErrorResponse(
            status="failed",
            error=AnnotationErrorDetail(code="source_not_found", message=str(exc)),
        )
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=payload.model_dump(mode="json"))

    @app.exception_handler(AnnotationServiceRequestError)
    async def handle_annotation_request_error(_: Request, exc: AnnotationServiceRequestError) -> JSONResponse:
        payload = AnnotationErrorResponse(
            status="failed",
            error=AnnotationErrorDetail(code="bad_request", message=str(exc)),
        )
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=payload.model_dump(mode="json"))

    @app.exception_handler(AzureBlobStorageError)
    async def handle_azure_storage_error(_: Request, exc: AzureBlobStorageError) -> JSONResponse:
        payload = AnnotationErrorResponse(
            status="failed",
            error=AnnotationErrorDetail(code="bad_request", message=str(exc)),
        )
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=payload.model_dump(mode="json"))

    @app.exception_handler(AnnotationProcessingError)
    async def handle_annotation_processing_error(_: Request, exc: AnnotationProcessingError) -> JSONResponse:
        LOGGER.exception("Annotation processing error: %s", exc)
        payload = AnnotationErrorResponse(
            status="failed",
            error=AnnotationErrorDetail(code="annotation_failed", message=str(exc)),
        )
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload.model_dump(mode="json"))

    @app.exception_handler(RoboTaskManipulatorError)
    async def handle_rtm_error(_: Request, exc: RoboTaskManipulatorError) -> JSONResponse:
        LOGGER.exception("Unhandled RoboTaskManipulator error: %s", exc)
        payload = AnnotationErrorResponse(
            status="failed",
            error=AnnotationErrorDetail(code="annotation_failed", message=str(exc)),
        )
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload.model_dump(mode="json"))

    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, Any]:
        return request.app.state.annotation_service.health_summary()

    @app.post(
        "/v1/annotations/image",
        response_model=AnnotationResponse,
        dependencies=[Depends(verify_internal_bearer_token)],
    )
    async def annotate_image(request_payload: AnnotationImageRequest, request: Request) -> AnnotationResponse:
        return request.app.state.annotation_service.annotate_image(request_payload)

    @app.post(
        "/v1/annotations/jobs",
        response_model=AnnotationJobAcceptedResponse,
        dependencies=[Depends(verify_internal_bearer_token)],
    )
    async def create_annotation_job() -> AnnotationJobAcceptedResponse:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={
                "code": "async_jobs_not_implemented",
                "message": "Async annotation jobs are reserved for Milestone 2 and are not implemented yet.",
            },
        )

    @app.get(
        "/v1/annotations/jobs/{job_id}",
        response_model=AnnotationJobAcceptedResponse,
        dependencies=[Depends(verify_internal_bearer_token)],
    )
    async def get_annotation_job(job_id: str) -> AnnotationJobAcceptedResponse:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={
                "code": "async_jobs_not_implemented",
                "message": f"Async annotation job status for '{job_id}' is reserved for Milestone 2.",
            },
        )

    return app
