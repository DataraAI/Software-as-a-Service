"""Storage adapters."""

from robotask_manipulator.storage.azure_blob import (
    AzureAnnotationStore,
    AzureBlobSource,
    AzureBlobStorageError,
    BlobReference,
    SourceBlobNotFoundError,
)

__all__ = [
    "AzureAnnotationStore",
    "AzureBlobSource",
    "AzureBlobStorageError",
    "BlobReference",
    "SourceBlobNotFoundError",
]
