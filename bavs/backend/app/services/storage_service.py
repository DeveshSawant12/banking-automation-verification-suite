"""
Storage Service — Cloudflare R2 (S3-compatible) client wrapper.

R2 was locked as the object storage backend for this project (uploaded
documents, selfies, and now explainability heatmap images). R2 exposes a
genuine S3-compatible API, so this uses boto3's real 's3' client pointed
at R2's endpoint — not a fabricated SDK. Credentials and endpoint come
from app.config.Settings (R2_ACCOUNT_ID, R2_ACCESS_KEY_ID,
R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_ENDPOINT_URL), already declared
in Module 1's config.py.

This module is being introduced now (Module 8) because it is the first
module that needs to WRITE a generated artifact (a heatmap image) back to
storage, rather than only reading previously-uploaded documents. Prior
modules (1-7) only needed document bytes passed in directly by the
caller; the actual "fetch the uploaded document from R2" responsibility
belongs to the orchestrator/API layer (not yet built), which is why no
prior module needed this file. Introducing it here, scoped to what
Module 8 actually requires (upload bytes, get a retrievable key), rather
than building out the full read-path prematurely with invented usage
patterns.
"""

from __future__ import annotations

import logging
import uuid

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

_r2_client = None


class StorageServiceError(Exception):
    """Raised when an R2 storage operation fails."""


def _get_r2_client():
    """
    Lazily initialize and return the singleton boto3 S3 client configured
    for Cloudflare R2's S3-compatible endpoint.

    Raises:
        StorageServiceError: if required R2 settings are not configured
            (empty string defaults from Settings) — refuses to silently
            attempt a connection with blank credentials, which would
            produce a confusing botocore error far from the real cause.
    """
    global _r2_client

    if _r2_client is not None:
        return _r2_client

    missing = [
        name
        for name, value in [
            ("R2_ACCOUNT_ID", settings.R2_ACCOUNT_ID),
            ("R2_ACCESS_KEY_ID", settings.R2_ACCESS_KEY_ID),
            ("R2_SECRET_ACCESS_KEY", settings.R2_SECRET_ACCESS_KEY),
            ("R2_BUCKET_NAME", settings.R2_BUCKET_NAME),
            ("R2_ENDPOINT_URL", settings.R2_ENDPOINT_URL),
        ]
        if not value
    ]
    if missing:
        raise StorageServiceError(
            f"Cannot initialize R2 client: missing required settings: "
            f"{', '.join(missing)}. Set these environment variables "
            f"(see backend/.env.example)."
        )

    _r2_client = boto3.client(
        "s3",
        endpoint_url=settings.R2_ENDPOINT_URL,
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        config=BotoConfig(signature_version="s3v4"),
        region_name="auto",  # R2 convention for the region parameter
    )
    return _r2_client


def upload_bytes(
    data: bytes, key_prefix: str, content_type: str = "application/octet-stream"
) -> str:
    """
    Upload raw bytes to R2 under a generated key, returning the object
    key (not a URL — callers store this key and construct access URLs
    separately, since whether objects are public/private is a deployment
    concern, not this function's responsibility).

    Args:
        data: raw bytes to upload
        key_prefix: logical path prefix, e.g. "explainability/gradcam"
            (a UUID-based filename is appended automatically so callers
            never need to handle collision avoidance themselves)
        content_type: MIME type for the stored object

    Returns:
        The full R2 object key, e.g.
        "explainability/gradcam/3f9e2b1a-....png"

    Raises:
        StorageServiceError: on any R2/boto3 failure.
    """
    client = _get_r2_client()
    object_key = f"{key_prefix.rstrip('/')}/{uuid.uuid4()}"

    try:
        client.put_object(
            Bucket=settings.R2_BUCKET_NAME,
            Key=object_key,
            Body=data,
            ContentType=content_type,
        )
    except ClientError as exc:
        raise StorageServiceError(
            f"Failed to upload object to R2 at key {object_key}: {exc}"
        ) from exc

    logger.info("Uploaded object to R2: %s (%d bytes)", object_key, len(data))
    return object_key


def download_bytes(object_key: str) -> bytes:
    """
    Download raw bytes for a given R2 object key.

    Raises:
        StorageServiceError: if the object does not exist or the
            download otherwise fails.
    """
    client = _get_r2_client()

    try:
        response = client.get_object(Bucket=settings.R2_BUCKET_NAME, Key=object_key)
        return response["Body"].read()
    except ClientError as exc:
        raise StorageServiceError(
            f"Failed to download object from R2 at key {object_key}: {exc}"
        ) from exc


def delete_object(object_key: str) -> None:
    """
    Delete an object from R2. Raises StorageServiceError on failure.
    Does NOT raise if the object never existed (idempotent delete,
    consistent with S3-compatible delete semantics).
    """
    client = _get_r2_client()

    try:
        client.delete_object(Bucket=settings.R2_BUCKET_NAME, Key=object_key)
    except ClientError as exc:
        raise StorageServiceError(
            f"Failed to delete object from R2 at key {object_key}: {exc}"
        ) from exc

    logger.info("Deleted object from R2: %s", object_key)
