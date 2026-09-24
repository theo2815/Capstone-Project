"""Private batch image staging with opaque keys shared by API and workers."""
from __future__ import annotations

import os
import re
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor

from src.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
_KEY = re.compile(r"^jobs/[0-9a-f-]{36}/[0-9]{5}\.bin$")


def _client():
    import boto3
    from botocore.config import Config

    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.AWS_ENDPOINT_URL or None,
        region_name=settings.AWS_DEFAULT_REGION,
        config=Config(s3={"addressing_style": settings.AWS_S3_URL_STYLE}),
    )


def _key(job_id: str, index: int) -> str:
    return f"jobs/{uuid.UUID(job_id)}/{index:05d}.bin"


def _local_path(key: str) -> str:
    if not _KEY.fullmatch(key):
        raise ValueError("Invalid blob key")
    return os.path.join(get_settings().BLOB_STORE_PATH, *key.split("/"))


def store_batch(job_id: str, raw_bytes_list: list[bytes]) -> list[str]:
    """Upload every image before publishing a task; remove partial batches on error."""
    settings = get_settings()
    keys = [_key(job_id, i) for i in range(len(raw_bytes_list))]
    bucket = settings.AWS_S3_BUCKET_NAME
    client = _client() if bucket else None

    def write(item: tuple[str, bytes]) -> None:
        key, raw = item
        if len(raw) > settings.MAX_FILE_SIZE:
            raise ValueError("Image blob exceeds maximum size")
        if client:
            client.put_object(Bucket=bucket, Key=key, Body=raw)
        else:
            path = _local_path(key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temp = path + ".tmp"
            with open(temp, "wb") as f:
                f.write(raw)
            os.replace(temp, path)

    try:
        with ThreadPoolExecutor(max_workers=min(8, len(keys)) or 1) as pool:
            list(pool.map(write, zip(keys, raw_bytes_list, strict=True)))
    except Exception:
        cleanup_batch(job_id)
        raise
    logger.debug("Blob batch stored", job_id=job_id, count=len(keys))
    return keys


def load_blob(key: str) -> bytes:
    """Load at most one validated upload; never accept an arbitrary path or URL."""
    if not _KEY.fullmatch(key):
        raise ValueError("Invalid blob key")
    settings = get_settings()
    limit = settings.MAX_FILE_SIZE
    if settings.AWS_S3_BUCKET_NAME:
        body = _client().get_object(Bucket=settings.AWS_S3_BUCKET_NAME, Key=key)["Body"]
        try:
            data = body.read(limit + 1)
        finally:
            body.close()
    else:
        with open(_local_path(key), "rb") as f:
            data = f.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Image blob exceeds maximum size")
    return data


def cleanup_batch(job_id: str) -> None:
    """Delete one job's private inputs; errors surface for scheduled retry."""
    job_id = str(uuid.UUID(job_id))
    prefix = f"jobs/{job_id}/"
    settings = get_settings()
    if settings.AWS_S3_BUCKET_NAME:
        client = _client()
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=settings.AWS_S3_BUCKET_NAME, Prefix=prefix
        ):
            for obj in page.get("Contents", []):
                client.delete_object(Bucket=settings.AWS_S3_BUCKET_NAME, Key=obj["Key"])
    else:
        path = os.path.join(settings.BLOB_STORE_PATH, "jobs", job_id)
        if os.path.isdir(path):
            shutil.rmtree(path)


def list_batches() -> list[tuple[str, float]]:
    """Return staged job IDs and newest object mtime for the retention reaper."""
    settings = get_settings()
    if settings.AWS_S3_BUCKET_NAME:
        # ponytail: full prefix scan; use inventory if retained object count grows large.
        batches: dict[str, float] = {}
        for page in _client().get_paginator("list_objects_v2").paginate(
            Bucket=settings.AWS_S3_BUCKET_NAME, Prefix="jobs/"
        ):
            for obj in page.get("Contents", []):
                parts = obj["Key"].split("/")
                if len(parts) == 3 and _KEY.fullmatch(obj["Key"]):
                    stamp = obj["LastModified"].timestamp()
                    batches[parts[1]] = max(batches.get(parts[1], 0), stamp)
        return list(batches.items())
    root = os.path.join(settings.BLOB_STORE_PATH, "jobs")
    if not os.path.isdir(root):
        return []
    return [(entry.name, entry.stat().st_mtime) for entry in os.scandir(root) if entry.is_dir()]
