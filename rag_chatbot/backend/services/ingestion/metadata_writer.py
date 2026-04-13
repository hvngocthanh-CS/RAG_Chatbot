"""
Persist per-document ingestion outcomes to disk.

For every document we ingest we drop a small JSON sidecar in
`settings.PROCESSED_DIR`. It is the audit trail used by ops/debugging to
answer "did this document get indexed, when, and how many chunks?" without
having to query the vector store.
"""
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

from backend.config import settings


class IngestionMetadataWriter:
    """Write per-document success/failure metadata as JSON sidecars."""

    def __init__(self, processed_dir: str = settings.PROCESSED_DIR):
        self.processed_dir = processed_dir

    def write_success(
        self,
        metadata: Dict[str, Any],
        chunks_count: int,
        tables_count: int,
    ) -> None:
        self._write({
            "document_id": metadata["document_id"],
            "filename": metadata["filename"],
            "processed_at": _utc_now_iso(),
            "status": "completed",
            "chunks_count": chunks_count,
            "tables_count": tables_count,
        })

    def write_failure(self, metadata: Dict[str, Any], error: str) -> None:
        self._write({
            "document_id": metadata["document_id"],
            "filename": metadata["filename"],
            "processed_at": _utc_now_iso(),
            "status": "failed",
            "error": error,
        })

    def _write(self, payload: Dict[str, Any]) -> None:
        os.makedirs(self.processed_dir, exist_ok=True)
        path = os.path.join(
            self.processed_dir,
            f"{payload['document_id']}_meta.json",
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
