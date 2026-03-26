from __future__ import annotations

import uuid

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from src.db.models import FaceEmbedding, Person


class SyncFaceRepository:
    """Synchronous face repo for Celery batch face operations."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_person(
        self,
        name: str,
        api_key_id: str | None = None,
        event_id: str | None = None,
    ) -> Person:
        person = Person(name=name, api_key_id=api_key_id, event_id=event_id)
        self.session.add(person)
        self.session.flush()
        return person

    def store_embedding(
        self,
        person_id: uuid.UUID,
        embedding: list[float],
        source_image_hash: str,
        quality_score: float | None = None,
    ) -> FaceEmbedding | None:
        """Store a face embedding. Returns None if duplicate (same person + image hash)."""
        existing = self.session.execute(
            select(FaceEmbedding).where(
                FaceEmbedding.person_id == person_id,
                FaceEmbedding.source_image_hash == source_image_hash,
            )
        )
        if existing.scalar_one_or_none() is not None:
            return None

        face_emb = FaceEmbedding(
            person_id=person_id,
            embedding=embedding,
            source_image_hash=source_image_hash,
            quality_score=quality_score,
        )
        self.session.add(face_emb)
        self.session.flush()
        return face_emb

    def search_similar(
        self,
        query_embedding: list[float],
        threshold: float = 0.4,
        top_k: int = 10,
        api_key_id: str | None = None,
        event_id: str | None = None,
    ) -> list[dict]:
        """Search using proper vector literal binding and single-computation subquery."""
        query_vec = "[" + ",".join(str(f) for f in query_embedding) + "]"
        tenant_filter = ""
        params: dict = {"query": query_vec, "threshold": threshold, "top_k": top_k}
        if api_key_id is not None:
            tenant_filter += " AND p.api_key_id = :api_key_id"
            params["api_key_id"] = api_key_id
        if event_id is not None:
            tenant_filter += " AND p.event_id = :event_id"
            params["event_id"] = event_id
        result = self.session.execute(
            text(f"""
                SELECT person_id, person_name, similarity
                FROM (
                    SELECT
                        fe.person_id,
                        p.name AS person_name,
                        1 - (fe.embedding <=> :query::vector) AS similarity
                    FROM face_embeddings fe
                    JOIN persons p ON p.id = fe.person_id
                    WHERE 1=1 {tenant_filter}
                ) sub
                WHERE similarity >= :threshold
                ORDER BY similarity DESC
                LIMIT :top_k
            """),
            params,
        )
        return [
            {
                "person_id": str(row.person_id),
                "person_name": row.person_name,
                "similarity": min(1.0, max(0.0, float(row.similarity))),
            }
            for row in result.fetchall()
        ]
