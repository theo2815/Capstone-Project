from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session


class SyncFaceRepository:
    """Synchronous face repo for Celery batch face search."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def search_similar(
        self,
        query_embedding: list[float],
        threshold: float = 0.4,
        top_k: int = 10,
    ) -> list[dict]:
        query_str = str(query_embedding)
        result = self.session.execute(
            text("""
                SELECT
                    fe.person_id,
                    p.name AS person_name,
                    1 - (fe.embedding <=> :query) AS similarity
                FROM face_embeddings fe
                JOIN persons p ON p.id = fe.person_id
                WHERE 1 - (fe.embedding <=> :query) >= :threshold
                ORDER BY fe.embedding <=> :query
                LIMIT :top_k
            """),
            {"query": query_str, "threshold": threshold, "top_k": top_k},
        )
        return [
            {
                "person_id": str(row.person_id),
                "person_name": row.person_name,
                "similarity": float(row.similarity),
            }
            for row in result.fetchall()
        ]
