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
        """Search using proper vector literal binding and single-computation subquery."""
        query_vec = "[" + ",".join(str(f) for f in query_embedding) + "]"
        result = self.session.execute(
            text("""
                SELECT person_id, person_name, similarity
                FROM (
                    SELECT
                        fe.person_id,
                        p.name AS person_name,
                        1 - (fe.embedding <=> :query::vector) AS similarity
                    FROM face_embeddings fe
                    JOIN persons p ON p.id = fe.person_id
                ) sub
                WHERE similarity >= :threshold
                ORDER BY similarity DESC
                LIMIT :top_k
            """),
            {"query": query_vec, "threshold": threshold, "top_k": top_k},
        )
        return [
            {
                "person_id": str(row.person_id),
                "person_name": row.person_name,
                "similarity": float(row.similarity),
            }
            for row in result.fetchall()
        ]
