from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import FaceEmbedding, Person


class FaceRepository:
    """Repository for face embedding CRUD and vector similarity search."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_person(
        self, name: str, metadata: dict | None = None, api_key_id: str | None = None
    ) -> Person:
        person = Person(name=name, metadata_=metadata, api_key_id=api_key_id)
        self.session.add(person)
        await self.session.flush()
        return person

    async def get_person(
        self, person_id: uuid.UUID, api_key_id: str | None = None
    ) -> Person | None:
        stmt = select(Person).where(Person.id == person_id)
        if api_key_id is not None:
            stmt = stmt.where(Person.api_key_id == api_key_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def delete_person(
        self, person_id: uuid.UUID, api_key_id: str | None = None
    ) -> bool:
        person = await self.get_person(person_id, api_key_id=api_key_id)
        if person is None:
            return False
        await self.session.delete(person)
        await self.session.flush()
        return True

    async def store_embedding(
        self,
        person_id: uuid.UUID,
        embedding: list[float],
        source_image_hash: str,
        quality_score: float | None = None,
    ) -> FaceEmbedding:
        face_emb = FaceEmbedding(
            person_id=person_id,
            embedding=embedding,
            source_image_hash=source_image_hash,
            quality_score=quality_score,
        )
        self.session.add(face_emb)
        await self.session.flush()
        return face_emb

    async def search_similar(
        self,
        query_embedding: list[float],
        threshold: float = 0.4,
        top_k: int = 10,
        api_key_id: str | None = None,
    ) -> list[dict]:
        """Search for similar face embeddings using pgvector cosine distance.

        Uses proper vector literal binding instead of Python str() conversion.
        Computes cosine distance once via a subquery to avoid triple evaluation.
        Optionally filters by api_key_id for tenant isolation.
        """
        query_vec = "[" + ",".join(str(f) for f in query_embedding) + "]"

        tenant_filter = ""
        params: dict = {"query": query_vec, "threshold": threshold, "top_k": top_k}
        if api_key_id is not None:
            tenant_filter = "AND p.api_key_id = :api_key_id"
            params["api_key_id"] = api_key_id

        result = await self.session.execute(
            text(f"""
                SELECT person_id, person_name, similarity
                FROM (
                    SELECT
                        fe.person_id,
                        p.name AS person_name,
                        1 - (fe.embedding <=> :query::vector) AS similarity
                    FROM face_embeddings fe
                    JOIN persons p ON p.id = fe.person_id
                    WHERE 1 = 1 {tenant_filter}
                ) sub
                WHERE similarity >= :threshold
                ORDER BY similarity DESC
                LIMIT :top_k
            """),
            params,
        )
        return [
            {
                "person_id": row.person_id,
                "person_name": row.person_name,
                "similarity": float(row.similarity),
            }
            for row in result.fetchall()
        ]

    async def batch_search_similar(
        self,
        embeddings: list[list[float]],
        threshold: float = 0.4,
        top_k: int = 10,
        api_key_id: str | None = None,
    ) -> list[list[dict]]:
        """Batch search for multiple embeddings in a single query.

        Uses UNION ALL to combine all embedding queries into one database
        round-trip instead of N sequential queries.
        """
        if not embeddings:
            return []

        # Build a UNION ALL query for all embeddings
        union_parts = []
        params: dict = {"threshold": threshold, "top_k": top_k}

        tenant_filter = ""
        if api_key_id is not None:
            tenant_filter = "AND p.api_key_id = :api_key_id"
            params["api_key_id"] = api_key_id

        for i, emb in enumerate(embeddings):
            vec = "[" + ",".join(str(f) for f in emb) + "]"
            param_key = f"q{i}"
            params[param_key] = vec
            union_parts.append(f"""
                SELECT
                    {i} AS query_idx,
                    sub.person_id,
                    sub.person_name,
                    sub.similarity
                FROM (
                    SELECT
                        fe.person_id,
                        p.name AS person_name,
                        1 - (fe.embedding <=> :{param_key}::vector) AS similarity
                    FROM face_embeddings fe
                    JOIN persons p ON p.id = fe.person_id
                    WHERE 1 = 1 {tenant_filter}
                ) sub
                WHERE sub.similarity >= :threshold
                ORDER BY sub.similarity DESC
                LIMIT :top_k
            """)

        full_query = " UNION ALL ".join(union_parts) + " ORDER BY query_idx, similarity DESC"

        result = await self.session.execute(text(full_query), params)
        rows = result.fetchall()

        # Group results by query_idx
        results_by_idx: dict[int, list[dict]] = {i: [] for i in range(len(embeddings))}
        for row in rows:
            results_by_idx[row.query_idx].append({
                "person_id": row.person_id,
                "person_name": row.person_name,
                "similarity": float(row.similarity),
            })

        return [results_by_idx[i] for i in range(len(embeddings))]

    async def get_embeddings_count(self, person_id: uuid.UUID) -> int:
        from sqlalchemy import func

        result = await self.session.execute(
            select(func.count()).select_from(FaceEmbedding).where(
                FaceEmbedding.person_id == person_id
            )
        )
        return result.scalar_one()
