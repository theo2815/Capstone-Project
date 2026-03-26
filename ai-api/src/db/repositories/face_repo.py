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
        self,
        name: str,
        metadata: dict | None = None,
        api_key_id: str | None = None,
        event_id: str | None = None,
    ) -> Person:
        person = Person(name=name, metadata_=metadata, api_key_id=api_key_id, event_id=event_id)
        self.session.add(person)
        await self.session.flush()
        return person

    async def get_person(
        self,
        person_id: uuid.UUID,
        api_key_id: str | None = None,
        event_id: str | None = None,
    ) -> Person | None:
        stmt = select(Person).where(Person.id == person_id)
        if api_key_id is not None:
            stmt = stmt.where(Person.api_key_id == api_key_id)
        if event_id is not None:
            stmt = stmt.where(Person.event_id == event_id)
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
    ) -> FaceEmbedding | None:
        """Store a face embedding. Returns None if duplicate (same person + image hash)."""
        existing = await self.session.execute(
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
        await self.session.flush()
        return face_emb

    async def search_similar(
        self,
        query_embedding: list[float],
        threshold: float = 0.4,
        top_k: int = 10,
        api_key_id: str | None = None,
        event_id: str | None = None,
    ) -> list[dict]:
        """Search for similar face embeddings using pgvector cosine distance.

        Uses proper vector literal binding instead of Python str() conversion.
        Computes cosine distance once via a subquery to avoid triple evaluation.
        Optionally filters by api_key_id and event_id for tenant/event isolation.
        """
        query_vec = "[" + ",".join(str(f) for f in query_embedding) + "]"

        tenant_filter = ""
        params: dict = {"query": query_vec, "threshold": threshold, "top_k": top_k}
        if api_key_id is not None:
            tenant_filter += " AND p.api_key_id = :api_key_id"
            params["api_key_id"] = api_key_id
        if event_id is not None:
            tenant_filter += " AND p.event_id = :event_id"
            params["event_id"] = event_id

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
                "similarity": min(1.0, max(0.0, float(row.similarity))),
            }
            for row in result.fetchall()
        ]

    async def batch_search_similar(
        self,
        embeddings: list[list[float]],
        threshold: float = 0.4,
        top_k: int = 10,
        api_key_id: str | None = None,
        event_id: str | None = None,
    ) -> list[list[dict]]:
        """Batch search for multiple embeddings.

        Delegates to the correct single-embedding ``search_similar()`` method
        per embedding. The previous UNION ALL approach silently dropped
        per-branch ORDER BY / LIMIT clauses in PostgreSQL.
        """
        if not embeddings:
            return []

        results = []
        for emb in embeddings:
            matches = await self.search_similar(
                query_embedding=emb,
                threshold=threshold,
                top_k=top_k,
                api_key_id=api_key_id,
                event_id=event_id,
            )
            results.append(matches)
        return results

    async def list_persons(
        self,
        api_key_id: str | None = None,
        event_id: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Person], int]:
        """List enrolled persons with pagination, filtered by api_key_id and event_id."""
        from sqlalchemy import func

        stmt = select(Person)
        count_stmt = select(func.count()).select_from(Person)

        if api_key_id is not None:
            stmt = stmt.where(Person.api_key_id == api_key_id)
            count_stmt = count_stmt.where(Person.api_key_id == api_key_id)
        if event_id is not None:
            stmt = stmt.where(Person.event_id == event_id)
            count_stmt = count_stmt.where(Person.event_id == event_id)

        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        stmt = stmt.order_by(Person.created_at.desc()).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        persons = list(result.scalars().all())

        return persons, total

    async def list_persons_with_counts(
        self,
        api_key_id: str | None = None,
        event_id: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[tuple[Person, int]], int]:
        """List persons with embedding counts in a single query (no N+1)."""
        from sqlalchemy import func

        # Count query
        count_stmt = select(func.count()).select_from(Person)
        if api_key_id is not None:
            count_stmt = count_stmt.where(Person.api_key_id == api_key_id)
        if event_id is not None:
            count_stmt = count_stmt.where(Person.event_id == event_id)
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        # Main query with LEFT JOIN + GROUP BY
        stmt = (
            select(Person, func.count(FaceEmbedding.id).label("emb_count"))
            .outerjoin(FaceEmbedding, Person.id == FaceEmbedding.person_id)
            .group_by(Person.id)
        )
        if api_key_id is not None:
            stmt = stmt.where(Person.api_key_id == api_key_id)
        if event_id is not None:
            stmt = stmt.where(Person.event_id == event_id)
        stmt = stmt.order_by(Person.created_at.desc()).offset(offset).limit(limit)

        result = await self.session.execute(stmt)
        rows = [(row[0], row[1]) for row in result.all()]

        return rows, total

    async def get_embeddings_count(self, person_id: uuid.UUID) -> int:
        from sqlalchemy import func

        result = await self.session.execute(
            select(func.count()).select_from(FaceEmbedding).where(
                FaceEmbedding.person_id == person_id
            )
        )
        return result.scalar_one()
