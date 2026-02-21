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

    async def create_person(self, name: str, metadata: dict | None = None) -> Person:
        person = Person(name=name, metadata_=metadata)
        self.session.add(person)
        await self.session.flush()
        return person

    async def get_person(self, person_id: uuid.UUID) -> Person | None:
        result = await self.session.execute(
            select(Person).where(Person.id == person_id)
        )
        return result.scalar_one_or_none()

    async def delete_person(self, person_id: uuid.UUID) -> bool:
        person = await self.get_person(person_id)
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
    ) -> list[dict]:
        """Search for similar face embeddings using pgvector cosine distance.

        Returns list of dicts with person_id, person_name, similarity.
        """
        query_str = str(query_embedding)
        result = await self.session.execute(
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
                "person_id": row.person_id,
                "person_name": row.person_name,
                "similarity": float(row.similarity),
            }
            for row in result.fetchall()
        ]

    async def get_embeddings_count(self, person_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(FaceEmbedding).where(FaceEmbedding.person_id == person_id)
        )
        return len(result.scalars().all())
