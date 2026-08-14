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

    def get_person(
        self,
        person_id: uuid.UUID,
        api_key_id: str | None = None,
        event_id: str | None = None,
    ) -> Person | None:
        """Fetch one person, optionally scoped by tenant and event.

        Mirrors FaceRepository.get_person. The enroll worker needs this rather
        than a bare ``select(Person).where(Person.id == pid)``: an existence-only
        check lets one tenant enroll faces into another tenant's person by
        passing its person_id.
        """
        stmt = select(Person).where(Person.id == person_id)
        if api_key_id is not None:
            stmt = stmt.where(Person.api_key_id == api_key_id)
        if event_id is not None:
            stmt = stmt.where(Person.event_id == event_id)
        return self.session.execute(stmt).scalar_one_or_none()

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

    def bulk_store_embeddings(
        self,
        person_id: uuid.UUID,
        faces: list[dict],
        min_conf: float = 0.0,
    ) -> tuple[int, int]:
        """Bulk-insert face embeddings using ON CONFLICT DO NOTHING.

        Uses the unique index ix_face_embeddings_person_hash on
        (person_id, source_image_hash) to skip duplicates without a prior SELECT.
        Reduces N SELECT+INSERT pairs down to a single INSERT statement per call.

        Args:
            person_id: UUID of the person to enroll.
            faces: List of face dicts with keys: embedding, source_image_hash,
                   and bbox (containing "confidence").
            min_conf: Skip faces whose bbox confidence is below this threshold.

        Returns:
            (stored_count, skipped_count) tuple.
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        qualifying = [
            f for f in faces if f.get("bbox", {}).get("confidence", 0.0) >= min_conf
        ]
        skipped_low_conf = len(faces) - len(qualifying)

        if not qualifying:
            return 0, skipped_low_conf

        rows = [
            {
                "id": uuid.uuid4(),
                "person_id": person_id,
                "embedding": f["embedding"],
                "source_image_hash": f["source_image_hash"],
                "quality_score": f["bbox"]["confidence"],
            }
            for f in qualifying
        ]

        stmt = (
            pg_insert(FaceEmbedding)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["person_id", "source_image_hash"])
        )
        result = self.session.execute(stmt)
        stored = result.rowcount if result.rowcount is not None and result.rowcount >= 0 else 0
        skipped_dup = len(rows) - stored

        return stored, skipped_low_conf + skipped_dup

    def batch_search_similar(
        self,
        embeddings: list[list[float]],
        threshold: float = 0.4,
        top_k: int = 10,
        api_key_id: str | None = None,
        event_id: str | None = None,
    ) -> list[list[dict]]:
        """Batch vector similarity search using a single LATERAL JOIN query.

        Replaces N separate ``search_similar()`` calls with one round-trip.
        Constructs a VALUES clause of (query_idx, vector_literal) rows, then
        uses a LATERAL subquery to apply the HNSW index once per query vector.

        Falls back to the per-query loop when only one embedding is provided,
        since the overhead of building the VALUES clause is not worth it.

        Args:
            embeddings: List of 512-dim face embedding vectors.
            threshold: Minimum cosine similarity to return.
            top_k: Max results per query embedding.
            api_key_id: Tenant isolation filter.
            event_id: Event isolation filter.

        Returns:
            List of match lists, one per input embedding (same order).
        """
        if not embeddings:
            return []
        if len(embeddings) == 1:
            return [self.search_similar(embeddings[0], threshold, top_k, api_key_id, event_id)]

        tenant_filter = ""
        params: dict = {"threshold": threshold, "top_k": top_k}
        if api_key_id is not None:
            tenant_filter += " AND p.api_key_id = :api_key_id"
            params["api_key_id"] = api_key_id
        if event_id is not None:
            tenant_filter += " AND p.event_id = :event_id"
            params["event_id"] = event_id

        # Build VALUES clause: each row is (integer_index, vector_literal::vector)
        values_rows = []
        for i, emb in enumerate(embeddings):
            vec_str = "[" + ",".join(str(f) for f in emb) + "]"
            values_rows.append(f"({i}, '{vec_str}'::vector)")
        values_clause = ", ".join(values_rows)

        sql = f"""
            SELECT q.query_idx, sub.person_id, sub.person_name, sub.similarity
            FROM (VALUES {values_clause}) AS q(query_idx, query_vec)
            LEFT JOIN LATERAL (
                SELECT
                    fe.person_id,
                    p.name AS person_name,
                    1 - (fe.embedding <=> q.query_vec) AS similarity
                FROM face_embeddings fe
                JOIN persons p ON p.id = fe.person_id
                WHERE 1 = 1 {tenant_filter}
                  AND 1 - (fe.embedding <=> q.query_vec) >= :threshold
                ORDER BY fe.embedding <=> q.query_vec
                LIMIT :top_k
            ) sub ON TRUE
            ORDER BY q.query_idx, sub.similarity DESC
        """
        rows = self.session.execute(text(sql), params).fetchall()

        grouped: list[list[dict]] = [[] for _ in embeddings]
        for row in rows:
            if row.person_id is not None:
                grouped[row.query_idx].append({
                    "person_id": str(row.person_id),
                    "person_name": row.person_name,
                    "similarity": min(1.0, max(0.0, float(row.similarity))),
                })
        return grouped

    def search_similar(
        self,
        query_embedding: list[float],
        threshold: float = 0.4,
        top_k: int = 10,
        api_key_id: str | None = None,
        event_id: str | None = None,
    ) -> list[dict]:
        """Search using proper vector literal binding and single-computation subquery.

        Uses ``CAST(:query AS vector)``, never ``:query::vector``. SQLAlchemy's
        bind-parameter regex for text() requires the char after the name not be
        a ':', so ``:query::vector`` backtracks and binds a parameter named
        ``quer`` — the real ``query`` value is never substituted and Postgres
        receives a literal ':' (``syntax error at or near ":"``). The async repo
        was fixed for this on 2026-05-20; this sync copy was missed, which broke
        every single-face image on /faces/search/batch and /search/mega (the
        len(embeddings) == 1 fallback below routes here).
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
        result = self.session.execute(
            text(f"""
                SELECT person_id, person_name, similarity
                FROM (
                    SELECT
                        fe.person_id,
                        p.name AS person_name,
                        1 - (fe.embedding <=> CAST(:query AS vector)) AS similarity
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
