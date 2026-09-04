"""Seed the development database with test data."""
from __future__ import annotations

import asyncio
import hashlib

from sqlalchemy import select

# Sentinel event the seeded person is filed under. A person MUST carry both
# api_key_id and event_id: every API path (search, list, get, delete-by-id,
# delete-by-event) filters on those values, so a row with NULL in either is
# unreachable and un-erasable through the API entirely.
DEV_EVENT_ID = "00000000-0000-0000-0000-000000000001"


async def seed() -> None:
    from src.db.models import APIKey, Person
    from src.db.session import get_session, init_db

    await init_db()

    async for session in get_session():
        # Create a development API key
        dev_key = "sk_dev_quickpitik_test_key_12345"
        key_hash = hashlib.sha256(dev_key.encode()).hexdigest()

        existing = await session.execute(
            select(APIKey).where(APIKey.key_hash == key_hash)
        )
        api_key = existing.scalar_one_or_none()
        if api_key is None:
            api_key = APIKey(
                key_hash=key_hash,
                name="Development Key",
                scopes=["*"],
                rate_tier="internal",
            )
            session.add(api_key)
            await session.flush()  # assign api_key.id for the person below
            print(f"Created dev API key: {dev_key}")
        else:
            print("Dev API key already exists")

        # Create a test person — scoped and idempotent. This was previously
        # Person(name="Test Person") with neither field set and no existence
        # check, so every seed run wrote a fresh orphan; five had accumulated
        # in the dev database before this was caught.
        existing_person = await session.execute(
            select(Person)
            .where(
                Person.api_key_id == str(api_key.id),
                Person.event_id == DEV_EVENT_ID,
                Person.name == "Test Person",
            )
            .limit(1)
        )
        # .first(), not .scalar_one_or_none(): the dev event legitimately holds
        # other persons, and this only needs to know whether one already exists.
        person = existing_person.scalars().first()
        if person is None:
            person = Person(
                name="Test Person",
                api_key_id=str(api_key.id),
                event_id=DEV_EVENT_ID,
            )
            session.add(person)
            await session.commit()
            print(f"Created test person: {person.id}")
        else:
            await session.commit()
            print(f"Test person already exists: {person.id}")

        print("Database seeded successfully")


if __name__ == "__main__":
    asyncio.run(seed())
