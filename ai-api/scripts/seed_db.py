"""Seed the development database with test data."""
from __future__ import annotations

import asyncio
import hashlib


async def seed() -> None:
    from src.db.models import APIKey, Person
    from src.db.session import get_session, init_db

    await init_db()

    async for session in get_session():
        # Create a development API key
        dev_key = "sk_dev_eventai_test_key_12345"
        key_hash = hashlib.sha256(dev_key.encode()).hexdigest()

        existing = await session.execute(
            __import__("sqlalchemy").select(APIKey).where(APIKey.key_hash == key_hash)
        )
        if existing.scalar_one_or_none() is None:
            api_key = APIKey(
                key_hash=key_hash,
                name="Development Key",
                scopes=["*"],
                rate_tier="internal",
            )
            session.add(api_key)
            print(f"Created dev API key: {dev_key}")
        else:
            print("Dev API key already exists")

        # Create a test person
        person = Person(name="Test Person")
        session.add(person)

        await session.commit()
        print(f"Created test person: {person.id}")
        print("Database seeded successfully")


if __name__ == "__main__":
    asyncio.run(seed())
