"""Event-isolation gate (FA-1) — verify no cross-event or cross-tenant leakage.

The suite proves the *boundary*: omitting `event_id` is rejected with 422. That
is a guard on the request, checked against mocks. This script proves the thing
the guard exists to protect — that the SQL filters actually partition real
embeddings in real pgvector, so a search scoped to event A can never surface a
face enrolled into event B, and one API key can never see, search, or delete
another key's data.

Why a script and not a pytest case: the assertions need a live Postgres with
pgvector, a loaded InsightFace model, and two authenticated tenants. The suite
has none of those (it mocks the embedder and has no test database), which is
exactly why this gap stayed open — see FA-1 in the vault.

Self-provisioning: the script creates two throwaway API keys directly in the
`api_keys` table, uses them over HTTP, and deletes them in a finally block. So
it needs no real credentials passed on the command line, and the two-tenant
check tests the genuine `api_key_id` filter rather than simulating it.

Faces are required: `tests/` has no face fixture and its generated images are
random noise, so the script sources real photos and FAILS LOUDLY if none of
them yield a detected face. A run that enrolled nothing would otherwise pass
every "returns zero matches" assertion vacuously.

Usage:
    python scripts/event_isolation_e2e.py \
        [--images-dir Training-Images/dataset/val] \
        [--url http://localhost:8000] [--candidates 25]

Requires a running ai-api with the face model loaded, reachable Postgres.
Exits 0 on PASS, 1 on FAIL, 2 on setup error.
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
import sys
import uuid
from pathlib import Path

# Running as `python scripts/event_isolation_e2e.py` puts scripts/ on sys.path,
# not the project root, so `src` would not import. Unlike blur_gate.py this
# script talks to the DB directly (to provision its own tenants).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import httpx
except ImportError:
    sys.exit("This script needs httpx:  pip install httpx  (it's already an ai-api test dep).")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Every check appends here; the gate is "no failures".
FAILURES: list[str] = []
CHECKS = 0


def check(ok: bool, label: str, detail: str = "") -> bool:
    """Record one assertion. Returns ok so callers can branch."""
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  PASS  {label}")
    else:
        msg = f"{label}{' — ' + detail if detail else ''}"
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)
    return ok


# ---------------------------------------------------------------------------
# Throwaway tenants
# ---------------------------------------------------------------------------

def provision_key(name: str) -> tuple[str, str]:
    """Insert an API key row and return (plaintext, key_id).

    Only the SHA-256 hash is stored, matching how gen_api_key.py + the auth
    middleware work. rate_tier=internal so a long run is not rate-limited.
    """
    from src.db.models import APIKey
    from src.db.sync_session import get_sync_session

    plaintext = f"sk_e2e_{secrets.token_hex(16)}"
    with get_sync_session() as session:
        row = APIKey(
            key_hash=hashlib.sha256(plaintext.encode()).hexdigest(),
            name=name,
            scopes=["*"],
            rate_tier="internal",
            active=True,
        )
        session.add(row)
        session.flush()
        return plaintext, str(row.id)


def revoke_keys(key_ids: list[str]) -> None:
    """Delete the throwaway rows. A stale Redis auth-cache entry may survive up
    to 5 minutes, which is harmless: the plaintext never left this process."""
    if not key_ids:
        return
    from sqlalchemy import delete

    from src.db.models import APIKey
    from src.db.sync_session import get_sync_session

    with get_sync_session() as session:
        session.execute(delete(APIKey).where(APIKey.id.in_([uuid.UUID(k) for k in key_ids])))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def hdr(key: str) -> dict:
    return {"X-API-Key": key}


def enroll(client, base, key, img, event_id, person_name):
    return client.post(
        f"{base}/api/v1/faces/enroll",
        headers=hdr(key),
        files={"file": ("photo.jpg", img, "image/jpeg")},
        data={"person_name": person_name, "event_id": event_id},
    )


def search(client, base, key, img, event_id):
    params = {"threshold": 0.4, "top_k": 50}
    if event_id is not None:
        params["event_id"] = event_id
    return client.post(
        f"{base}/api/v1/faces/search",
        headers=hdr(key), params=params,
        files={"file": ("probe.jpg", img, "image/jpeg")},
    )


def matched_person_ids(resp) -> set[str]:
    return {m["person_id"] for m in resp.json()["data"]["matches"]}


def list_persons(client, base, key, event_id) -> list[dict]:
    r = client.get(
        f"{base}/api/v1/faces/persons",
        headers=hdr(key), params={"event_id": event_id, "limit": 200},
    )
    return r.json().get("data", {}).get("persons", []) if r.status_code == 200 else []


def delete_by_event(client, base, key, event_id) -> int:
    r = client.delete(
        f"{base}/api/v1/faces/persons", headers=hdr(key), params={"event_id": event_id}
    )
    if r.status_code != 200:
        return -1
    return r.json().get("data", {}).get("deleted", -1)


# ---------------------------------------------------------------------------
# Fixture selection
# ---------------------------------------------------------------------------

def find_face_image(
    client, base, key, images_dir: Path, candidates: int
) -> tuple[bytes, str] | None:
    """Return (bytes, filename) for the first image ai-api reports a face in.

    Detection is the ground truth here, not the file name: an image that only
    LOOKS like a portrait proves nothing if the model finds no face in it.
    """
    paths = [
        p for p in sorted(images_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not paths:
        return None
    step = max(1, len(paths) // candidates)
    for path in paths[::step][:candidates]:
        data = path.read_bytes()
        try:
            r = client.post(
                f"{base}/api/v1/faces/detect",
                headers=hdr(key),
                files={"file": (path.name, data, "image/jpeg")},
            )
        except httpx.HTTPError:
            continue
        if r.status_code == 200 and r.json()["data"]["faces_detected"] > 0:
            return data, path.name
    return None


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Event + tenant isolation gate (FA-1).")
    ap.add_argument("--url", default="http://localhost:8000", help="ai-api base URL.")
    ap.add_argument("--images-dir", type=Path, default=Path("Training-Images/dataset/val"),
                    help="Directory to source a real face-bearing photo from.")
    ap.add_argument("--candidates", type=int, default=25,
                    help="How many images to try before giving up on finding a face.")
    args = ap.parse_args()

    base = args.url.rstrip("/")

    if not args.images_dir.is_dir():
        print(f"Not a directory: {args.images_dir}", file=sys.stderr)
        return 2

    # Two tenants + three events. Event C is never enrolled into — it is the
    # "scope that should see nothing" control.
    event_a = f"e2e-a-{uuid.uuid4()}"
    event_b = f"e2e-b-{uuid.uuid4()}"
    event_c = f"e2e-c-{uuid.uuid4()}"

    key_ids: list[str] = []
    try:
        key1, key1_id = provision_key("e2e-isolation-tenant-1")
        key_ids.append(key1_id)
        key2, key2_id = provision_key("e2e-isolation-tenant-2")
        key_ids.append(key2_id)
    except Exception as e:
        print(f"Could not provision throwaway API keys: {e}", file=sys.stderr)
        return 2

    print(f"Event isolation gate against {base}")
    print(f"  tenant 1: {key1_id}")
    print(f"  tenant 2: {key2_id}")
    print(f"  events:   A={event_a[:14]}…  B={event_b[:14]}…  C={event_c[:14]}… (control)")

    try:
        with httpx.Client(timeout=120.0) as client:
            try:
                ready = client.get(f"{base}/api/v1/health/ready", headers=hdr(key1))
            except httpx.HTTPError as e:
                print(f"\nai-api unreachable at {base}: {e}", file=sys.stderr)
                return 2
            if ready.status_code != 200:
                print(f"\nai-api not ready: {ready.status_code} {ready.text[:200]}",
                      file=sys.stderr)
                return 2

            print("\nselecting a real face-bearing photo…")
            found = find_face_image(client, base, key1, args.images_dir, args.candidates)
            if found is None:
                print(
                    f"\nFAIL: no face detected in any of {args.candidates} candidate images "
                    f"under {args.images_dir}. Refusing to run — every isolation assertion "
                    f"below would pass vacuously on an empty embedding set.",
                    file=sys.stderr,
                )
                return 1
            img, img_name = found
            print(f"  using {img_name}")

            # -- enroll the SAME photo into two events, under one tenant -----
            print("\nenroll (tenant 1):")
            ra = enroll(client, base, key1, img, event_a, "runner-in-event-a")
            rb = enroll(client, base, key1, img, event_b, "runner-in-event-b")
            ok_a = ra.status_code == 200 and ra.json()["success"]
            ok_b = rb.status_code == 200 and rb.json()["success"]
            if not check(ok_a and ok_b, "enrolled into events A and B",
                         f"A={ra.status_code}:{ra.text[:100]} B={rb.status_code}:{rb.text[:100]}"):
                return 1
            person_a = ra.json()["data"]["person_id"]
            person_b = rb.json()["data"]["person_id"]
            check(person_a != person_b, "same photo in two events -> two distinct persons")

            # -- event isolation --------------------------------------------
            print("\nevent isolation (tenant 1):")
            sa = matched_person_ids(search(client, base, key1, img, event_a))
            check(person_a in sa, "search(event A) finds the event-A person")
            check(person_b not in sa, "search(event A) does NOT leak the event-B person",
                  f"matched {sorted(sa)}")

            sb = matched_person_ids(search(client, base, key1, img, event_b))
            check(person_b in sb, "search(event B) finds the event-B person")
            check(person_a not in sb, "search(event B) does NOT leak the event-A person",
                  f"matched {sorted(sb)}")

            sc = matched_person_ids(search(client, base, key1, img, event_c))
            check(not sc, "search(event C, never enrolled) returns nothing",
                  f"matched {sorted(sc)}")

            # -- fail-closed boundary, on the live server --------------------
            print("\nfail-closed boundary:")
            check(search(client, base, key1, img, None).status_code == 422,
                  "search without event_id -> 422")
            no_scope = client.post(
                f"{base}/api/v1/faces/enroll", headers=hdr(key1),
                files={"file": ("photo.jpg", img, "image/jpeg")},
                data={"person_name": "orphan-maker"},
            )
            check(no_scope.status_code == 422, "enroll without event_id -> 422",
                  f"got {no_scope.status_code}")

            # -- tenant isolation --------------------------------------------
            # The part mocked tests cannot reach: tenant 2 holds a valid key and
            # the correct event id, and must still see nothing.
            print("\ntenant isolation (tenant 2, same event id):")
            s2 = matched_person_ids(search(client, base, key2, img, event_a))
            check(not s2, "search(event A) as tenant 2 returns nothing", f"matched {sorted(s2)}")
            check(not list_persons(client, base, key2, event_a),
                  "list persons(event A) as tenant 2 returns nothing")
            check(delete_by_event(client, base, key2, event_a) == 0,
                  "delete-by-event(A) as tenant 2 deletes nothing")
            check(len(list_persons(client, base, key1, event_a)) == 1,
                  "tenant 1's event-A person survived tenant 2's delete attempt")

            # -- GDPR cascade + cleanup --------------------------------------
            print("\nGDPR delete-by-event (tenant 1):")
            check(delete_by_event(client, base, key1, event_a) == 1, "deleted event A's person")
            check(delete_by_event(client, base, key1, event_b) == 1, "deleted event B's person")
            check(not list_persons(client, base, key1, event_a), "event A now empty")
            check(not list_persons(client, base, key1, event_b), "event B now empty")
            check(not matched_person_ids(search(client, base, key1, img, event_a)),
                  "search(event A) after erasure returns nothing")
    finally:
        revoke_keys(key_ids)

    print("\n" + "=" * 68)
    print("EVENT + TENANT ISOLATION GATE (FA-1)")
    print("=" * 68)
    print(f"checks run:   {CHECKS}")
    print(f"failures:     {len(FAILURES)}")
    for f in FAILURES:
        print(f"  - {f}")
    verdict = "PASS" if not FAILURES else "FAIL"
    print(f"\n{verdict}")
    print("=" * 68)
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
