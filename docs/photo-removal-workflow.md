# Runner photo removal: operations and release

How admins run the photo-removal workflow and how to release or roll it back. The API, state machine and data model are in [`BACKEND_ARCHITECTURE_ANALYSIS.md` section 7.7](BACKEND_ARCHITECTURE_ANALYSIS.md#77-admin-operations--governance); this page does not repeat them. Face-erasure provider rules: `ai-api/docs/integration-contracts.md` (Confirmed per-photo face erasure).

## Admin operations

- **Intake is through support only.** There is no public form. An admin opens the case from what support received.
- **Overdue reviews.** Photographers get a response window (`app.photo-removal.photographer-response-window` in `backend/src/main/resources/application.yml`, bound by `PhotoRemovalProperties`). After it passes, an admin may decide without the owner's answer. Deciding earlier, or with no reachable owner, needs an urgent reason.
- **Cleanup retries.** Erasure that cannot be confirmed (storage, face provider, AI disabled or misconfigured) stays pending and is retried by a sweep. Tune it with `PHOTO_REMOVAL_CLEANUP_INTERVAL` (how often it runs) and `PHOTO_REMOVAL_CLEANUP_BACKOFF` (wait after a failure). A case stays RESOLVING until every authorized erasure is confirmed. Fix the cause (credentials, `AI_API_ENABLED`, permissions) rather than clearing state by hand.
- **Retained-restricted items.** RETAIN_RESTRICTED needs a reason and a future review date. The photo stays unavailable and the item comes back to admins on that date for a new decision. It is not a deletion.
- **Deciding while RESOLVING.** An admin may decide a RETAINED_RESTRICTED item while the case is RESOLVING (another photo's cleanup is still retrying). Any other item state is a 409, and so is any decision on a case with a cancel requested. Rule: `PhotoRemovalService.decide()`. MORE_EVIDENCE keeps the item VERIFIED and reopens the case to OPEN (`reconcile`); evidence already purged during RESOLVING is not restored, so the requester must supply it again.
- **Guest buyers.** Guests and deleted accounts cannot be notified by the inbox. Admin case detail lists the guest notices; support sends the message and records the update there.
- **Outside our control.** Copies a buyer already downloaded, screenshots, and anything shared elsewhere cannot be recalled. Tell requesters this.

## Media and retention facts

What deletion covers. The shared eraser `PhotoErasureService` (`backend/src/main/kotlin/com/quickpitik/service/photos/`) deletes the photo's `s3Key`, `thumbnailS3Key`, `watermarkS3Key`, the fixed `events/{eventId}/photos/{id}/watermark.jpg` key, the keys of its `photo_work_outputs` (`erasureWorkKeys`), and the staging and original keys of its upload receipts. A key still used by another photo is skipped (`SHARED_REFERENCE`). A key counts as `ABSENT` only after `storage.exists` confirms it; otherwise it retries. Face records go through `FaceBibProvider.erasePhotoFaces`. The case selfie is removed by `PhotoRemovalEvidenceService`. The photo row can remain as a tombstone: `PhotoRepository.minimizeErasedPhoto` clears the blob keys and descriptive fields but keeps the ID and the attribution and finance links. So a backend database backup holds case rows and tombstones, not image bytes (bytes live only in object storage).

What deletion does not cover:

- **Issued signed URLs stay valid until they expire.** A database flag stops new delivery; it does not revoke a URL already handed out. Lifetimes are the defaults in `StorageProperties.PresignedTtl`: thumbnail, cover, watermark, avatar and selfie 1 h (previews have no TTL field and use the thumbnail/watermark TTL, 1 h); runner download 15 min; photographer download 5 min. `S3StorageService` reuses a signature for at most min(5 min, ttl/4); this does not extend expiry. Immediate revocation would need an approved delivery-control extension. This release does not claim it.
- **Completed downloads, screenshots and caches** cannot be recalled.
- **Open release gates, not yet verified:** bucket object versioning (a delete marker is not a purge of all versions), R2/S3 backups, CDN caches, Railway Postgres backups and their retention, and, when `AI_PROVIDER` is `ai-api` (the default), backups of the ai-api face-embedding database (pgvector, `pgdata` volume in `ai-api/docker-compose.prod.yml`). Rekognition collections (`AI_PROVIDER=rekognition`) are provider-managed and not covered by QuickPitik backups. If any holds copies, keep the cleanup visibly pending with a documented retention reason and expiry until purged.

## Release order

1. **Migrations.** The removal migrations are additive: V68 to V76 and V80 under `backend/src/main/resources/db/migration`. V81 to V84 belong to other lanes (V77 to V79 were reserved but are not present). Flyway applies them at backend boot and they cannot be undone. Rehearse on a throwaway database first: a clean install, and an upgrade from the version the target is on now (`docs/testing-cloud-services.md` section 4).
2. **Backend** (safety chain: availability restrictions, erasure, decisions). Deploy it before any client.
3. **Website** last, once the backend is live. It only uses the additive API.
4. **Do not roll out while required cleanup is unverified.** In an authorized nonproduction environment, confirm the selected provider's least-privilege per-photo erase permissions work. ai-api (`AiApiClient.erasePhotoFaces`): list the event's persons, `DELETE /api/v1/faces/persons/{id}`, list again (the `?event_id=` form is the event-wide bulk erase, not this path). Rekognition (`RekognitionAiClient.erasePhotoFaces`): `ListFaces` and `DeleteFaces` on the event collection.

Release gates (steps and rules in `docs/testing-cloud-services.md` section 1; ask the owner first; skipped is not passed):

1. `RekognitionCredentialsSmokeTest`: collection create and delete.
2. `RekognitionR2E2eSmokeTest`: R2 storage round-trip.
3. `RekognitionR2E2eSmokeTest`: face enroll and search.
4. `RekognitionR2E2eSmokeTest`: OCR.
5. `R2DirectUploadSmokeTest`: direct-upload bucket rules.
6. Signed URL before and after removal with authorized test assets, recording that it works until expiry.
7. The open storage, backup and CDN gates above.

This page authorizes no deployment or production deletion.

## Rollback

- Never down-migrate the removal tables or history, and never restore deleted bytes.
- Never roll the backend back to a build that predates the restrictions. Code from before V69 ignores `removal_state` and would serve restricted photos. A build missing already-applied migrations (V80 to V84) also fails Flyway validation at boot.
- The website can roll back on its own to a build that talks to the same API, since the API is additive.
- There is no config switch to disable new case intake. To stop intake, admins stop creating cases. Restriction reads and cleanup sweeps keep running either way.
