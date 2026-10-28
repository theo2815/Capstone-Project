# Desktop App - QuickPitik Upload Integration Guide

How BatchMyPhotos (the Electron desktop app, external repo) signs in to the QuickPitik Spring Boot backend as a photographer and uploads a whole shoot folder to an event. Design: BatchMyPhotos `desktop/docs/superpowers/specs/2026-10-10-quickpitik-upload-workflow-design.md`. This is the only home for the hand-off contract; the upload contract itself lives in the backend code named below.

The desktop never calls `ai-api` for this flow and uses no API key. It is a signed-in photographer, exactly like the website and the mobile app.

## 1. Sign-in hand-off (one-time code + PKCE)

1. The app makes `state` and `verifier`, each `base64url(32 random bytes)` (43 characters), computes `challenge = base64url(sha256(verifier))`, keeps `{ state, verifier }` in main-process memory for 10 minutes, and opens the browser at `https://quickpitik.com/connect/desktop?state=<state>&challenge=<challenge>`.
2. The website page requires a signed-in photographer (a signed-out visitor goes through `/login?redirect=…` and comes back). It validates both parameters against `^[A-Za-z0-9_-]{43}$`.
3. On Connect the website calls `POST /api/v1/auth/desktop/codes` with `{ "codeChallenge": "<challenge>" }` (bearer, PHOTOGRAPHER role). Answer: `{ "code": "<43 chars>", "expiresInSeconds": 120 }`, `Cache-Control: no-store`.
4. The website sends the browser to `batchmyphotos://quickpitik/callback?code=<code>&state=<state>`.
5. The app checks `state` against its pending request, then calls the public `POST /api/v1/auth/desktop/exchange` with `{ "code": "<code>", "codeVerifier": "<verifier>" }`. Answer: the normal `AuthResponse` `{ accessToken, refreshToken, user }`. The code is single use; expired, used, mismatched or malformed codes all answer `401 INVALID_TOKEN` with "This connect code is invalid or has expired." A suspended account answers `401 ACCOUNT_SUSPENDED`. The route shares the `auth-login` per-IP rate policy.
6. The app stores the pair in its own encrypted store and refreshes with `POST /api/v1/auth/refresh { refreshToken }` (rotating; save both new tokens at once). If the answer to a refresh is lost, presenting the same refresh token again within `jwt.refresh-rotation-grace` (10 minutes) gives a fresh pair, unless the session was revoked in between (password change, "revoke all"). Disconnect calls `POST /api/v1/auth/logout { refreshToken }` best effort, then clears the store.

Code: `backend/src/main/kotlin/com/quickpitik/service/auth/DesktopAuthService.kt`, `AuthController.kt`, table `desktop_auth_codes` (V85). Website page: `website/src/components/auth/connect-desktop.tsx`.

## 2. Events the app offers for upload

Public events from `GET /api/v1/events?limit=200` whose date is within race day and the next 3 days (Asia/Manila), plus the photographer's own and covered events from `GET /api/v1/me/photographer/events?limit=200`. Each item of that list carries `ownedByMe`: an owned event (`true`) takes uploads until one calendar month after its date, with no lower bound; a covered event (`false`) gets the same race-day-plus-3 window as a public one. Same rules as `website/src/lib/event-catalog.ts`.

Do not send `includeListed=true` on that public list. It adds other photographers' events an admin chose to show on `/events` (2026-10-28), and those don't take this photographer's uploads. Only the website `/events` page and mobile runner discovery send it.

That list is not filtered by status or date, and only ACTIVE or COMPLETED events take uploads, so an owned event still waiting for review, or an archived one, can pass the date check and still be refused. The backend gate in `PhotoUploadService.gate` has the final say on every upload: it answers `422 EVENT_NOT_UPLOADABLE` with a message the app shows to the photographer.

## 3. Upload protocol (unchanged, per photo)

The desktop uses the same three calls the website and mobile app use, with no `client` field:

| Step | Call | Notes |
|---|---|---|
| Pre-flight | `POST /me/photographer/events/{eventId}/photos/exists { hashes: [sha256 hex…] }` | Up to 500 hashes; answers `new` / `same_event` / `different_event` per hash, with `inTrash: true` when that photo sits in the photographer's trash (begin's `existing` carries the same flag); the desktop counts those apart from "already there" |
| Begin | `POST /me/photographer/events/{eventId}/photos/direct { contentHash, contentType, sizeBytes, originalFilename, requestId }` | Answers `mode: direct` (PUT the bytes to `uploadUrl` with `uploadHeaders` before `expiresAt`), `existing` (already stored) or `multipart` |
| Commit | `POST /me/photographer/events/{eventId}/photos/direct/commit { photoId, key, contentHash, sizeBytes, contentType }` | Idempotent; `UPLOAD_IN_PROGRESS` means wait |
| Multipart | `POST /me/photographer/events/{eventId}/photos` with form field `file` | Only when begin answered `multipart` |

Accepted: `image/jpeg`, `image/png`, `image/webp`, at most `storage.direct-upload.max-bytes` (25 MiB default). Limits: `photographer-upload` 600 calls/min per photographer; pending direct uploads per user capped (`max-pending-per-user`, `max-pending-bytes-per-user`). The app keeps 4 files in flight and one subfolder at a time. Retry and error rules the app follows: spec section "Upload protocol rules for the desktop client". Authoritative code: `backend/src/main/kotlin/com/quickpitik/service/photographer/PhotoUploadService.kt`; website reference client: `website/src/lib/api-photographer.ts` (`uploadPhotographerPhoto`).

## 4. Desktop configuration

Set in BatchMyPhotos `desktop/src/main/config.js` (packaged defaults) and overridable in development with `BATCH_QUICKPITIK_API_URL`, `BATCH_QUICKPITIK_WEB_URL`, `BATCH_QUICKPITIK_UPLOAD_ORIGIN`, `BATCH_QUICKPITIK_UPLOAD_ENABLED`. The upload origin is the storage host every presigned `uploadUrl` must match before bytes are sent, the same value as the website's `NEXT_PUBLIC_R2_UPLOAD_ORIGIN`. Any environment can answer `multipart` on begin, not only local: staging and production do too while direct upload is switched off (`STORAGE_DIRECT_UPLOAD_ENABLED`, off by default), so the app supports both paths everywhere.

| | Local | Staging | Production |
|---|---|---|---|
| API | `http://localhost:8080/api/v1` | the Railway staging domain + `/api/v1` (`docs/testing-cloud-services.md` § 2) | `https://api.quickpitik.com/api/v1` |
| Website | `http://localhost:3000` | the Vercel preview URL | `https://quickpitik.com` |
| Upload origin | not needed (local storage answers `multipart`) | the dev bucket's origin | the production bucket's origin |
