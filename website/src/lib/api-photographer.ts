import { api, refreshAccessToken } from "@/lib/api";
import { getAccessToken } from "@/lib/auth";
import { API_BASE_URL } from "@/lib/constants";
import type {
  PhotographerEventSummary,
  PhotographerLibraryPhoto,
} from "@/lib/photographer-mock";
import type { PaginatedResponse } from "@/types/api";

// Phase F.2 photographer backend contract
//   Q-013 (upload contract), Q-014 (covered events listing + photoCount semantics),
//   Q-015 (download URL signing) RESOLVED 2026-05-10. See vault decisions.
//
//   GET    /api/v1/me/photographer/events?withUploads=&offset=&limit=
//          → PaginatedResponse<PhotographerEventSummary>
//   GET    /api/v1/me/photographer/events/{id}                       → PhotographerEventDetail
//   GET    /api/v1/me/photographer/events/{id}/photos?offset=&limit=&order=
//          → PaginatedResponse<PhotographerLibraryPhoto>
//   GET    /api/v1/me/photographer/photos/{id}/download              → { url, expiresAt }
//   POST   /api/v1/me/photographer/events/{id}/photos                (multipart, single file)
//          → UploadedPhoto

export interface PhotographerEventDetail extends PhotographerEventSummary {
  firstUploadAt: string | null;
  lastUploadAt: string | null;
  // V46 — prefill for the owner's edit form.
  description: string;
  organizerName: string;
  pricePerPhoto: number;
}

// Photographer-owned events (V46). Multipart like the admin create; text
// fields ride as form parts, `cover` carries the bytes.
//   POST   /api/v1/me/photographer/events        → PhotographerEventDetail (pending review)
//   PATCH  /api/v1/me/photographer/events/{id}   → PhotographerEventDetail
// On a live event the pricing trio (pricingMode · pricePerPhoto ·
// watermarkPolicy) becomes an edit request the admin approves; the BE
// parks it in `pendingChange` and the event keeps its current settings.
export interface MyEventFields {
  title?: string;
  date?: string;
  location?: string;
  organizerName?: string;
  description?: string;
  visibility?: "public" | "unlisted";
  pricingMode?: "paid" | "free";
  pricePerPhoto?: number;
  watermarkPolicy?: "platform" | "own" | "none";
  cover?: File | null;
  withdrawPendingChange?: boolean;
}

function toEventForm(fields: MyEventFields): FormData {
  const form = new FormData();
  for (const [key, value] of Object.entries(fields)) {
    if (value === undefined || value === null || key === "cover") continue;
    form.append(key, String(value));
  }
  if (fields.cover) form.append("cover", fields.cover);
  return form;
}

export async function createMyEvent(
  fields: MyEventFields,
): Promise<PhotographerEventDetail> {
  return api.post<PhotographerEventDetail>(
    "/me/photographer/events",
    toEventForm(fields),
  );
}

export async function updateMyEvent(
  eventId: string,
  fields: MyEventFields,
): Promise<PhotographerEventDetail> {
  return api.fetch<PhotographerEventDetail>(
    `/me/photographer/events/${encodeURIComponent(eventId)}`,
    { method: "PATCH", body: toEventForm(fields) },
  );
}

// aiDetectionStatus surfaces partial failure of the best-effort ai-api
// faces+bibs pipeline so the upload page can warn the photographer that
// runners may not be able to find the photo by selfie/bib search yet.
// Absent (legacy) or "ok" means both subsystems ran cleanly. (H-5)
export type UploadedPhotoAiDetectionStatus =
  | "ok"
  | "faces_unavailable"
  | "bibs_unavailable"
  | "none";

export interface UploadedPhoto {
  id: string;
  status: "live" | "processing";
  uploadedAt: string;
  thumbnailUrl: string;
  span: "default" | "wide";
  aiDetectionStatus?: UploadedPhotoAiDetectionStatus;
}

export interface PhotographerEventListArgs {
  withUploads?: boolean;
  offset?: number;
  limit?: number;
}

export async function fetchPhotographerEvents(
  args: PhotographerEventListArgs = {},
): Promise<PaginatedResponse<PhotographerEventSummary>> {
  const p = new URLSearchParams();
  if (args.withUploads) p.set("withUploads", "true");
  p.set("offset", String(args.offset ?? 0));
  // Default to the BE MAX_LIMIT: dashboard consumers (setup-mode fork,
  // Next-up glance) derive from this list and omit `limit` — the old 24
  // default silently truncated what they saw.
  p.set("limit", String(args.limit ?? 200));
  return api.get<PaginatedResponse<PhotographerEventSummary>>(
    `/me/photographer/events?${p.toString()}`,
  );
}

export async function fetchPhotographerEventDetail(
  eventId: string,
): Promise<PhotographerEventDetail | null> {
  return api.get<PhotographerEventDetail>(
    `/me/photographer/events/${encodeURIComponent(eventId)}`,
  );
}

export interface PhotographerEventPhotosArgs {
  offset?: number;
  limit?: number;
  order?: "newest" | "oldest";
  /** true → the Deleted Photos (trash) list; false/omitted → active gallery. */
  deleted?: boolean;
}

export async function fetchPhotographerEventPhotos(
  eventId: string,
  args: PhotographerEventPhotosArgs = {},
): Promise<PaginatedResponse<PhotographerLibraryPhoto>> {
  const p = new URLSearchParams();
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? 120));
  if (args.order) p.set("order", args.order);
  if (args.deleted) p.set("deleted", "true");
  return api.get<PaginatedResponse<PhotographerLibraryPhoto>>(
    `/me/photographer/events/${encodeURIComponent(eventId)}/photos?${p.toString()}`,
  );
}

export interface PhotographerDownloadResponse {
  url: string;
  expiresAt: string;
}

export async function fetchPhotographerPhotoDownload(
  photoId: string,
): Promise<PhotographerDownloadResponse | null> {
  return api.get<PhotographerDownloadResponse>(
    `/me/photographer/photos/${encodeURIComponent(photoId)}/download`,
  );
}

// ── Photo library management (V52) ────────────────────────────────────────
// Trash / restore / permanent-delete are bulk endpoints; the per-photo action
// in the UI just sends a one-element list. Ownership + event scoping are
// enforced server-side, so a foreign id is silently ignored.
//   POST /me/photographer/events/{id}/photos/trash            { photoIds } → { affected }
//   POST /me/photographer/events/{id}/photos/restore          { photoIds } → { affected }
//   POST /me/photographer/events/{id}/photos/permanent-delete { photoIds } → { deleted, blockedSold }
//   POST /me/photographer/events/{id}/download-bundle                      → { url, expiresAt }

export async function trashPhotographerPhotos(
  eventId: string,
  photoIds: string[],
): Promise<{ affected: number; failed: string[] }> {
  return movePhotographerPhotos(eventId, photoIds, "trash");
}

export async function restorePhotographerPhotos(
  eventId: string,
  photoIds: string[],
): Promise<{ affected: number; failed: string[] }> {
  return movePhotographerPhotos(eventId, photoIds, "restore");
}

async function movePhotographerPhotos(eventId: string, photoIds: string[], action: "trash" | "restore") {
  const result = { affected: 0, failed: [] as string[] };
  const ids = [...new Set(photoIds)];
  // PhotoIdsRequest caps trash/restore at 500; the UI can select many pages.
  for (let offset = 0; offset < ids.length; offset += 500) {
    const batch = ids.slice(offset, offset + 500);
    try {
      const res = await api.post<{ affected: number }>(
        `/me/photographer/events/${encodeURIComponent(eventId)}/photos/${action}`,
        { photoIds: batch },
      );
      result.affected += res.affected;
    } catch {
      result.failed.push(...batch);
    }
  }
  return result;
}

export interface PermanentDeleteResult {
  /** Photo ids actually purged (row + R2 objects gone). */
  deleted: string[];
  /** Photo ids refused because they've been sold — kept so buyers can still
   *  download them. */
  blockedSold: string[];
  blockedPending?: string[];
  failed?: string[];
}

export async function permanentDeletePhotographerPhotos(
  eventId: string,
  photoIds: string[],
): Promise<PermanentDeleteResult> {
  const result = { deleted: [] as string[], blockedSold: [] as string[], blockedPending: [] as string[], failed: [] as string[] };
  const ids = [...new Set(photoIds)];
  // MePhotographerController caps permanent-delete at 100 storage operations.
  for (let offset = 0; offset < ids.length; offset += 100) {
    const batch = ids.slice(offset, offset + 100);
    try {
      const res = await api.post<PermanentDeleteResult>(
        `/me/photographer/events/${encodeURIComponent(eventId)}/photos/permanent-delete`,
        { photoIds: batch },
      );
      result.deleted.push(...res.deleted);
      result.blockedSold.push(...res.blockedSold);
      result.blockedPending.push(...(res.blockedPending ?? []));
      const handled = new Set([...res.deleted, ...res.blockedSold, ...(res.blockedPending ?? [])]);
      result.failed.push(...batch.filter((id) => !handled.has(id)));
    } catch {
      result.failed.push(...batch);
    }
  }
  return result;
}

export interface PhotographerBundleResponse {
  /** Short-lived tokened URL to navigate to for the ZIP of all active photos. */
  url: string;
  expiresAt: string;
}

export async function preparePhotographerBundle(
  eventId: string,
): Promise<PhotographerBundleResponse> {
  return api.post<PhotographerBundleResponse>(
    `/me/photographer/events/${encodeURIComponent(eventId)}/download-bundle`,
    {},
  );
}

// XHR upload — fetch has no native onprogress for the request body. Caller
// fan-outs N parallel uploads + dedupe; one POST per file per Q-013.
export interface UploadProgressEvent {
  loaded: number;
  total: number;
  percent: number;
}

export function uploadPhotographerPhoto(
  eventId: string,
  file: File,
  onProgress?: (event: UploadProgressEvent) => void,
): Promise<UploadedPhoto> {
  const url = `${API_BASE_URL}/me/photographer/events/${encodeURIComponent(eventId)}/photos`;

  // One attempt = one XHR. A 401 mid-batch is routine, not exceptional:
  // access tokens live 15 minutes and a big queue at 4-concurrent outlives
  // that — before this, every file after expiry failed "retryable" while the
  // manual retry re-sent the same dead token. On the first 401, refresh
  // through the shared single-flight (same mutex the ApiClient uses, so a
  // concurrent pre-flight can't double-rotate) and re-send once.
  const attempt = (
    token: string | null,
    retried: boolean,
  ): Promise<UploadedPhoto> =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      xhr.open("POST", url);
      if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
      // A stalled connection must settle, not hang — a dead XHR pinned one of
      // the upload page's 4 concurrency slots forever. Whole-request timer,
      // sized for an 8 MB original on slow venue Wi-Fi.
      xhr.timeout = 120_000;

      if (onProgress) {
        xhr.upload.onprogress = (event) => {
          if (!event.lengthComputable) return;
          onProgress({
            loaded: event.loaded,
            total: event.total,
            percent: Math.round((event.loaded / event.total) * 100),
          });
        };
      }

      xhr.onload = () => {
        if (xhr.status === 401 && !retried) {
          refreshAccessToken()
            .then((fresh) =>
              fresh
                ? resolve(attempt(fresh, true))
                : reject(new Error("Session expired. Sign in again to retry.")),
            )
            .catch(() =>
              reject(new Error("Session expired. Sign in again to retry.")),
            );
          return;
        }
        try {
          const body = JSON.parse(xhr.responseText) as
            | { success: true; data: UploadedPhoto }
            | {
                success: false;
                errors: Array<{ code: string; message: string }>;
              };
          if (xhr.status >= 200 && xhr.status < 300 && body.success) {
            resolve(body.data);
            return;
          }
          const message =
            !body.success && body.errors[0]?.message
              ? body.errors[0].message
              : `Upload failed (${xhr.status})`;
          // Carry the backend error code so the caller can tell a terminal
          // duplicate rejection (never retryable) from a transient network glitch.
          const error = new Error(message) as Error & { code?: string };
          if (!body.success && body.errors[0]?.code) {
            error.code = body.errors[0].code;
          }
          reject(error);
        } catch {
          reject(new Error(`Upload failed (${xhr.status})`));
        }
      };

      xhr.onerror = () => reject(new Error("Upload network error"));
      xhr.ontimeout = () =>
        reject(new Error("Upload timed out. Check your connection and retry."));
      xhr.onabort = () => reject(new Error("Upload aborted"));

      const fd = new FormData();
      fd.append("file", file);
      xhr.send(fd);
    });

  return attempt(getAccessToken(), false);
}

// Pre-flight duplicate check (dedup Phase 2). The upload page hashes each file
// locally and asks the backend which are already stored in one of the
// photographer's events, so it can skip re-sending bytes that are already there
// (the "stop on mobile, continue on web" case). The hash is the SHA-256 of the
// raw file bytes — the same identity the backend recomputes on upload (it hashes
// BEFORE watermarking), so client and server always agree. The unique index on
// (photographer_id, content_hash) is still the authoritative backstop; this
// pre-flight is purely a bandwidth/UX optimization.
//   POST /api/v1/me/photographer/events/{id}/photos/exists  { hashes: [...] }
//        → { results: [{ hash, status, eventName? }] }
export type PhotoExistsStatus = "new" | "same_event" | "different_event";

export interface PhotoExistsResult {
  hash: string;
  status: PhotoExistsStatus;
  eventName: string | null;
}

interface PhotoExistsResponse {
  results: PhotoExistsResult[];
}

// SHA-256 of a file's bytes as lowercase hex, via the Web Crypto API. Mirrors
// the backend's MessageDigest("SHA-256") + HexFormat output exactly.
export async function sha256Hex(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export async function checkPhotosExist(
  eventId: string,
  hashes: string[],
): Promise<PhotoExistsResult[]> {
  const res = await api.post<PhotoExistsResponse>(
    `/me/photographer/events/${encodeURIComponent(eventId)}/photos/exists`,
    { hashes },
  );
  return res.results;
}
