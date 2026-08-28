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
  status: "live";
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
): Promise<PhotographerEventSummary[]> {
  const p = new URLSearchParams();
  if (args.withUploads) p.set("withUploads", "true");
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? 24));
  const res = await api.get<PaginatedResponse<PhotographerEventSummary>>(
    `/me/photographer/events?${p.toString()}`,
  );
  return res.items;
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
}

export async function fetchPhotographerEventPhotos(
  eventId: string,
  args: PhotographerEventPhotosArgs = {},
): Promise<PhotographerLibraryPhoto[]> {
  const p = new URLSearchParams();
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? 120));
  if (args.order) p.set("order", args.order);
  const res = await api.get<PaginatedResponse<PhotographerLibraryPhoto>>(
    `/me/photographer/events/${encodeURIComponent(eventId)}/photos?${p.toString()}`,
  );
  return res.items;
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
