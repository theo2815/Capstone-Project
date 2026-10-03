import { ApiError, api, refreshAccessToken } from "@/lib/api";
import { getAccessToken } from "@/lib/auth";
import { captureAuthSession } from "@/lib/auth-session";
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
//   POST   /api/v1/me/photographer/events/{id}/photos/direct         (begin)
//   PUT    response uploadUrl                                        (raw file)
//   POST   /api/v1/me/photographer/events/{id}/photos/direct/commit  (receipt)
//   POST   /api/v1/me/photographer/events/{id}/photos                (multipart fallback)

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
  status: "live" | "processing" | "hidden";
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
  signal?: AbortSignal,
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
    { signal },
  );
}

export async function fetchPhotographerEventDetail(
  eventId: string,
  signal?: AbortSignal,
): Promise<PhotographerEventDetail | null> {
  return api.get<PhotographerEventDetail>(
    `/me/photographer/events/${encodeURIComponent(eventId)}`,
    { signal },
  );
}

export interface MetadataSettingsDto {
  metadataPublicByDefault: boolean;
}

export interface EventMetadataVisibilityDto {
  metadataPublic: boolean;
}

export async function getMetadataSettings(signal?: AbortSignal): Promise<MetadataSettingsDto> {
  return api.get<MetadataSettingsDto>("/me/photographer/metadata-settings", { signal });
}

export async function putMetadataSettings(settings: MetadataSettingsDto): Promise<MetadataSettingsDto> {
  return api.put<MetadataSettingsDto>("/me/photographer/metadata-settings", settings);
}

export async function getEventMetadataVisibility(eventId: string, signal?: AbortSignal): Promise<EventMetadataVisibilityDto> {
  return api.get<EventMetadataVisibilityDto>(`/me/photographer/events/${encodeURIComponent(eventId)}/metadata-visibility`, { signal });
}

export async function putEventMetadataVisibility(eventId: string, visibility: EventMetadataVisibilityDto): Promise<EventMetadataVisibilityDto> {
  return api.put<EventMetadataVisibilityDto>(`/me/photographer/events/${encodeURIComponent(eventId)}/metadata-visibility`, visibility);
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
  signal?: AbortSignal,
): Promise<PaginatedResponse<PhotographerLibraryPhoto>> {
  const p = new URLSearchParams();
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? 120));
  if (args.order) p.set("order", args.order);
  if (args.deleted) p.set("deleted", "true");
  return api.get<PaginatedResponse<PhotographerLibraryPhoto>>(
    `/me/photographer/events/${encodeURIComponent(eventId)}/photos?${p.toString()}`,
    { signal },
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

// XHR retains upload progress, including on the raw storage PUT.
export interface UploadProgressEvent {
  loaded: number;
  total: number;
  percent: number;
}

export interface DirectUploadCheckpoint {
  requestId: string;
  photoId?: string;
  key?: string;
  putExpiresAt?: string;
  intentExpiresAt?: string;
}

export interface UploadRetryState {
  beginRetries?: number;
  transferRetries?: number;
  commitRetries?: number;
  commitBusyUntil?: number;
  commitBusyWaits?: number;
  transferDeadline?: number;
  /** Grant (photoId) that transferDeadline belongs to; a renewed grant gets its own cap. */
  transferGrant?: string;
}

export interface PhotographerUploadOptions {
  contentHash?: string;
  signal?: AbortSignal;
  checkpoint?: DirectUploadCheckpoint;
  retryState?: UploadRetryState;
  onCheckpoint?: (value: DirectUploadCheckpoint) => void;
  onPhase?: (phase: "preparing" | "uploading" | "confirming") => void;
}

interface DirectBeginResponse {
  mode: "direct" | "existing" | "multipart";
  photoId?: string;
  key?: string;
  uploadUrl?: string;
  expiresAt?: string;
  intentExpiresAt?: string;
  uploadHeaders?: Record<string, string>;
  existing?: UploadedPhoto;
}

export class OfflineUploadError extends Error {
  constructor() { super("Waiting for connection"); }
}

function uploadError(error: unknown): boolean {
  if (error instanceof ApiError) {
    if (error.errors.some((item) => item.code === "UPLOAD_IN_PROGRESS" || item.code === "RATE_LIMITED" && error.status === 429)) return true;
    return error.status === undefined || error.status >= 500;
  }
  return error instanceof TypeError;
}

function assertUploadActive(signal: AbortSignal | undefined, isCurrent: () => boolean) {
  if (signal?.aborted) throw new DOMException("Upload cancelled", "AbortError");
  if (!isCurrent()) throw new Error("Session changed");
  if (typeof navigator !== "undefined" && !navigator.onLine) throw new OfflineUploadError();
}

function retryDelay(error: unknown, attempt: number): number {
  const jittered = 1000 * 2 ** attempt * (0.8 + Math.random() * 0.4);
  return Math.max(jittered, error instanceof ApiError ? (error.retryAfterSeconds ?? 0) * 1000 : 0);
}

// Commit is idempotent and the original is already staged: verification
// backpressure is a wait, not a failure. Bounded per entry.
const COMMIT_BUSY_WINDOW_MS = 10 * 60_000;

function verificationBusy(error: unknown): boolean {
  return error instanceof ApiError && error.errors.some((item) =>
    item.code === "UPLOAD_IN_PROGRESS" || (item.code === "RATE_LIMITED" && error.status === 429));
}

function waitForRetry(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new DOMException("Upload cancelled", "AbortError")); return; }
    const timer = setTimeout(done, ms);
    function done() { signal?.removeEventListener("abort", abort); resolve(); }
    function abort() { clearTimeout(timer); reject(new DOMException("Upload cancelled", "AbortError")); }
    signal?.addEventListener("abort", abort, { once: true });
  });
}

async function scheduleUploadRetry(
  error: unknown,
  signal: AbortSignal | undefined,
  isCurrent: () => boolean,
  retryState: UploadRetryState,
  phase: "beginRetries" | "transferRetries" | "commitRetries",
): Promise<void> {
  if (signal?.aborted || !isCurrent()) assertUploadActive(signal, isCurrent);
  if (phase === "commitRetries" && verificationBusy(error)) {
    const until = retryState.commitBusyUntil ??= Date.now() + COMMIT_BUSY_WINDOW_MS;
    if (Date.now() < until) {
      const waits = (retryState.commitBusyWaits ?? 0) + 1;
      retryState.commitBusyWaits = waits;
      assertUploadActive(signal, isCurrent);
      await waitForRetry(Math.min(15_000, retryDelay(error, Math.min(waits - 1, 4))), signal);
      return;
    }
  }
  const attempt = retryState[phase] ?? 0;
  if (attempt >= 3) throw error;
  retryState[phase] = attempt + 1;
  assertUploadActive(signal, isCurrent);
  await waitForRetry(retryDelay(error, attempt), signal);
}

async function retryUpload<T>(
  work: () => Promise<T>,
  signal: AbortSignal | undefined,
  isCurrent: () => boolean,
  retryState: UploadRetryState,
  phase: "beginRetries" | "transferRetries" | "commitRetries",
  canRetry: (error: unknown) => boolean = uploadError,
): Promise<T> {
  for (;;) {
    assertUploadActive(signal, isCurrent);
    try {
      const result = await work();
      assertUploadActive(signal, isCurrent);
      return result;
    } catch (error) {
      if (signal?.aborted || !isCurrent()) assertUploadActive(signal, isCurrent);
      if (!canRetry(error)) throw error;
      await scheduleUploadRetry(error, signal, isCurrent, retryState, phase);
    }
  }
}

function uploadRequestId(): string {
  if (crypto.randomUUID) return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function validateUploadUrl(url: string): void {
  const configured = process.env.NEXT_PUBLIC_R2_UPLOAD_ORIGIN;
  if (!configured) throw new Error("Direct upload is not configured.");
  let expected: URL;
  let actual: URL;
  try {
    expected = new URL(configured);
    actual = new URL(url);
  } catch { throw new Error("Invalid upload URL."); }
  if (expected.protocol !== "https:" || expected.origin !== configured || actual.protocol !== "https:" || actual.origin !== expected.origin || actual.username || actual.password) {
    throw new Error("Invalid upload URL.");
  }
}

function rawPut(
  url: string,
  file: File,
  headers: Record<string, string>,
  onProgress: ((event: UploadProgressEvent) => void) | undefined,
  signal: AbortSignal | undefined,
  deadline: number,
): Promise<void> {
  validateUploadUrl(url);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    let settled = false;
    let inactivity: ReturnType<typeof setTimeout>;
    let lastLoaded = 0;
    const absolute = setTimeout(() => fail(new ApiError([{ code: "TIMEOUT", message: "Upload timed out." }])), Math.max(0, deadline - Date.now()));
    const resetInactivity = () => {
      clearTimeout(inactivity);
      inactivity = setTimeout(() => fail(new ApiError([{ code: "TIMEOUT", message: "Upload stalled." }])), 120_000);
    };
    const finish = () => {
      if (settled) return false;
      settled = true;
      clearTimeout(absolute);
      clearTimeout(inactivity);
      signal?.removeEventListener("abort", abort);
      return true;
    };
    const fail = (error: Error) => { if (finish()) { xhr.abort(); reject(error); } };
    const abort = () => fail(new DOMException("Upload cancelled", "AbortError"));
    if (signal?.aborted) { abort(); return; }
    signal?.addEventListener("abort", abort, { once: true });
    xhr.open("PUT", url);
    xhr.withCredentials = false;
    for (const [name, value] of Object.entries(headers)) xhr.setRequestHeader(name, value);
    xhr.upload.onprogress = (event) => {
      if (event.loaded <= lastLoaded) return;
      lastLoaded = event.loaded;
      resetInactivity();
      if (event.lengthComputable) onProgress?.({ loaded: event.loaded, total: event.total, percent: Math.round(event.loaded / event.total * 100) });
    };
    xhr.onload = () => {
      if (!finish()) return;
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else reject(new ApiError([{ code: "UPLOAD_TRANSFER_FAILED", message: `Upload failed (${xhr.status})` }], xhr.status));
    };
    xhr.onerror = () => fail(new TypeError("Upload network error"));
    xhr.onabort = () => fail(new DOMException("Upload cancelled", "AbortError"));
    resetInactivity();
    xhr.send(file);
  });
}

export async function uploadPhotographerPhoto(
  eventId: string,
  file: File,
  onProgress?: (event: UploadProgressEvent) => void,
  options: PhotographerUploadOptions = {},
): Promise<UploadedPhoto> {
  const isCurrent = captureAuthSession();
  const signal = options.signal;
  const retryState = options.retryState ?? {};
  assertUploadActive(signal, isCurrent);
  options.onPhase?.("preparing");
  const base = `/me/photographer/events/${encodeURIComponent(eventId)}/photos`;
  // Web Crypto can be unavailable on insecure origins. No direct grant has
  // been requested yet, so the backend's multipart route is safe here.
  if ((!globalThis.crypto?.subtle && (!options.contentHash || !options.checkpoint?.photoId || !options.checkpoint.key)) ||
      (!options.checkpoint && !globalThis.crypto?.getRandomValues)) {
    return multipartUpload(eventId, file, onProgress, signal, isCurrent, retryState, options.onPhase);
  }
  const contentHash = options.contentHash ?? await sha256Hex(file);
  assertUploadActive(signal, isCurrent);
  let checkpoint = options.checkpoint;
  if (!checkpoint) {
    checkpoint = { requestId: uploadRequestId() };
    options.onCheckpoint?.(checkpoint);
  }
  const save = (value: DirectUploadCheckpoint) => {
    assertUploadActive(signal, isCurrent);
    checkpoint = value;
    options.onCheckpoint?.(value);
  };
  const renewGrant = () => save({ requestId: uploadRequestId() });
  const commit = () => retryUpload(() => api.post<UploadedPhoto>(`${base}/direct/commit`, {
    photoId: checkpoint!.photoId, key: checkpoint!.key, contentHash,
    contentType: file.type, sizeBytes: file.size,
  }, { signal, timeoutMs: 120_000 }), signal, isCurrent, retryState, "commitRetries");

  // A retained receipt may already have reached storage before the tab lost
  // its reply. Ask the backend before making another grant or sending bytes.
  if (checkpoint.photoId && checkpoint.key) {
    options.onPhase?.("confirming");
    try { return await commit(); }
    catch (error) {
      if (!(error instanceof ApiError) || !error.errors.some((item) => item.code === "UPLOAD_OBJECT_MISSING")) throw error;
      if ((checkpoint.putExpiresAt && Date.parse(checkpoint.putExpiresAt) <= Date.now()) ||
          (checkpoint.intentExpiresAt && Date.parse(checkpoint.intentExpiresAt) <= Date.now())) {
        if (!globalThis.crypto?.getRandomValues) return multipartUpload(eventId, file, onProgress, signal, isCurrent, retryState, options.onPhase);
        await scheduleUploadRetry(error, signal, isCurrent, retryState, "beginRetries");
        renewGrant();
      }
    }
  }

  for (;;) {
    let begin: DirectBeginResponse;
    try {
      begin = await retryUpload(() => api.post<DirectBeginResponse>(`${base}/direct`, {
        contentHash, contentType: file.type, sizeBytes: file.size,
        originalFilename: file.name, requestId: checkpoint!.requestId, client: "website",
      }, { signal }), signal, isCurrent, retryState, "beginRetries");
    } catch (error) {
      if (error instanceof ApiError && error.errors.some((item) => item.code === "UPLOAD_OBJECT_MISSING")) {
        await scheduleUploadRetry(error, signal, isCurrent, retryState, "beginRetries");
        renewGrant();
        continue;
      }
      // Older backend without the direct route: only a generic 404 qualifies.
      if (error instanceof ApiError && error.status === 404 && error.errors.every((item) => item.code === "UNKNOWN" || item.code === "NOT_FOUND")) {
        return multipartUpload(eventId, file, onProgress, signal, isCurrent, retryState, options.onPhase);
      }
      throw error;
    }
    if (begin.mode === "existing" && begin.existing) return begin.existing;
    if (begin.mode === "multipart") return multipartUpload(eventId, file, onProgress, signal, isCurrent, retryState, options.onPhase);
    if (begin.mode !== "direct" || !begin.photoId || !begin.key || !begin.uploadUrl) throw new Error("Invalid direct upload grant.");
    validateUploadUrl(begin.uploadUrl);
    save({ requestId: checkpoint.requestId, photoId: begin.photoId, key: begin.key, putExpiresAt: begin.expiresAt, intentExpiresAt: begin.intentExpiresAt });
    options.onPhase?.("uploading");
    let recovered: UploadedPhoto | undefined;
    let confirmationUnknown = false;
    let renewAfterPut = false;
    if (retryState.transferGrant !== begin.photoId) {
      retryState.transferGrant = begin.photoId;
      retryState.transferDeadline = undefined;
    }
    const transferDeadline = retryState.transferDeadline ??= Date.now() + 15 * 60_000;
    try {
      await retryUpload(async () => {
        try {
          await rawPut(begin.uploadUrl!, file, begin.uploadHeaders ?? {}, onProgress, signal, transferDeadline);
        } catch (error) {
          const storageDenied = error instanceof ApiError && error.status === 403 && error.errors.some((item) => item.code === "UPLOAD_TRANSFER_FAILED");
          // A lost PUT reply or storage 403 may hide a complete object.
          // Reconcile before sending bytes again or renewing a grant.
          if (uploadError(error) || storageDenied) {
            options.onPhase?.("confirming");
            try { recovered = await commit(); return; }
            catch (commitError) {
              if (commitError instanceof OfflineUploadError) {
                const retries = retryState.transferRetries ?? 0;
                if (retries >= 3) throw error;
                retryState.transferRetries = retries + 1;
                throw commitError;
              }
              if (!(commitError instanceof ApiError) || !commitError.errors.some((item) => item.code === "UPLOAD_OBJECT_MISSING")) {
                confirmationUnknown = true;
                throw commitError;
              }
              if (begin.expiresAt && Date.parse(begin.expiresAt) <= Date.now()) {
                // The grant expired and storage confirms nothing arrived: only a
                // fresh reservation can finish this file, whatever failed the PUT.
                renewAfterPut = true;
                throw error;
              }
              options.onPhase?.("uploading");
            }
          }
          throw error;
        }
      }, signal, isCurrent, retryState, "transferRetries", (error) => !confirmationUnknown && !renewAfterPut && uploadError(error));
    } catch (error) {
      if (!renewAfterPut) throw error;
      await scheduleUploadRetry(error, signal, isCurrent, retryState, "transferRetries");
      renewGrant();
      continue;
    }
    assertUploadActive(signal, isCurrent);
    if (recovered) return recovered;
    options.onPhase?.("confirming");
    return commit();
  }
}

function multipartUpload(
  eventId: string,
  file: File,
  onProgress: ((event: UploadProgressEvent) => void) | undefined,
  signal: AbortSignal | undefined,
  isCurrent: () => boolean,
  retryState: UploadRetryState,
  onPhase?: PhotographerUploadOptions["onPhase"],
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
      const abort = () => xhr.abort();
      if (signal?.aborted || !isCurrent()) { reject(new DOMException("Upload cancelled", "AbortError")); return; }
      signal?.addEventListener("abort", abort, { once: true });
      const done = () => signal?.removeEventListener("abort", abort);

      xhr.open("POST", url);
      xhr.withCredentials = false;
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
        done();
        if (signal?.aborted || !isCurrent()) { reject(new DOMException("Upload cancelled", "AbortError")); return; }
        if (xhr.status === 401 && !retried) {
          refreshAccessToken()
            .then((fresh) =>
              fresh && isCurrent() && !signal?.aborted
                ? resolve(attempt(fresh, true))
                : reject(new Error("Session expired. Sign in again to retry.")),
            )
            .catch(() =>
              reject(new Error("Session expired. Sign in again to retry.")),
            );
          return;
        }
        onPhase?.("confirming");
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
          const retryAfter = Number(xhr.getResponseHeader?.("Retry-After"));
          reject(new ApiError(
            !body.success && body.errors.length ? body.errors : [{ code: "UNKNOWN", message: `Upload failed (${xhr.status})` }],
            xhr.status,
            Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter : undefined,
          ));
        } catch {
          reject(new Error(`Upload failed (${xhr.status})`));
        }
      };

      xhr.onerror = () => { done(); reject(new TypeError("Upload network error")); };
      xhr.ontimeout = () => { done(); reject(new ApiError([{ code: "TIMEOUT", message: "Upload timed out. Check your connection and retry." }])); };
      xhr.onabort = () => { done(); reject(new DOMException("Upload cancelled", "AbortError")); };

      const fd = new FormData();
      fd.append("file", file);
      onPhase?.("uploading");
      xhr.send(fd);
    });

  return retryUpload(() => attempt(getAccessToken(), false), signal, isCurrent, retryState, "transferRetries");
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
  signal?: AbortSignal,
): Promise<PhotoExistsResult[]> {
  const res = await api.post<PhotoExistsResponse>(
    `/me/photographer/events/${encodeURIComponent(eventId)}/photos/exists`,
    { hashes },
    { signal },
  );
  return res.results;
}
