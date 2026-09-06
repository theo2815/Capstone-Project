import { isIOS, isInAppBrowser } from "@/lib/in-app-browser";
import { safeHttpUrl, triggerDownload } from "@/lib/utils";
import { useToastStore } from "@/store/toast-store";

export type SaveResult = "downloaded" | "shared" | "hint";

// Tuning knobs for the bulk "save to device" flow. Conservative defaults —
// revisit once we have real order-size data.
//   SHARE_BATCH_SIZE  files handed to one share sheet. iOS degrades past ~10,
//                     and every File sits in memory until the sheet closes.
//   SHARE_MAX_TOTAL   above this, chunked sharing is more taps than it's worth
//                     — the caller falls back to the server ZIP instead.
export const SHARE_BATCH_SIZE = 10;
export const SHARE_MAX_TOTAL = 30;
// Parallel fetches while assembling a batch: high enough to finish inside the
// tap's ~5 s activation window (or the share loses its user gesture), low
// enough to stay under the photo-download rate limit and not swamp memory.
const FETCH_CONCURRENCY = 4;
const FETCH_RETRIES = 2;

// Save a signed photo download URL to the runner's device.
//
// The plain path is `triggerDownload` — a browser's own attachment download —
// which is right on desktop and Android. Two cases need the native share sheet
// (`navigator.share({ files })`, whose "Save Image" saves straight to the
// gallery) instead:
//   - in-app webviews (Facebook / Messenger / Instagram — WKWebView on iOS),
//     which render a navigated image inline with no way to save it, and
//   - iOS Safari, where a normal download lands in the Files app, not Photos.
// When the share sheet isn't available, an in-app webview gets a how-to-save
// hint (a navigation there won't save); everywhere else falls back to the plain
// download.
//
// The `fetch` is cross-origin (site → api.quickpitik.com) but CORS-allowed
// (CorsConfig whitelists the site origin on `/**`). `navigator.share` needs a
// secure context (prod is HTTPS) and a user gesture, which the button supplies.
export async function savePhoto(url: string): Promise<SaveResult> {
  const href = safeHttpUrl(url);
  if (!href) return "downloaded"; // safeHttpUrl rejected it — nothing to save

  const inApp = isInAppBrowser();

  if (inApp || isIOS()) {
    if (await shareFile(href)) return "shared";
    if (inApp) {
      // A webview navigation can't save the image — tell the runner how.
      showInAppSaveHint();
      return "hint";
    }
    // iOS Safari without file sharing (pre-iOS-15) — fall through to a download.
  }

  triggerDownload(href);
  return "downloaded";
}

// Fetches one photo and hands it to the OS share sheet. Returns true when the
// sheet handled it (a user-cancelled sheet counts — that's a choice, not a
// failure), false when file sharing isn't available so the caller can fall back.
async function shareFile(href: string): Promise<boolean> {
  if (!canShareFiles()) return false;
  try {
    const file = await fetchAsFile(href, 0);
    if (!navigator.canShare({ files: [file] })) return false;
    await navigator.share({ files: [file] });
    return true;
  } catch (err) {
    // AbortError = the runner closed the share sheet; that isn't a failure.
    return err instanceof DOMException && err.name === "AbortError";
  }
}

// Save the inline QR Ph code (the `data:` URL PayMongo returns) to the runner's
// device. Same two cases as savePhoto need the share sheet — in-app webviews
// render the image inline with no way to save it, iOS Safari drops a download in
// Files not Photos — so the runner can get the QR out to scan on another device.
// Unlike savePhoto the bytes are already on the page, so we decode the base64
// locally instead of fetching (a `data:` URL isn't in connect-src). Desktop and
// Android keep the plain `<a download>`, which the caller leaves in place and
// only intercepts for in-app/iOS.
export async function saveQrCode(
  dataUrl: string,
  filename: string,
): Promise<SaveResult> {
  const inApp = isInAppBrowser();
  if (inApp || isIOS()) {
    if (await shareDataUrl(dataUrl, filename)) return "shared";
    if (inApp) {
      showInAppSaveHint();
      return "hint";
    }
    // iOS Safari without file sharing (pre-iOS-15) — fall through to a download.
  }
  downloadDataUrl(dataUrl, filename);
  return "downloaded";
}

// Hands a decoded data-URL image to the OS share sheet. Same contract as
// shareFile (true = handled, incl. a user-cancelled sheet; false = fall back).
async function shareDataUrl(dataUrl: string, filename: string): Promise<boolean> {
  if (!canShareFiles()) return false;
  const file = dataUrlToFile(dataUrl, filename);
  if (!file || !navigator.canShare({ files: [file] })) return false;
  try {
    await navigator.share({ files: [file] });
    return true;
  } catch (err) {
    return err instanceof DOMException && err.name === "AbortError";
  }
}

// Decodes a base64 (or percent-encoded) `data:` URL into a named File using
// atob, so no `fetch` — and therefore no `connect-src` allowance for `data:` —
// is needed. Returns null on a malformed URL so the caller can fall back.
export function dataUrlToFile(dataUrl: string, filename: string): File | null {
  const comma = dataUrl.indexOf(",");
  if (!dataUrl.startsWith("data:") || comma === -1) return null;
  const meta = dataUrl.slice(5, comma);
  const mime = meta.split(";")[0] || "image/png";
  const body = dataUrl.slice(comma + 1);
  try {
    const bytes = /;base64/i.test(meta)
      ? Uint8Array.from(atob(body), (c) => c.charCodeAt(0))
      : new TextEncoder().encode(decodeURIComponent(body));
    return new File([bytes], filename, { type: mime });
  } catch {
    return null;
  }
}

function downloadDataUrl(dataUrl: string, filename: string): void {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

export interface ShareProgress {
  done: number;
  total: number;
}

export interface SharePhotosResult {
  // "shared"      — the sheet opened (or was dismissed); `sharedUrls` reached it.
  // "unsupported" — this browser can't share files; caller should fall back.
  // "error"       — the share threw (e.g. the gesture lapsed); caller falls back.
  status: "shared" | "unsupported" | "error";
  sharedUrls: string[];
  failedUrls: string[];
}

// Fetches a batch of photos and hands them to the OS share sheet in one action,
// so iOS "Save N Images" drops them all in the camera roll. Keep the batch small
// (see SHARE_BATCH_SIZE) — every File is held in memory and the fetch must beat
// the tap's activation window. `sharedUrls` are the ones that made it into the
// sheet; `failedUrls` failed to fetch and can be retried by the caller.
export async function sharePhotos(
  urls: string[],
  opts: { onProgress?: (p: ShareProgress) => void } = {},
): Promise<SharePhotosResult> {
  if (!canShareFiles()) {
    return { status: "unsupported", sharedUrls: [], failedUrls: urls };
  }
  const { files, sharedUrls, failedUrls } = await fetchFiles(urls, opts.onProgress);
  if (files.length === 0) {
    // Nothing fetched (all failed / bad URLs) — let the caller fall back.
    return { status: "error", sharedUrls: [], failedUrls: urls };
  }
  if (!navigator.canShare({ files })) {
    return { status: "unsupported", sharedUrls: [], failedUrls: urls };
  }
  try {
    await navigator.share({ files });
    return { status: "shared", sharedUrls, failedUrls };
  } catch (err) {
    // A dismissed sheet (AbortError) counts as handled; anything else — most
    // often NotAllowedError when a slow fetch outlived the gesture — is a real
    // failure and the caller should offer the ZIP instead.
    if (err instanceof DOMException && err.name === "AbortError") {
      return { status: "shared", sharedUrls, failedUrls };
    }
    return { status: "error", sharedUrls: [], failedUrls: urls };
  }
}

function canShareFiles(): boolean {
  return (
    typeof navigator !== "undefined" &&
    typeof navigator.canShare === "function" &&
    typeof navigator.share === "function"
  );
}

// Fetches `urls` into Files with a fixed concurrency, reporting progress as each
// resolves. Failures (after per-file retries) are collected, not thrown, so one
// dead URL never sinks the whole batch.
async function fetchFiles(
  urls: string[],
  onProgress?: (p: ShareProgress) => void,
): Promise<{ files: File[]; sharedUrls: string[]; failedUrls: string[] }> {
  const files: File[] = [];
  const sharedUrls: string[] = [];
  const failedUrls: string[] = [];
  let done = 0;
  let cursor = 0;

  async function worker() {
    while (cursor < urls.length) {
      const i = cursor++;
      const url = urls[i];
      try {
        files.push(await fetchAsFile(url, i));
        sharedUrls.push(url);
      } catch {
        failedUrls.push(url);
      }
      done += 1;
      onProgress?.({ done, total: urls.length });
    }
  }

  const workers = Array.from(
    { length: Math.min(FETCH_CONCURRENCY, urls.length) },
    worker,
  );
  await Promise.all(workers);
  return { files, sharedUrls, failedUrls };
}

// Fetches a signed download URL into a named File, retrying transient failures
// (network / non-ok / 429) with backoff. Filename comes from the response's
// Content-Disposition (CORS-exposed) so the sheet shows the real name.
async function fetchAsFile(url: string, index: number): Promise<File> {
  const href = safeHttpUrl(url);
  if (!href) throw new Error("unsafe url");
  let lastErr: unknown;
  for (let attempt = 0; attempt <= FETCH_RETRIES; attempt++) {
    try {
      const res = await fetch(href);
      if (res.status === 429 && attempt < FETCH_RETRIES) {
        await sleep(retryAfterMs(res, attempt));
        continue;
      }
      if (!res.ok) throw new Error(`status ${res.status}`);
      const blob = await res.blob();
      return new File([blob], filenameFrom(res, index), {
        type: blob.type || "image/jpeg",
      });
    } catch (err) {
      lastErr = err;
      if (attempt < FETCH_RETRIES) await sleep(backoffMs(attempt));
    }
  }
  throw lastErr ?? new Error("fetch failed");
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function backoffMs(attempt: number): number {
  return Math.min(300 * 2 ** attempt, 3000);
}

function retryAfterMs(res: Response, attempt: number): number {
  const header = res.headers?.get?.("retry-after");
  const secs = header ? Number(header) : NaN;
  if (Number.isFinite(secs) && secs >= 0) return Math.min(secs * 1000, 5000);
  return backoffMs(attempt);
}

function filenameFrom(res: Response, index: number): string {
  const cd = res.headers?.get?.("content-disposition") ?? "";
  const encoded = /filename\*=UTF-8''([^;]+)/i.exec(cd);
  if (encoded) {
    try {
      return decodeURIComponent(encoded[1]);
    } catch {
      /* fall through to the plain form */
    }
  }
  const plain = /filename="?([^";]+)"?/i.exec(cd);
  if (plain) return plain[1];
  return `quickpitik-photo-${index + 1}.jpg`;
}

// Shown when an in-app webview can't save files itself — the runner has to open
// a real browser first.
export function showInAppSaveHint(): void {
  useToastStore.getState().showToast({
    kind: "info",
    message:
      "Can't save inside this app's browser. Tap ⋯ (top-right) → Open in Chrome or Safari, then save your photos.",
    duration: 8000,
  });
}
