import { isIOS, isInAppBrowser } from "@/lib/in-app-browser";
import { safeHttpUrl, triggerDownload } from "@/lib/utils";
import { useToastStore } from "@/store/toast-store";

export type SaveResult = "downloaded" | "shared" | "hint";

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

// Fetches the photo and hands it to the OS share sheet. Returns true when the
// sheet handled it (a user-cancelled sheet counts — that's a choice, not a
// failure), false when file sharing isn't available so the caller can fall back.
async function shareFile(href: string): Promise<boolean> {
  if (
    typeof navigator === "undefined" ||
    typeof navigator.canShare !== "function" ||
    typeof navigator.share !== "function"
  ) {
    return false;
  }
  try {
    const res = await fetch(href);
    if (!res.ok) return false;
    const blob = await res.blob();
    const file = new File([blob], "quickpitik-photo.jpg", {
      type: blob.type || "image/jpeg",
    });
    if (!navigator.canShare({ files: [file] })) return false;
    await navigator.share({ files: [file] });
    return true;
  } catch (err) {
    // AbortError = the runner closed the share sheet; that isn't a failure.
    return err instanceof DOMException && err.name === "AbortError";
  }
}

// Shown when an in-app webview can't save a file itself. Also used directly by
// the "Download all" ZIP buttons, which are too heavy to pull into memory for
// a share.
export function showInAppSaveHint(): void {
  useToastStore.getState().showToast({
    kind: "info",
    message:
      "Can't save inside this app's browser. Tap ⋯ (top-right) → Open in Chrome or Safari, then save the photo.",
    duration: 8000,
  });
}
