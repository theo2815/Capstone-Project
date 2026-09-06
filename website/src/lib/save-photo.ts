import { isInAppBrowser } from "@/lib/in-app-browser";
import { safeHttpUrl, triggerDownload } from "@/lib/utils";
import { useToastStore } from "@/store/toast-store";

export type SaveResult = "downloaded" | "shared" | "hint";

// Save a signed photo download URL to the runner's device.
//
// Meta's in-app browsers (Facebook / Messenger / Instagram — WKWebView on iOS)
// render a navigated image inline and expose no "Save to device" control, so
// the cross-origin `<a download>` that saves everywhere else just strands the
// runner on a full-screen photo with no way out. Inside those webviews we fetch
// the bytes and hand them to the native share sheet ("Save Image" / "Save to
// Files"), which saves in place; where file sharing isn't available (older
// devices) we tell the runner how to reopen the photo in a real browser.
//
// Normal browsers are untouched — they keep the plain attachment download. This
// completes the owed leg of `722f9c26` (backend-streamed downloads), which
// fixed the R2/fbclid signature 403 but left the webview "no Save option"
// symptom the FINDINGS screenshots show.
//
// The `fetch` is cross-origin (site → api.quickpitik.com) but CORS-allowed:
// CorsConfig whitelists the site origin on `/**`. `navigator.share` needs a
// secure context (prod is HTTPS) and a user gesture, which the button supplies.
export async function savePhoto(url: string): Promise<SaveResult> {
  const href = safeHttpUrl(url);
  if (!href) return "downloaded"; // safeHttpUrl rejected it — nothing to save

  if (!isInAppBrowser()) {
    triggerDownload(href);
    return "downloaded";
  }

  if (
    typeof navigator !== "undefined" &&
    typeof navigator.canShare === "function" &&
    typeof navigator.share === "function"
  ) {
    try {
      const res = await fetch(href);
      if (res.ok) {
        const blob = await res.blob();
        const file = new File([blob], "quickpitik-photo.jpg", {
          type: blob.type || "image/jpeg",
        });
        if (navigator.canShare({ files: [file] })) {
          await navigator.share({ files: [file] });
          return "shared"; // saved, or the runner dismissed the sheet
        }
      }
    } catch (err) {
      // AbortError = the runner closed the share sheet; that isn't a failure.
      if (err instanceof DOMException && err.name === "AbortError") return "shared";
      // fetch blocked / share rejected → fall through to the hint
    }
  }

  showInAppSaveHint();
  return "hint";
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
