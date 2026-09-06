"use client";

import { useMemo, useState } from "react";
import { cn, triggerDownload } from "@/lib/utils";
import { BTN_PRIMARY, BTN_SECONDARY } from "@/components/ui/button-styles";
import { useToast } from "@/hooks/use-toast";
import { isAndroid, isIOS, isInAppBrowser } from "@/lib/in-app-browser";
import {
  SHARE_BATCH_SIZE,
  SHARE_MAX_TOTAL,
  sharePhotos,
  showInAppSaveHint,
  type ShareProgress,
} from "@/lib/save-photo";

// The master "save every photo in this order" action.
//
// Desktop keeps the server ZIP (one file, right tool). On a phone a ZIP is
// friction — download, find, extract — and on iOS it lands in Files, not
// Photos. So mobile fetches the photos and hands them to the native share
// sheet in batches: iOS "Save N Images" drops them straight in the camera roll.
// Large orders (or browsers without file sharing) fall back to the ZIP.
//
// The share-vs-ZIP choice is made on click, not render, so the first paint is
// platform-agnostic and hydration matches.

export type SavePhotosVariant = "primary" | "secondary";

export function SavePhotosButton({
  photos,
  bundleUrl,
  variant = "primary",
  className,
}: {
  // Per-photo signed download URLs (the clean originals). Empty ⇒ ZIP only.
  photos: ReadonlyArray<{ url: string }>;
  // Server ZIP endpoint — the desktop path and the mobile fallback.
  bundleUrl: string | null;
  variant?: SavePhotosVariant;
  className?: string;
}) {
  const { showToast } = useToast();
  const urls = useMemo(() => photos.map((p) => p.url).filter(Boolean), [photos]);
  const total = urls.length;

  // null = not started. Otherwise the URLs still to save — this doubles as the
  // chunk cursor and the retry queue: a failed fetch stays here, so the next
  // tap picks it up again.
  const [remaining, setRemaining] = useState<string[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<ShareProgress | null>(null);

  const pending = remaining ?? urls;
  const started = remaining !== null;
  const finished = started && pending.length === 0;
  const nextBatch = Math.min(SHARE_BATCH_SIZE, pending.length);

  function downloadZip() {
    if (!bundleUrl) return;
    // In-app webviews can't save a streamed ZIP — steer them to a real browser.
    if (isInAppBrowser()) {
      showInAppSaveHint();
      return;
    }
    triggerDownload(bundleUrl);
    showToast({
      kind: "success",
      message: `Preparing ${total} photo${total === 1 ? "" : "s"}…`,
    });
  }

  async function handleClick() {
    if (busy || finished) return;

    const useShare =
      urls.length > 0 &&
      total <= SHARE_MAX_TOTAL &&
      (isIOS() || isAndroid() || isInAppBrowser());
    if (!useShare) {
      downloadZip();
      return;
    }

    const batch = pending.slice(0, SHARE_BATCH_SIZE);
    setBusy(true);
    setProgress({ done: 0, total: batch.length });
    const result = await sharePhotos(batch, { onProgress: setProgress });
    setBusy(false);
    setProgress(null);

    if (result.status === "unsupported") {
      downloadZip(); // file sharing turned out unavailable
      return;
    }
    if (result.status === "error") {
      // The gesture lapsed or the share failed — the ZIP is the safety net.
      if (bundleUrl && !isInAppBrowser()) {
        triggerDownload(bundleUrl);
        showToast({ kind: "info", message: "Opening the .zip instead." });
      } else {
        showToast({
          kind: "error",
          message: "Couldn't save those. Try again in a moment.",
        });
      }
      return;
    }

    // Shared: drop what reached the sheet; failed fetches stay queued to retry.
    const next = pending.filter((u) => !result.sharedUrls.includes(u));
    setRemaining(next);
    if (result.failedUrls.length > 0) {
      showToast({
        kind: "info",
        message: `${result.failedUrls.length} couldn't load — tap to retry.`,
      });
    } else if (next.length === 0) {
      showToast({
        kind: "success",
        message: `Saved ${total} photo${total === 1 ? "" : "s"}.`,
      });
    }
  }

  const label = (() => {
    if (busy && progress) return `Preparing ${progress.done} of ${progress.total}…`;
    if (finished) return "Saved";
    if (started && pending.length < total) {
      return `Save next ${nextBatch} of ${pending.length}`;
    }
    return total === 1 ? "Save photo" : `Save ${total} photos`;
  })();

  const spinnerTone =
    variant === "primary"
      ? "border-surface/40 border-t-surface"
      : "border-ink/40 border-t-ink";

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={busy || finished}
      aria-live="polite"
      className={cn(
        variant === "primary" ? BTN_PRIMARY : BTN_SECONDARY,
        variant === "primary" ? "w-full px-7 py-4 text-[15px]" : "px-6 py-3 text-base",
        busy && "cursor-wait",
        className,
      )}
    >
      {busy ? (
        <span
          aria-hidden="true"
          className={cn("size-3.5 rounded-full border-2 animate-spin", spinnerTone)}
        />
      ) : (
        <span aria-hidden="true">{finished ? "✓" : "↓"}</span>
      )}
      <span className="tnum">{label}</span>
    </button>
  );
}
