"use client";

import { useRef, useState } from "react";
import { Kicker } from "@/components/ui/kicker";
import { FieldError } from "@/components/ui/field-error";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { BTN_GHOST, BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ApiError } from "@/lib/api";
import { BLURRY_LABELS } from "@/lib/photographer-relevance-copy";
import { usePhotographerRelevancePhotos, useSetRelevancePhotosHidden } from "@/hooks/use-photographer-relevance";
import { cn } from "@/lib/utils";

// The photo library's grid and tile classes (dashboard/events/[id]), at a fixed square tile.
const GRID = "grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6";

// RV-3.4b: the blurry photos still on sale, selectable, with Hide; below them the ones the photographer hid, each with
// Undo. Both lists come from the server and are read again after every answer; nothing moves before it.
export function RelevanceBlurryReview({ caseId }: { caseId: string }) {
  const blurry = usePhotographerRelevancePhotos(caseId, "blurry");
  const hidden = usePhotographerRelevancePhotos(caseId, "hidden");
  const change = useSetRelevancePhotosHidden(caseId);
  const [selected, setSelected] = useState(new Set<string>());
  const [error, setError] = useState<string | null>(null);
  // isPending reaches the buttons only on the next render; this stops a second click in the same tick.
  const sending = useRef(false);
  const busy = change.isPending;
  const chosen = blurry.items.filter((p) => selected.has(p.photoId)).map((p) => p.photoId);

  async function send(hide: boolean, photoIds: string[]) {
    if (sending.current || photoIds.length === 0) return;
    sending.current = true;
    setError(null);
    try {
      await change.mutateAsync({ hide, photoIds });
      if (hide) setSelected(new Set());
    } catch (failure) {
      setError(failure instanceof ApiError ? (failure.errors[0]?.message ?? failure.message) : failure instanceof Error ? failure.message : null);
    } finally {
      sending.current = false;
    }
  }

  const toggle = (id: string) => setSelected((current) => {
    const next = new Set(current);
    if (!next.delete(id)) next.add(id);
    return next;
  });

  return (
    <div className="space-y-10">
      <section aria-label={BLURRY_LABELS.review} className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="font-display text-xl font-extrabold text-ink">{BLURRY_LABELS.review}</h2>
          <div className="flex items-center gap-3">
            {chosen.length > 0 && <Kicker as="p" tone="soft" tnum>{BLURRY_LABELS.selected(chosen.length)}</Kicker>}
            <button type="button" disabled={busy || chosen.length === 0} onClick={() => void send(true, chosen)} className={cn(BTN_PRIMARY, BTN_SIZE.sm)}>
              {busy ? BLURRY_LABELS.hiding : BLURRY_LABELS.hide}
            </button>
          </div>
        </div>
        <FieldError id="relevance-blurry-error" message={error} />
        <ul className={GRID}>
          {blurry.items.map((p, i) => (
            <li key={p.photoId}>
              <Tile>
                <button type="button" aria-pressed={selected.has(p.photoId)} aria-label={BLURRY_LABELS.select(i + 1, selected.has(p.photoId))}
                  disabled={busy} onClick={() => toggle(p.photoId)}
                  className={cn("absolute inset-0 rounded-xl", selected.has(p.photoId) && "ring-2 ring-ink ring-offset-2 ring-offset-bone")}>
                  <Thumb url={p.thumbnailUrl} />
                  <span aria-hidden="true" className={cn("absolute top-2 left-2 size-6 rounded-full border flex items-center justify-center",
                    selected.has(p.photoId) ? "bg-ink border-ink text-surface" : "bg-bone/70 border-line text-transparent backdrop-blur-sm")}>
                    <svg viewBox="0 0 20 20" className="size-3.5" fill="none" stroke="currentColor" strokeWidth="3">
                      <path d="M4 10l4 4 8-9" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </span>
                </button>
              </Tile>
              <p className="mt-1.5 font-sans text-sm text-slate">{BLURRY_LABELS.looksBlurry}</p>
            </li>
          ))}
        </ul>
        <LoadMoreButton shown={blurry.items.length} total={blurry.total} increment={50} onLoadMore={blurry.fetchNextPage} isLoading={blurry.isFetchingNextPage} />
      </section>

      <section aria-label={BLURRY_LABELS.hidden} className="space-y-4">
        <h2 className="font-display text-xl font-extrabold text-ink">{BLURRY_LABELS.hidden}</h2>
        <ul className={GRID}>
          {hidden.items.map((p, i) => (
            <li key={p.photoId}>
              <Tile><Thumb url={p.thumbnailUrl} /></Tile>
              <button type="button" disabled={busy} aria-label={BLURRY_LABELS.undoPhoto(i + 1)} onClick={() => void send(false, [p.photoId])}
                className={cn(BTN_GHOST, BTN_SIZE.sm, "mt-1")}>
                {BLURRY_LABELS.undo}
              </button>
            </li>
          ))}
        </ul>
        <LoadMoreButton shown={hidden.items.length} total={hidden.total} increment={50} onLoadMore={hidden.fetchNextPage} isLoading={hidden.isFetchingNextPage} />
      </section>
    </div>
  );
}

function Tile({ children }: { children: React.ReactNode }) {
  return <div className="relative aspect-square overflow-hidden rounded-xl bg-bone-deep">{children}</div>;
}

// A short-lived link that fails to load hides its image; the tile, and its selection, stay.
function Thumb({ url }: { url: string | null }) {
  const [failed, setFailed] = useState(false);
  if (!url || failed) return null;
  // eslint-disable-next-line @next/next/no-img-element -- presigned link; Next/Image would re-proxy and lose the signature.
  return <img src={url} alt="" loading="lazy" draggable={false} onError={() => setFailed(true)} className="absolute inset-0 w-full h-full object-cover" />;
}
