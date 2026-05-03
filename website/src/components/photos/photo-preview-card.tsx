"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

const TONE_COLORS = [
  "var(--ink)",
  "var(--ink-soft)",
  "var(--slate)",
  "var(--slate-soft)",
];

export interface PhotoPreviewItem {
  id: string;
  bib: string | null;
  time: string;
  tone: number;
  price: number;
  span?: "default" | "wide";
  imageUrl?: string | null;
  alt?: string;
}

interface PhotoPreviewCardProps {
  photo: PhotoPreviewItem;
  eventName: string;
  index: number;
  total: number;
  inCart: boolean;
  onClose: () => void;
  onPrev?: () => void;
  onNext?: () => void;
  onToggleCart: () => void;
}

export function PhotoPreviewCard({
  photo,
  eventName,
  index,
  total,
  inCart,
  onClose,
  onPrev,
  onNext,
  onToggleCart,
}: PhotoPreviewCardProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft" && onPrev) onPrev();
      else if (e.key === "ArrowRight" && onNext) onNext();
    };
    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [onClose, onPrev, onNext]);

  const colorIdx = photo.tone % TONE_COLORS.length;
  const wide = photo.span === "wide";
  const hasImage = Boolean(photo.imageUrl);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);

  useEffect(() => {
    setImageLoaded(false);
    setImageFailed(false);
  }, [photo.id, photo.imageUrl]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={
        photo.bib ? `Preview photo ${photo.bib}` : "Preview untagged photo"
      }
      className="fixed inset-0 z-50 flex items-center justify-center px-3 py-3 sm:px-6 sm:py-6 md:p-10"
    >
      <button
        type="button"
        onClick={onClose}
        aria-label="Close preview"
        className="absolute inset-0 bg-ink/85 backdrop-blur-md cursor-default"
        style={{ animation: "fade-in 0.25s ease-out both" }}
      />

      {onPrev && (
        <button
          type="button"
          onClick={onPrev}
          aria-label="Previous photo"
          className="hidden md:flex absolute left-4 lg:left-8 top-1/2 -translate-y-1/2 z-10 size-12 rounded-full bg-bone/10 hover:bg-bone/20 backdrop-blur-sm text-bone items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh"
        >
          <svg
            viewBox="0 0 24 24"
            className="size-5"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M15 18 L9 12 L15 6"
              stroke="currentColor"
              strokeWidth="1.75"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      )}
      {onNext && (
        <button
          type="button"
          onClick={onNext}
          aria-label="Next photo"
          className="hidden md:flex absolute right-4 lg:right-8 top-1/2 -translate-y-1/2 z-10 size-12 rounded-full bg-bone/10 hover:bg-bone/20 backdrop-blur-sm text-bone items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh"
        >
          <svg
            viewBox="0 0 24 24"
            className="size-5"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M9 6 L15 12 L9 18"
              stroke="currentColor"
              strokeWidth="1.75"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      )}

      <div
        className={cn(
          "relative w-full h-[92vh] flex flex-col overflow-hidden rounded-2xl bg-bone shadow-[0_30px_80px_-20px_rgba(0,0,0,0.6)]",
          wide ? "max-w-5xl" : "max-w-lg sm:max-w-xl",
        )}
        style={{ animation: "fade-up 0.4s ease-out both" }}
      >
        <div className="flex items-center justify-between gap-3 px-5 md:px-7 py-4 border-b border-line">
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate truncate">
            <span className="text-ink">{eventName}</span>
            <span className="mx-2 text-slate-soft" aria-hidden="true">
              ·
            </span>
            <span className="tnum">{photo.time}</span>
          </p>
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft tnum hidden sm:block whitespace-nowrap">
            <span className="text-ink">{index}</span> / {total}
          </p>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close preview"
            className="size-9 shrink-0 rounded-full border border-line text-ink hover:bg-bone-deep flex items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh"
          >
            <svg
              viewBox="0 0 16 16"
              className="size-3.5"
              fill="none"
              aria-hidden="true"
            >
              <path
                d="M3 3 L13 13 M13 3 L3 13"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </div>

        <div
          key={photo.id}
          className="relative flex-1 min-h-0 overflow-hidden"
          style={{
            backgroundColor: TONE_COLORS[colorIdx],
            animation: "fade-in 0.4s ease-out both",
          }}
        >
          <div
            aria-hidden="true"
            className="absolute inset-0 pointer-events-none overflow-hidden select-none"
          >
            <span
              className={cn(
                "absolute inset-0 flex items-center justify-center font-mono uppercase tracking-[0.4em] text-2xl sm:text-3xl md:text-5xl rotate-[-18deg] whitespace-nowrap transition-colors duration-300",
                hasImage && imageLoaded && !imageFailed
                  ? "text-bone/25 mix-blend-overlay"
                  : "text-bone/15",
              )}
            >
              QuickPitik · Preview
            </span>
            {hasImage && imageLoaded && !imageFailed && (
              <div className="absolute inset-0 grid grid-cols-2 grid-rows-3 gap-0">
                {Array.from({ length: 6 }).map((_, i) => (
                  <span
                    key={i}
                    className="flex items-center justify-center font-mono uppercase tracking-[0.4em] text-[10px] sm:text-xs text-bone/30 mix-blend-overlay rotate-[-18deg] whitespace-nowrap"
                  >
                    QuickPitik
                  </span>
                ))}
              </div>
            )}
          </div>

          {hasImage && !imageFailed && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={photo.imageUrl ?? ""}
              alt={
                photo.alt ??
                (photo.bib
                  ? `Race photo of bib ${photo.bib}`
                  : "Untagged race photo")
              }
              onLoad={() => setImageLoaded(true)}
              onError={() => setImageFailed(true)}
              className={cn(
                "absolute inset-0 w-full h-full object-contain transition-opacity duration-500",
                imageLoaded ? "opacity-100" : "opacity-0",
              )}
              draggable={false}
            />
          )}

          {inCart && (
            <div className="absolute top-4 right-4 inline-flex items-center gap-2 bg-fresh text-bone rounded-full px-3 py-1 font-mono uppercase tracking-[0.25em] text-[10px] z-10">
              <span
                className="size-1.5 rounded-full bg-bone"
                aria-hidden="true"
              />
              In cart
            </div>
          )}

          <div className="absolute bottom-0 inset-x-0 z-10 px-4 py-3 flex items-end justify-between gap-3 text-bone/85 bg-gradient-to-t from-ink/50 to-transparent pointer-events-none">
            <span className="font-mono uppercase tracking-[0.3em] text-[10px] tnum">
              {photo.time}
            </span>
            <span className="font-mono uppercase tracking-[0.3em] text-[10px]">
              Watermarked preview
            </span>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row items-stretch sm:items-end sm:justify-between gap-4 px-5 md:px-7 py-5 md:py-6 bg-bone-deep border-t border-line">
          <div>
            <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate mb-1.5">
              {photo.bib ? "Bib" : "Untagged"}
            </p>
            <p className="font-display text-3xl md:text-4xl font-medium text-ink tracking-tight leading-none tnum">
              {photo.bib ?? "—"}
            </p>
          </div>
          <div className="flex flex-col sm:items-end gap-2.5">
            <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
              <span className="tnum text-ink">₱{photo.price}</span>
              <span className="mx-2 text-slate-soft" aria-hidden="true">
                ·
              </span>
              pay once, download forever
            </p>
            <button
              type="button"
              onClick={onToggleCart}
              aria-pressed={inCart}
              className={cn(
                "inline-flex items-center justify-center px-6 py-3 rounded-full font-mono uppercase tracking-[0.2em] text-xs transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone-deep",
                inCart
                  ? "border border-line bg-bone hover:bg-bone-deep text-ink"
                  : "bg-fresh hover:bg-fresh-deep text-bone",
              )}
            >
              {inCart ? "Remove from cart" : "Add to cart →"}
            </button>
          </div>
        </div>

        {(onPrev || onNext) && (
          <div className="md:hidden flex items-center justify-between gap-3 px-5 py-3 border-t border-line bg-bone">
            <button
              type="button"
              onClick={onPrev}
              disabled={!onPrev}
              aria-label="Previous photo"
              className="flex-1 inline-flex items-center justify-center gap-2 font-mono uppercase tracking-[0.25em] text-[10px] text-slate hover:text-ink py-2 disabled:opacity-30 disabled:hover:text-slate transition-colors"
            >
              <span aria-hidden="true">←</span> Prev
            </button>
            <span className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft tnum whitespace-nowrap">
              <span className="text-ink">{index}</span> / {total}
            </span>
            <button
              type="button"
              onClick={onNext}
              disabled={!onNext}
              aria-label="Next photo"
              className="flex-1 inline-flex items-center justify-center gap-2 font-mono uppercase tracking-[0.25em] text-[10px] text-slate hover:text-ink py-2 disabled:opacity-30 disabled:hover:text-slate transition-colors"
            >
              Next <span aria-hidden="true">→</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
