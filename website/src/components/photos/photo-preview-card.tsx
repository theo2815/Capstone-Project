"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useScrollLock } from "@/lib/scroll-lock";
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
  // BE-provided clean URL — only set when the requester owns the photo.
  // The lightbox prefers it over `imageUrl` so owned-mode (and any owned
  // gallery thumbnail upgraded mid-browse) renders an unwatermarked source.
  cleanUrl?: string | null;
  alt?: string;
}

interface BasePhotoPreviewProps {
  photo: PhotoPreviewItem;
  eventName: string;
  index: number;
  total: number;
  onClose: () => void;
  onPrev?: () => void;
  onNext?: () => void;
}

interface BrowsePhotoPreviewProps extends BasePhotoPreviewProps {
  mode?: "browse";
  inCart: boolean;
  onToggleCart: () => void;
  onBuyNow: () => void;
  onViewCart?: () => void;
}

interface OwnedPhotoPreviewProps extends BasePhotoPreviewProps {
  mode: "owned";
  onDownload: () => void;
}

// "review" — read-only mode for admin disputes (and any future internal
// review surface). No cart, no buy, no download CTA: the admin is judging
// the runner's complaint, not transacting. Footer carries a short kicker
// instead of a button bar.
interface ReviewPhotoPreviewProps extends BasePhotoPreviewProps {
  mode: "review";
  footerLabel?: string;
}

type PhotoPreviewCardProps =
  | BrowsePhotoPreviewProps
  | OwnedPhotoPreviewProps
  | ReviewPhotoPreviewProps;

export function PhotoPreviewCard(props: PhotoPreviewCardProps) {
  const { photo, eventName, index, total, onClose, onPrev, onNext } = props;
  const mode = props.mode ?? "browse";
  useScrollLock(true);

  // Mount through a portal so the modal escapes any ancestor that establishes
  // a containing block (e.g. an `animate-fade-up` ancestor with a non-`none`
  // transform). Without this, `fixed inset-0` would size to the parent column.
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft" && onPrev) onPrev();
      else if (e.key === "ArrowRight" && onNext) onNext();
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose, onPrev, onNext]);

  const colorIdx = photo.tone % TONE_COLORS.length;
  const wide = photo.span === "wide";
  // Owned-mode photos get the clean original from BE; everyone else sees the
  // watermarked thumbnail. Closes G-2.
  const renderedSrc = photo.cleanUrl ?? photo.imageUrl ?? null;
  const hasImage = Boolean(renderedSrc);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);

  useEffect(() => {
    setImageLoaded(false);
    setImageFailed(false);
  }, [photo.id, renderedSrc]);

  if (!mounted) return null;

  const content = (
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
          "relative w-full h-[92dvh] max-h-[92vh] flex flex-col overflow-hidden rounded-2xl bg-bone shadow-[0_30px_80px_-20px_rgba(0,0,0,0.6)]",
          // Modal width is stable per-photo — image fits inside via object-contain
          // instead of the modal dancing to each photo's aspect. Widening at md/lg
          // so portrait shots aren't squeezed into max-w-xl on desktop where the
          // backdrop swallowed ~half the screen.
          wide
            ? "max-w-5xl"
            : "max-w-lg sm:max-w-xl md:max-w-2xl lg:max-w-3xl",
        )}
        style={{ animation: "fade-up 0.4s ease-out both" }}
      >
        <div className="flex items-center justify-between gap-3 px-5 md:px-7 py-4 border-b border-line">
          <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate truncate">
            <span className="text-ink">{eventName}</span>
          </p>
          <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft tnum hidden sm:block whitespace-nowrap">
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
          {mode === "browse" && hasImage && imageLoaded && !imageFailed && !photo.cleanUrl && (
            // Platform watermark — bottom-LEFT of the image area. Pairs with the
            // BE-baked photographer watermark in the bottom-RIGHT of the JPEG
            // (see backend WatermarkService.drawWatermark). Two corner marks:
            // photographer (BE) + QuickPitik (FE). No diagonal/tile overlays.
            //
            // Suppressed when the BE handed us a `cleanUrl` — the requester
            // owns the photo, so the FE-rendered platform mark would defeat
            // the point of serving the unwatermarked original.
            //
            // TODO: replace the `bg-bone/35` square with the QuickPitik logo SVG
            // once the asset lands in /public. Keep the flex row + spacing so
            // the wordmark sits beside the logo at the same baseline.
            <div
              aria-hidden="true"
              className="absolute bottom-3 left-3 sm:bottom-4 sm:left-4 z-10 pointer-events-none select-none flex items-center gap-1.5 mix-blend-overlay"
            >
              <span
                className="size-3.5 rounded-sm bg-bone/35"
                aria-hidden="true"
              />
              <span className="font-mono uppercase tracking-[0.3em] text-[10px] sm:text-[11px] text-bone/55">
                QuickPitik
              </span>
            </div>
          )}

          {(mode === "owned" || mode === "review") && !hasImage && (
            <div
              aria-hidden="true"
              className="absolute inset-0 flex items-center justify-center pointer-events-none"
            >
              <span className="font-mono uppercase tracking-[0.4em] text-sm sm:text-base text-bone/40 tnum">
                {photo.id.replace(/^mock-/, "")}
              </span>
            </div>
          )}

          {hasImage && !imageFailed && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={renderedSrc ?? ""}
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

          {props.mode !== "owned" && props.mode !== "review" && props.inCart && (
            <div className="absolute top-4 right-4 inline-flex items-center gap-2 bg-fresh text-bone rounded-full px-3 py-1 font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] z-10">
              <span
                className="size-1.5 rounded-full bg-bone"
                aria-hidden="true"
              />
              In cart
            </div>
          )}

        </div>

        <div className="px-5 md:px-7 py-4 sm:py-5 md:py-6 bg-bone-deep border-t border-line">
          {props.mode === "owned" ? (
            <>
              <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft mb-3 sm:mb-4 text-center">
                Yours to keep
              </p>
              <button
                type="button"
                onClick={props.onDownload}
                className="w-full inline-flex items-center justify-center gap-2 px-3 sm:px-6 py-2.5 sm:py-3 rounded-full font-mono uppercase tracking-[0.15em] sm:tracking-[0.2em] text-[13px] min-[400px]:text-[14px] md:text-[12px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone-deep whitespace-nowrap bg-fresh hover:bg-fresh-deep text-bone"
              >
                Download photo
                <span aria-hidden="true">↓</span>
              </button>
            </>
          ) : props.mode === "review" ? (
            <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft text-center">
              {props.footerLabel ?? "Admin · Review only"}
            </p>
          ) : (
            <>
              {!props.inCart && (
                <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft mb-3 sm:mb-4 text-center">
                  Pay once, download forever
                </p>
              )}
              <div className="flex flex-row gap-2 sm:gap-3">
                <button
                  type="button"
                  onClick={props.onToggleCart}
                  aria-pressed={props.inCart}
                  className={cn(
                    "inline-flex flex-1 items-center justify-center px-3 sm:px-6 py-2.5 sm:py-3 rounded-full font-mono uppercase tracking-[0.15em] sm:tracking-[0.2em] text-[13px] min-[400px]:text-[14px] md:text-[12px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone-deep whitespace-nowrap",
                    "border border-line bg-bone hover:bg-bone-deep text-ink",
                  )}
                >
                  {props.inCart ? "− Remove" : "+ Add to cart"}
                </button>
                <button
                  type="button"
                  onClick={props.onBuyNow}
                  className={cn(
                    "inline-flex flex-1 items-center justify-center px-3 sm:px-6 py-2.5 sm:py-3 rounded-full font-mono uppercase tracking-[0.15em] sm:tracking-[0.2em] text-[13px] min-[400px]:text-[14px] md:text-[12px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone-deep whitespace-nowrap",
                    "bg-fresh hover:bg-fresh-deep text-bone",
                  )}
                >
                  {props.inCart ? "Checkout now" : "Buy now"} ·
                  <span className="tnum ml-1 sm:ml-1.5">₱{photo.price}</span>
                  <span aria-hidden="true" className="ml-1 sm:ml-1.5">→</span>
                </button>
              </div>
              {props.inCart && (
                <p className="mt-3 sm:mt-4 font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-fresh text-center">
                  <span aria-hidden="true">✓</span> In cart
                  {props.onViewCart && (
                    <>
                      <span
                        className="mx-2 text-slate-soft"
                        aria-hidden="true"
                      >
                        ·
                      </span>
                      <button
                        type="button"
                        onClick={props.onViewCart}
                        className="underline underline-offset-4 decoration-line hover:text-fresh-deep hover:decoration-fresh-deep transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh rounded-sm"
                      >
                        view cart →
                      </button>
                    </>
                  )}
                </p>
              )}
            </>
          )}
        </div>

        {(onPrev || onNext) && (
          <div className="md:hidden flex items-center justify-between gap-3 px-5 py-2 border-t border-line bg-bone">
            <button
              type="button"
              onClick={onPrev}
              disabled={!onPrev}
              aria-label="Previous photo"
              className="flex-1 inline-flex items-center justify-center gap-2 font-mono uppercase tracking-[0.25em] text-[10px] text-slate hover:text-ink py-2 disabled:opacity-30 disabled:hover:text-slate transition-colors"
            >
              <span aria-hidden="true">←</span> Prev
            </button>
            <span className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft tnum whitespace-nowrap">
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

  return createPortal(content, document.body);
}
