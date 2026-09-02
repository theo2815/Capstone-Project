"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { ZoomableImage } from "@/components/photos/zoomable-image";
import { useScrollLock } from "@/lib/scroll-lock";
import { cn, formatPrice } from "@/lib/utils";

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
  // Attribution from the BE. A null `photographerHandle` on a present name
  // means the photographer isn't verified yet and has no public profile —
  // the credit renders as plain text, never a link to /{null}.
  photographerHandle?: string | null;
  photographerName?: string | null;
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
  /**
   * Set false where the credit would be noise because the surrounding page is
   * already the photographer's — e.g. their own public gallery. Defaults true;
   * surfaces whose data simply carries no attribution render nothing anyway.
   */
  showPhotographerCredit?: boolean;
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
  const {
    photo,
    eventName,
    index,
    total,
    onClose,
    onPrev,
    onNext,
    showPhotographerCredit = true,
  } = props;
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
          className="hidden md:flex absolute left-4 lg:left-8 top-1/2 -translate-y-1/2 z-10 size-12 rounded-full bg-surface/90 hover:bg-surface text-ink shadow-[0_4px_16px_-4px_rgba(0,0,0,0.4)] items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh"
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
          className="hidden md:flex absolute right-4 lg:right-8 top-1/2 -translate-y-1/2 z-10 size-12 rounded-full bg-surface/90 hover:bg-surface text-ink shadow-[0_4px_16px_-4px_rgba(0,0,0,0.4)] items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh"
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
          <p className="font-mono uppercase tracking-[0.18em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate truncate">
            <span className="text-ink">{eventName}</span>
          </p>
          <p className="font-mono uppercase tracking-[0.18em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft tnum hidden sm:block whitespace-nowrap">
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
          {/* No client-side platform mark: the backend bakes the QuickPitik
              credit tiles + caption + photographer logo into imageUrl, and
              cleanUrl (owned) is deliberately unmarked. */}

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
            // Pinch / double-click zoom over whatever was served: the
            // watermarked preview, or the clean original for an owner.
            <ZoomableImage
              src={renderedSrc ?? ""}
              alt={
                photo.alt ??
                (photo.bib
                  ? `Race photo of bib ${photo.bib}`
                  : "Untagged race photo")
              }
              loaded={imageLoaded}
              onLoad={() => setImageLoaded(true)}
              onError={() => setImageFailed(true)}
            />
          )}

          {props.mode !== "owned" && props.mode !== "review" && props.inCart && (
            <div className="absolute top-4 right-4 inline-flex items-center gap-2 bg-fresh text-surface rounded-full px-3 py-1 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] z-10">
              <span
                className="size-1.5 rounded-full bg-surface"
                aria-hidden="true"
              />
              In cart
            </div>
          )}

        </div>

        <div className="px-5 md:px-7 py-4 sm:py-5 md:py-6 bg-bone-deep border-t border-line">
          {showPhotographerCredit && (
            <PhotographerCredit
              handle={photo.photographerHandle}
              name={photo.photographerName}
            />
          )}
          {props.mode === "owned" ? (
            <>
              <p className="font-mono uppercase tracking-[0.18em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft mb-3 sm:mb-4 text-center">
                Yours to keep
              </p>
              <button
                type="button"
                onClick={props.onDownload}
                className="w-full inline-flex items-center justify-center gap-2 px-3 sm:px-6 py-2.5 sm:py-3 rounded-full font-display font-bold text-[15px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone-deep whitespace-nowrap bg-fresh hover:bg-fresh-deep text-surface"
              >
                Download photo
                <span aria-hidden="true">↓</span>
              </button>
            </>
          ) : props.mode === "review" ? (
            <p className="font-mono uppercase tracking-[0.18em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft text-center">
              {props.footerLabel ?? "Admin · Review only"}
            </p>
          ) : (
            <>
              {!props.inCart && (
                <p className="font-mono uppercase tracking-[0.18em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft mb-3 sm:mb-4 text-center">
                  Pay once, download forever
                </p>
              )}
              <div className="flex flex-row gap-2 sm:gap-3">
                <button
                  type="button"
                  onClick={props.onToggleCart}
                  aria-pressed={props.inCart}
                  className={cn(
                    "inline-flex flex-1 items-center justify-center px-3 sm:px-6 py-2.5 sm:py-3 rounded-full font-display font-bold text-[15px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone-deep whitespace-nowrap",
                    "border border-ink bg-bone text-ink hover:bg-ink hover:text-surface",
                  )}
                >
                  {props.inCart ? "− Remove" : "+ Add to cart"}
                </button>
                <button
                  type="button"
                  onClick={props.onBuyNow}
                  className={cn(
                    "inline-flex flex-1 items-center justify-center px-3 sm:px-6 py-2.5 sm:py-3 rounded-full font-display font-bold text-[15px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone-deep whitespace-nowrap",
                    "bg-fresh hover:bg-fresh-deep text-surface",
                  )}
                >
                  {props.inCart ? "Checkout now" : "Buy now"} ·
                  <span className="tnum ml-1 sm:ml-1.5">
                    {formatPrice(photo.price)}
                  </span>
                  <span aria-hidden="true" className="ml-1 sm:ml-1.5">→</span>
                </button>
              </div>
              {props.inCart && (
                <p className="mt-3 sm:mt-4 font-mono uppercase tracking-[0.18em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-fresh text-center">
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
              className="flex-1 inline-flex items-center justify-center gap-2 rounded-full border border-line font-sans text-sm font-medium text-ink hover:bg-bone-deep min-h-[44px] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent transition-colors"
            >
              <span aria-hidden="true">←</span> Prev
            </button>
            <span className="font-mono uppercase tracking-[0.18em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft tnum whitespace-nowrap">
              <span className="text-ink">{index}</span> / {total}
            </span>
            <button
              type="button"
              onClick={onNext}
              disabled={!onNext}
              aria-label="Next photo"
              className="flex-1 inline-flex items-center justify-center gap-2 rounded-full border border-line font-sans text-sm font-medium text-ink hover:bg-bone-deep min-h-[44px] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent transition-colors"
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

// Photo credit line above the footer's CTA block.
//
// Both branches are real BE states, not a loading fallback: a photographer's
// handle is only minted at verification, so an approved-but-unverified
// photographer has a name and no public profile to link to. Linking anyway
// would send the runner to /{null}. Legacy rows carry neither field and get
// no credit at all.
//
// Stays off `fresh` deliberately — the Buy-now CTA a few pixels below owns the
// one accent this viewport is allowed.
function PhotographerCredit({
  handle,
  name,
}: {
  handle?: string | null;
  name?: string | null;
}) {
  if (!handle && !name) return null;

  return (
    // `truncate` is load-bearing, not defensive: mono uppercase at kicker
    // tracking measures roughly double its intuitive width, so a long handle
    // overflows the 375px modal without it. See vault notes/ui-pitfalls
    // 2026-05-06 "mono-uppercase chip wrapped mid-word at 375px".
    <Kicker as="p" tone="soft" className="mb-3 sm:mb-4 text-center truncate">
      Photo by{" "}
      {handle ? (
        <Link
          href={`/${handle}`}
          className="text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone-deep rounded-sm"
        >
          @{handle}
          <span aria-hidden="true" className="ml-1.5 no-underline">
            →
          </span>
        </Link>
      ) : (
        <span className="text-ink">{name}</span>
      )}
    </Kicker>
  );
}
