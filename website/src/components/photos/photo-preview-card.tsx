"use client";

import { useEffect, useState, type SyntheticEvent } from "react";
import { createPortal } from "react-dom";
import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ZoomableImage } from "@/components/photos/zoomable-image";
import { useScrollLock } from "@/lib/scroll-lock";
import { formatRaceDate } from "@/lib/format";
import { useToast } from "@/hooks/use-toast";
import { cn, copyToClipboard, formatPrice } from "@/lib/utils";

// The lightbox is a photograph on a dark stage with one caption rail beside
// it. Every fact and every action lives in that rail, top to bottom: event ·
// credit · price · CTAs · hints. From `lg` up the rail is a fixed column to
// the right of the photo. Below `lg` it collapses into a bottom strip and the
// chrome floats over the photo: counter top-left, close top-right, prev/next
// at the vertical centre where a thumb can reach. A portrait photo sizes the
// stage to its own aspect so it fills edge to edge; a landscape photo keeps
// the full-height stage and its letterbox.

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
  // Photographer coupon (V45), priced by the BE: `couponPrice` is what the
  // runner pays with the code. Absent when the photographer has no live
  // coupon or the photo is free.
  couponCode?: string | null;
  couponPercentOff?: number | null;
  couponPrice?: number | null;
  alt?: string;
}

interface BasePhotoPreviewProps {
  photo: PhotoPreviewItem;
  eventName: string;
  /** Event date as YYYY-MM-DD. Shown in the `lg` header row beside the counter;
   *  surfaces without it (cart, orders) simply render no date. */
  eventDate?: string;
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

const NAV_PILL =
  "inline-flex items-center justify-center rounded-full bg-surface/90 hover:bg-surface text-ink shadow-[0_4px_16px_-4px_rgba(0,0,0,0.4)] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh disabled:opacity-50 disabled:cursor-not-allowed";

function Chevron({ dir }: { dir: "left" | "right" }) {
  return (
    <svg viewBox="0 0 24 24" className="size-5" fill="none" aria-hidden="true">
      <path
        d={dir === "left" ? "M15 18 L9 12 L15 6" : "M9 6 L15 12 L9 18"}
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function CloseGlyph() {
  return (
    <svg viewBox="0 0 16 16" className="size-3.5" fill="none" aria-hidden="true">
      <path
        d="M3 3 L13 13 M13 3 L3 13"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function PhotoPreviewCard(props: PhotoPreviewCardProps) {
  const {
    photo,
    eventName,
    eventDate,
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

  // Owned-mode photos get the clean original from BE; everyone else sees the
  // watermarked thumbnail. Closes G-2.
  const renderedSrc = photo.cleanUrl ?? photo.imageUrl ?? null;
  const hasImage = Boolean(renderedSrc);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);
  // Natural width / height of the served image, known once it loads.
  const [aspect, setAspect] = useState<number | null>(null);

  useEffect(() => {
    setImageLoaded(false);
    setImageFailed(false);
    setAspect(null);
  }, [photo.id, renderedSrc]);

  if (!mounted) return null;

  const inCart =
    props.mode !== "owned" && props.mode !== "review" && props.inCart;
  const hasNav = Boolean(onPrev || onNext) || total > 1;
  const dateLabel = eventDate ? formatRaceDate(eventDate) : null;
  // Below lg only: a portrait photo sizes the stage to itself (no letterbox);
  // landscape and not-yet-loaded keep the full-height stage.
  const fitPortrait = aspect !== null && aspect < 1;

  const counter = hasNav && (
    <Kicker as="span" tone="soft" tnum className="whitespace-nowrap">
      <span className="text-ink">{index}</span> / {total}
    </Kicker>
  );

  const credit = showPhotographerCredit && (
    <PhotographerCredit
      handle={photo.photographerHandle}
      name={photo.photographerName}
    />
  );
  // The photographer's coupon, priced by the BE. Browse mode only — an owned
  // or review photo has nothing left to discount.
  const offer =
    mode === "browse" && photo.couponCode && photo.couponPercentOff != null ? (
      <CouponOffer
        code={photo.couponCode}
        percentOff={photo.couponPercentOff}
        handle={photo.photographerHandle}
        name={photo.photographerName}
      />
    ) : null;
  const hasCouponPrice = mode === "browse" && photo.couponPrice != null && Boolean(photo.couponCode);

  const content = (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={
        photo.bib ? `Preview photo ${photo.bib}` : "Preview untagged photo"
      }
      className="fixed inset-0 z-50 flex items-center justify-center md:p-8 lg:p-10"
    >
      <button
        type="button"
        onClick={onClose}
        aria-label="Close preview"
        className="absolute inset-0 bg-ink/85 backdrop-blur-md cursor-default"
        style={{ animation: "fade-in 0.25s ease-out both" }}
      />

      {/* lg prev / next — outside the card, centred on the stage. */}
      {onPrev && (
        <button
          type="button"
          onClick={onPrev}
          aria-label="Previous photo"
          className={cn(NAV_PILL, "hidden lg:flex absolute left-8 top-1/2 -translate-y-1/2 z-10 size-12")}
        >
          <Chevron dir="left" />
        </button>
      )}
      {onNext && (
        <button
          type="button"
          onClick={onNext}
          aria-label="Next photo"
          className={cn(NAV_PILL, "hidden lg:flex absolute right-8 top-1/2 -translate-y-1/2 z-10 size-12")}
        >
          <Chevron dir="right" />
        </button>
      )}

      <div
        className={cn(
          "relative flex w-full md:max-w-3xl lg:h-[90dvh] lg:max-w-6xl flex-col lg:flex-row overflow-hidden md:rounded-2xl bg-bone shadow-[0_30px_80px_-20px_rgba(0,0,0,0.6)]",
          fitPortrait
            ? "max-h-[100dvh] md:max-h-[90dvh]"
            : "h-[100dvh] md:h-[90dvh]",
        )}
        style={{ animation: "fade-up 0.4s ease-out both" }}
      >
        {/* ── Stage ─────────────────────────────────────────────────── */}
        <div
          key={photo.id}
          className={cn(
            "relative min-h-0 min-w-0 overflow-hidden bg-ink lg:aspect-auto lg:flex-1",
            fitPortrait ? "shrink aspect-(--photo-aspect)" : "flex-1",
          )}
          style={{
            ["--photo-aspect" as string]: String(aspect ?? 1),
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
              onLoad={(e: SyntheticEvent<HTMLImageElement>) => {
                const img = e.currentTarget;
                if (img.naturalWidth && img.naturalHeight) {
                  setAspect(img.naturalWidth / img.naturalHeight);
                }
                setImageLoaded(true);
              }}
              onError={() => setImageFailed(true)}
            />
          )}

          {/* Below-lg chrome floats over the photo: counter top-left, close
              top-right, prev/next at the vertical centre for the thumb. */}
          <div className="lg:hidden absolute inset-x-3 top-3 flex items-center justify-between z-10 pointer-events-none">
            {hasNav ? (
              <span className="pointer-events-auto inline-flex items-center h-11 px-4 rounded-full bg-surface/90 shadow-[0_4px_16px_-4px_rgba(0,0,0,0.4)]">
                {counter}
              </span>
            ) : (
              <span />
            )}
            <button
              type="button"
              onClick={onClose}
              aria-label="Close preview"
              className={cn(NAV_PILL, "pointer-events-auto size-11")}
            >
              <CloseGlyph />
            </button>
          </div>
          {onPrev && (
            <button
              type="button"
              onClick={onPrev}
              aria-label="Previous photo"
              className={cn(NAV_PILL, "lg:hidden absolute left-3 top-1/2 -translate-y-1/2 z-10 size-11")}
            >
              <Chevron dir="left" />
            </button>
          )}
          {onNext && (
            <button
              type="button"
              onClick={onNext}
              aria-label="Next photo"
              className={cn(NAV_PILL, "lg:hidden absolute right-3 top-1/2 -translate-y-1/2 z-10 size-11")}
            >
              <Chevron dir="right" />
            </button>
          )}

          {inCart && (
            <div className="hidden lg:inline-flex absolute top-4 left-4 items-center gap-2 bg-fresh text-surface rounded-full px-3 py-1 font-mono uppercase tracking-[0.14em] text-[13px] z-10">
              <span className="size-1.5 rounded-full bg-surface" aria-hidden="true" />
              In cart
            </div>
          )}
        </div>

        {/* ── Rail ──────────────────────────────────────────────────── */}
        <aside className="flex flex-col shrink-0 lg:w-[340px] bg-bone border-t lg:border-t-0 lg:border-l border-line">
          {/* lg header: date on the left, counter + close on the right. */}
          <div className="hidden lg:flex items-center justify-between gap-4 px-6 py-4 border-b border-line">
            {dateLabel ? (
              <Kicker as="span" tone="soft" tnum className="whitespace-nowrap">
                {dateLabel}
              </Kicker>
            ) : (
              <span />
            )}
            <div className="flex items-center gap-4">
              {counter}
              <button
                type="button"
                onClick={onClose}
                aria-label="Close preview"
                className="size-9 shrink-0 rounded-full border border-line text-ink hover:bg-bone-deep flex items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh"
              >
                <CloseGlyph />
              </button>
            </div>
          </div>

          <div className="px-5 pt-4 pb-3 md:px-6 lg:pt-6 lg:pb-0 lg:flex-1 lg:min-h-0">
            {/* Below lg: event name with the price beside it, then the credit. */}
            <div className="lg:hidden">
              <div className="flex items-start justify-between gap-4">
                <Kicker as="p" className="min-w-0 flex-1 text-ink line-clamp-2">
                  {eventName}
                </Kicker>
                {mode === "browse" && (
                  <div className="shrink-0 text-right">
                    <p className="font-mono font-semibold tnum text-[22px] leading-none text-ink">
                      {formatPrice(hasCouponPrice ? photo.couponPrice! : photo.price)}
                    </p>
                    {hasCouponPrice && (
                      <Kicker as="p" tone="soft" tnum className="mt-1 whitespace-nowrap">
                        list {formatPrice(photo.price)}
                      </Kicker>
                    )}
                  </div>
                )}
                {mode === "owned" && (
                  <Kicker as="p" tone="soft" className="shrink-0">
                    Yours to keep
                  </Kicker>
                )}
              </div>
              {credit && <div className="mt-1.5">{credit}</div>}
              {offer}
            </div>

            {/* lg rail: event name, credit. (Date lives in the header row.) */}
            <div className="hidden lg:block">
              <p className="font-display font-extrabold text-[26px] leading-tight tracking-tight text-ink">
                {eventName}
              </p>
              {credit && <div className="mt-5">{credit}</div>}
              {offer}
            </div>
          </div>

          {/* Actions — anchored to the rail bottom on lg. */}
          <div className="px-5 pb-4 md:px-6 md:pb-5 lg:pb-6 lg:pt-5 lg:border-t lg:border-line">
            {mode === "browse" && (
              <div className="hidden lg:block mb-4">
                <p className="font-mono font-semibold tnum text-[26px] leading-none text-ink">
                  {formatPrice(hasCouponPrice ? photo.couponPrice! : photo.price)}
                </p>
                <Kicker as="p" tone="soft" tnum className="mt-1.5">
                  {hasCouponPrice
                    ? `With ${photo.couponCode} · list ${formatPrice(photo.price)}`
                    : "Per photo · download forever"}
                </Kicker>
              </div>
            )}

            {props.mode === "owned" ? (
              <>
                <Kicker as="p" tone="soft" className="hidden lg:block mb-3">
                  Yours to keep
                </Kicker>
                <button
                  type="button"
                  onClick={props.onDownload}
                  className={cn(BTN_PRIMARY, BTN_SIZE.sm, "w-full")}
                >
                  Download photo
                  <span aria-hidden="true">↓</span>
                </button>
              </>
            ) : props.mode === "review" ? (
              <Kicker as="p" tone="soft" className="text-center lg:text-left">
                {props.footerLabel ?? "Admin · Review only"}
              </Kicker>
            ) : (
              <>
                <div className="flex flex-row lg:flex-col-reverse gap-2 lg:gap-2.5">
                  <button
                    type="button"
                    onClick={props.onToggleCart}
                    aria-pressed={props.inCart}
                    className={cn(BTN_SECONDARY, BTN_SIZE.sm, "flex-1 lg:w-full")}
                  >
                    {props.inCart ? "− Remove" : "+ Add to cart"}
                  </button>
                  <button
                    type="button"
                    onClick={props.onBuyNow}
                    className={cn(BTN_PRIMARY, BTN_SIZE.sm, "flex-1 lg:w-full")}
                  >
                    {props.inCart ? "Checkout" : "Buy now"}
                    <span aria-hidden="true">→</span>
                  </button>
                </div>
                {props.inCart && (
                  <Kicker as="p" tone="active" className="mt-3 text-center lg:text-left">
                    <span aria-hidden="true">✓</span> In cart
                    {props.onViewCart && (
                      <>
                        <span className="mx-2 text-slate-soft" aria-hidden="true">·</span>
                        <button
                          type="button"
                          onClick={props.onViewCart}
                          className="underline underline-offset-4 decoration-line hover:text-fresh-deep hover:decoration-fresh-deep transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh rounded-sm"
                        >
                          view cart →
                        </button>
                      </>
                    )}
                  </Kicker>
                )}
              </>
            )}

            {/* Gesture hints — invisible gestures don't ship. */}
            {hasImage && (
              <p className="mt-3 lg:mt-4 font-mono uppercase tracking-[0.14em] text-[12px] text-slate-soft text-center lg:text-left">
                <span className="lg:hidden">Pinch or double-tap to zoom</span>
                <span className="hidden lg:inline">
                  {hasNav && "← → navigate · "}Esc close · double-click to zoom
                </span>
              </p>
            )}
          </div>
        </aside>
      </div>
    </div>
  );

  return createPortal(content, document.body);
}

// Photo credit line in the rail.
//
// Both branches are real BE states, not a loading fallback: a photographer's
// handle is only minted at verification, so an approved-but-unverified
// photographer has a name and no public profile to link to. Linking anyway
// would send the runner to /{null}. Legacy rows carry neither field and get
// no credit at all.
//
// Stays off `fresh` deliberately — the Buy-now CTA below owns the one accent
// this viewport is allowed.
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
    // overflows the 375px strip without it. See vault notes/ui-pitfalls
    // 2026-05-06 "mono-uppercase chip wrapped mid-word at 375px".
    <Kicker as="p" tone="soft" className="min-w-0 truncate">
      Photo by{" "}
      {handle ? (
        <Link
          href={`/${handle}`}
          className="text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone rounded-sm"
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

// The photographer's coupon offer in the rail: the code, whose photos it
// covers, and a one-tap copy. Ink outline, not fresh — Buy now owns the
// accent. The BE has already priced it (see couponPrice on the item).
function CouponOffer({
  code,
  percentOff,
  handle,
  name,
}: {
  code: string;
  percentOff: number;
  handle?: string | null;
  name?: string | null;
}) {
  const { showToast } = useToast();
  // The credit line directly above already names the photographer; a full
  // handle here truncates on the 340px rail. Keep the row about the offer.
  const who = handle || name ? "their" : "this photographer's";

  const copy = async () => {
    const ok = await copyToClipboard(code);
    showToast(
      ok
        ? { kind: "success", message: `Code ${code} copied. Paste it at checkout.` }
        : { kind: "error", message: "Couldn't copy the code." },
    );
  };

  return (
    <div className="mt-3 flex items-center justify-between gap-3 rounded-xl border border-line bg-bone-deep/60 px-4 py-3">
      <div className="min-w-0">
        <p className="font-mono font-semibold tnum text-ink truncate">{code}</p>
        <Kicker as="p" tone="soft" tnum className="leading-snug">
          {percentOff}% off {who} photos
        </Kicker>
      </div>
      <button
        type="button"
        onClick={() => void copy()}
        className={cn(BTN_SECONDARY, "shrink-0 px-4 py-2 text-sm min-h-[44px]")}
      >
        Copy
      </button>
    </div>
  );
}
