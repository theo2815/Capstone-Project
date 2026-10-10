"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode, type SyntheticEvent } from "react";
import { createPortal } from "react-dom";
import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ZoomableImage } from "@/components/photos/zoomable-image";
import { PhotoMetadataDialog } from "@/components/photos/photo-metadata-dialog";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import type { PhotoMetadataAccess } from "@/types/photo-metadata";
import { useScrollLock } from "@/lib/scroll-lock";
import { formatRaceDate } from "@/lib/format";
import { cn, formatPrice } from "@/lib/utils";

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
  // Event coupon, priced by the BE: `couponPrice` is what the runner pays
  // with the code. Absent unless this exact event/photo/photographer is
  // eligible.
  couponCode?: string | null;
  couponPercentOff?: number | null;
  couponPrice?: number | null;
  alt?: string;
}

interface BasePhotoPreviewProps {
  photo: PhotoPreviewItem;
  metadataAccess?: PhotoMetadataAccess;
  eventName: string;
  /** Event date as YYYY-MM-DD. Shown in the `lg` header row beside the counter;
   *  surfaces without it (cart, orders) simply render no date. */
  eventDate?: string;
  index: number;
  total: number;
  onClose: () => void;
  onPrev?: () => void;
  onNext?: () => void;
  details?: ReactNode;
  onRefreshImage?: () => void;
  imageReloadKey?: number;
  isRefreshingImage?: boolean;
  imageRefreshError?: string | null;
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
  downloadDisabled?: boolean;
  /** Replaces the "Yours to keep" kicker — e.g. "Free from @handle" when a
   *  free-event photo is opened by a visitor rather than its owner. */
  footnote?: string;
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
  } = props;
  const mode = props.mode ?? "browse";
  const viewer = usePrivateQuerySession();
  const access = props.metadataAccess;
  const metadataContext = JSON.stringify([photo.id, access, ...viewer.key]);
  const [openMetadataContext, setOpenMetadataContext] = useState<string | null>(null);
  const [deniedMetadataContext, setDeniedMetadataContext] = useState<string | null>(null);
  const metadataTrigger = useRef<HTMLButtonElement>(null);
  const mobileClose = useRef<HTMLButtonElement>(null);
  const desktopClose = useRef<HTMLButtonElement>(null);
  const focusPreviewClose = useCallback(() => {
    const desktop = window.matchMedia?.("(min-width: 1024px)").matches;
    (desktop ? desktopClose.current : mobileClose.current)?.focus();
  }, []);
  const metadataOpen = openMetadataContext === metadataContext;
  const showMetadata = (access === "public" || (access === "owner" && viewer.enabled && viewer.key[1] === "PHOTOGRAPHER")) && deniedMetadataContext !== metadataContext;
  useScrollLock(true);

  useEffect(() => {
    setOpenMetadataContext(null);
    setDeniedMetadataContext(null);
  }, [metadataContext]);

  // Mount through a portal so the modal escapes any ancestor that establishes
  // a containing block (e.g. an `animate-fade-up` ancestor with a non-`none`
  // transform). Without this, `fixed inset-0` would size to the parent column.
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  // Modal focus: start on the dialog itself (a keyed remount after Prev/Next
  // must not land on Close), keep the page behind inert except the toast
  // region preview actions report into, and hand focus back on close.
  // Mount-only so metadata's later portal stays live.
  const dialogRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!mounted) return;
    const trigger = document.activeElement;
    const background = Array.from(document.body.children).filter((el) => el !== dialogRef.current
      && !el.hasAttribute("inert") && !el.matches('[role="region"][aria-label="Notifications"]'));
    background.forEach((el) => el.setAttribute("inert", ""));
    dialogRef.current?.focus();
    return () => {
      background.forEach((el) => el.removeAttribute("inert"));
      if (trigger instanceof HTMLElement) trigger.focus({ preventScroll: true });
    };
  }, [mounted]);

  useEffect(() => {
    if (metadataOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft" && onPrev) onPrev();
      else if (e.key === "ArrowRight" && onNext) onNext();
      else if (e.key === "Tab") {
        const controls = Array.from(dialogRef.current?.querySelectorAll<HTMLElement>('button:not(:disabled):not([tabindex="-1"]), a[href]') ?? [])
          .filter((el) => el.checkVisibility?.() ?? true);
        const first = controls[0];
        const last = controls[controls.length - 1];
        const active = document.activeElement;
        if (!dialogRef.current?.contains(active) || active === dialogRef.current || active === (e.shiftKey ? first : last)) {
          e.preventDefault();
          (e.shiftKey ? last : first)?.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
    };
  }, [metadataOpen, onClose, onPrev, onNext]);

  // Owned-mode photos get the clean original from BE; everyone else sees the
  // watermarked thumbnail. Closes G-2.
  const renderedSrc = photo.cleanUrl ?? photo.imageUrl ?? null;
  const hasImage = Boolean(renderedSrc);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);
  const [imageAttempt, setImageAttempt] = useState(0);
  const priorImageReloadKey = useRef(props.imageReloadKey);
  // Natural width / height of the served image, known once it loads.
  const [aspect, setAspect] = useState<number | null>(null);

  useEffect(() => {
    setImageLoaded(false);
    setImageFailed(false);
    setAspect(null);
    if (priorImageReloadKey.current !== props.imageReloadKey) {
      priorImageReloadKey.current = props.imageReloadKey;
      setImageAttempt((attempt) => attempt + 1);
    }
  }, [photo.id, renderedSrc, props.imageReloadKey]);

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

  const credit = (
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
        percentOff={photo.couponPercentOff}
        handle={photo.photographerHandle}
        name={photo.photographerName}
      />
    ) : null;
  const hasCouponPrice = mode === "browse" && photo.couponPrice != null && Boolean(photo.couponCode);

  const content = (
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      tabIndex={-1}
      aria-label={
        photo.bib ? `Preview photo ${photo.bib}` : "Preview untagged photo"
      }
      className="fixed inset-0 z-50 flex items-center justify-center outline-none md:p-8 lg:p-10"
    >
      <button
        type="button"
        tabIndex={-1}
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
              key={imageAttempt}
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

          {props.details && (!hasImage || imageFailed) && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-surface">
              <p>Preview unavailable</p>
              {(hasImage || props.onRefreshImage) && <button type="button" className={cn(BTN_SECONDARY, BTN_SIZE.sm)} disabled={props.isRefreshingImage} onClick={props.onRefreshImage ?? (() => { setImageFailed(false); setImageAttempt((n) => n + 1); })}>{props.isRefreshingImage ? "Refreshing preview…" : "Refresh preview"}</button>}
              {props.imageRefreshError && <p role="alert">{props.imageRefreshError}</p>}
            </div>
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
              ref={mobileClose}
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
        <aside className={cn("flex flex-col shrink-0 lg:w-[340px] bg-bone border-t lg:border-t-0 lg:border-l border-line", props.details && "min-h-0 max-h-[50dvh] overflow-hidden lg:max-h-none")}>
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
                ref={desktopClose}
                type="button"
                onClick={onClose}
                aria-label="Close preview"
                className="size-9 shrink-0 rounded-full border border-line text-ink hover:bg-bone-deep flex items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh"
              >
                <CloseGlyph />
              </button>
            </div>
          </div>

          <div className={cn("px-5 pt-4 pb-3 md:px-6 lg:pt-6 lg:pb-0 lg:flex-1 lg:min-h-0", props.details && "shrink-0")}>
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
                {props.mode === "owned" && (
                  <Kicker as="p" tone="soft" className="shrink-0">
                    {props.footnote ?? "Yours to keep"}
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

          {props.details && <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-4 md:px-6">{props.details}</div>}

          {/* Actions — anchored to the rail bottom on lg. */}
          <div className={cn("px-5 pb-4 md:px-6 md:pb-5 lg:pb-6 lg:pt-5 lg:border-t lg:border-line", props.details && "shrink-0")}>
            {mode === "browse" && (
              <div className="hidden lg:block mb-4">
                <p className="font-mono font-semibold tnum text-[26px] leading-none text-ink">
                  {formatPrice(hasCouponPrice ? photo.couponPrice! : photo.price)}
                </p>
                <Kicker as="p" tone="soft" tnum className="mt-1.5">
                  {hasCouponPrice
                    ? `−${photo.couponPercentOff}% · list ${formatPrice(photo.price)}`
                    : "Per photo · download forever"}
                </Kicker>
              </div>
            )}

            {props.mode === "owned" ? (
              <>
                <Kicker as="p" tone="soft" className="hidden lg:block mb-3">
                  {props.footnote ?? "Yours to keep"}
                </Kicker>
                <button
                  type="button"
                  onClick={props.onDownload}
                  disabled={props.downloadDisabled ?? false}
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

            {showMetadata && (
              <button
                ref={metadataTrigger}
                type="button"
                onClick={() => setOpenMetadataContext(metadataContext)}
                className={cn(BTN_SECONDARY, BTN_SIZE.sm, "mt-2.5 w-full")}
              >
                Show Metadata
              </button>
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

  return <>
    {createPortal(content, document.body)}
    {(access === "owner" || access === "public") && (
      <PhotoMetadataDialog
        photoId={photo.id}
        access={access}
        isOpen={metadataOpen}
        onClose={() => setOpenMetadataContext(null)}
        onDenied={() => setDeniedMetadataContext(metadataContext)}
        triggerRef={metadataTrigger}
        fallbackFocus={focusPreviewClose}
      />
    )}
  </>;
}

// Photo credit line in the rail.
//
// The public name is always the visible credit. A verified photographer's
// name links to their profile; an unverified photographer stays plain text.
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
          {name ?? `@${handle}`}
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

// The photographer's coupon offer in the rail: how much, whose photos, and
// that it applies itself at checkout (2026-09-05 — nothing to copy). Ink
// outline, not fresh — Buy now owns the accent. The BE has already priced it
// (see couponPrice on the item).
function CouponOffer({
  percentOff,
  handle,
  name,
}: {
  percentOff: number;
  handle?: string | null;
  name?: string | null;
}) {
  // The credit line directly above already names the photographer; a full
  // handle here truncates on the 340px rail. Keep the row about the offer.
  const who = handle || name ? "their" : "this photographer's";

  return (
    <div className="mt-3 rounded-xl border border-line bg-bone-deep/60 px-4 py-3">
      <p className="font-mono font-semibold tnum text-ink">−{percentOff}%</p>
      <Kicker as="p" tone="soft" tnum className="leading-snug">
        {percentOff}% off {who} photos · applied at checkout
      </Kicker>
    </div>
  );
}
