"use client";

import { PROTECTED_IMG_CLASS, PROTECTED_IMG_PROPS } from "@/lib/protected-image";
import { useEffect, useState, type MouseEvent } from "react";
import { useCartStore } from "@/store/cart-store";
import { useUiStore } from "@/store/ui-store";
import { useToast } from "@/hooks/use-toast";
import { cn, copyToClipboard, triggerDownload } from "@/lib/utils";
import type { EventDetail } from "@/types/event";
import type { MockPhoto } from "@/types/photo";

const TONE_COLORS = [
  "var(--ink)",
  "var(--ink-soft)",
  "var(--slate)",
  "var(--slate-soft)",
];

// Single mosaic tile used by every browse-style photo grid (runner browse +
// per-photographer event gallery). Span comes from the photo (`wide` →
// row-span-1, `default` → row-span-2). Owns its own always-visible Cart / Buy
// pills (Buy is the heavier ink pill — fresh is reserved for the in-cart
// state so a full grid doesn't flood the accent); tile click opens the
// parent's preview via `onOpen`.
export function PhotoMosaicTile({
  event,
  photo,
  index,
  onOpen,
}: {
  event: EventDetail;
  photo: MockPhoto;
  index: number;
  onOpen: () => void;
}) {
  const inCart = useCartStore((s) =>
    s.items.some((i) => i.photoId === photo.id),
  );
  const addItem = useCartStore((s) => s.addItem);
  const removeItem = useCartStore((s) => s.removeItem);
  const startExpressCheckout = useUiStore((s) => s.startExpressCheckout);
  const openCheckout = useUiStore((s) => s.openCheckout);
  const { showToast } = useToast();

  const wide = photo.span === "wide";
  const colorIdx = photo.tone % TONE_COLORS.length;
  const opacity = 0.7 + (index % 3) * 0.1;
  const hasImage = Boolean(photo.imageUrl);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);

  useEffect(() => {
    setImageLoaded(false);
    setImageFailed(false);
  }, [photo.id, photo.imageUrl]);

  const cartPayload = {
    photoId: photo.id,
    eventId: event.id,
    thumbnailUrl: photo.imageUrl ?? "",
    price: photo.price,
    bib: photo.bib,
    eventName: event.name,
    eventSlug: event.slug,
    tone: photo.tone,
    time: photo.time,
  };

  const handleToggleCart = (e: MouseEvent) => {
    e.stopPropagation();
    if (inCart) {
      removeItem(photo.id);
    } else {
      addItem(cartPayload);
    }
  };

  const handleBuyNow = (e: MouseEvent) => {
    e.stopPropagation();
    if (inCart) {
      openCheckout();
      return;
    }
    startExpressCheckout();
    addItem(cartPayload);
  };

  // The photographer's coupon rides on every one of their tiles so a runner
  // sees the offer before the lightbox. Ink, not fresh — the grid's single
  // accent stays reserved for the in-cart state.
  const handleCopyCoupon = async (e: MouseEvent) => {
    e.stopPropagation();
    if (!photo.couponCode) return;
    const ok = await copyToClipboard(photo.couponCode);
    showToast(
      ok
        ? { kind: "success", message: `Code ${photo.couponCode} copied. Paste it at checkout.` }
        : { kind: "error", message: "Couldn't copy the code." },
    );
  };

  return (
    <div
      className={cn("group relative", wide ? "row-span-1" : "row-span-2")}
      style={{
        animation: `fade-up 0.55s ${Math.min(index * 0.025, 0.6)}s both`,
        opacity: 0,
      }}
    >
      <button
        type="button"
        onClick={onOpen}
        aria-haspopup="dialog"
        aria-label={
          photo.bib
            ? `Open preview of photo ${photo.bib}`
            : "Open preview of untagged crowd photo"
        }
        className="block w-full h-full text-left rounded-xl focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
      >
        <div
          className="relative overflow-hidden rounded-xl h-full transition-transform duration-300 group-hover:-translate-y-[2px]"
          style={{
            backgroundColor: TONE_COLORS[colorIdx],
            opacity,
          }}
        >
          {hasImage && !imageFailed && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={photo.imageUrl ?? ""}
              loading="lazy"
              decoding="async"
              alt={
                photo.alt ??
                (photo.bib
                  ? `Race photo of bib ${photo.bib}`
                  : "Untagged race photo")
              }
              onLoad={() => setImageLoaded(true)}
              onError={() => setImageFailed(true)}
              className={cn(
                "absolute inset-0 w-full h-full object-cover transition-opacity duration-500",
                PROTECTED_IMG_CLASS,
                imageLoaded ? "opacity-100" : "opacity-0",
              )}
              {...PROTECTED_IMG_PROPS}
            />
          )}
          <span
            aria-hidden="true"
            className="absolute inset-0 flex items-center justify-center bg-ink/0 group-hover:bg-ink/30 transition-colors duration-300"
          >
            <span className="font-mono uppercase tracking-[0.14em] text-[10px] text-bone/0 group-hover:text-bone/95 transition-colors duration-300">
              View →
            </span>
          </span>
        </div>
      </button>
      {photo.couponCode && photo.couponPercentOff != null && (
        <button
          type="button"
          onClick={handleCopyCoupon}
          aria-label={`Copy coupon ${photo.couponCode} for ${photo.couponPercentOff}% off this photographer's photos`}
          className={cn(
            "absolute top-3 left-3 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full font-mono text-[12px] tnum whitespace-nowrap",
            "bg-ink/85 backdrop-blur-sm text-surface shadow-[0_4px_12px_-2px_rgba(0,0,0,0.25)]",
            "transition-colors duration-200 hover:bg-ink",
            "focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
          )}
        >
          <span>−{photo.couponPercentOff}%</span>
          <span className="hidden sm:inline">· {photo.couponCode}</span>
        </button>
      )}
      {photo.free && photo.downloadUrl ? (
        // Free event (V46): the original is anyone's — one ink pill, no cart.
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            triggerDownload(photo.downloadUrl!);
          }}
          aria-label={`Download ${photo.bib ?? "untagged photo"} for free`}
          className={cn(
            "absolute bottom-3 right-3 inline-flex items-center gap-1 px-3 py-1.5 rounded-full font-display font-bold text-[12px] whitespace-nowrap",
            "shadow-[0_4px_12px_-2px_rgba(0,0,0,0.25)] transition-colors duration-200",
            "focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
            "bg-ink/85 backdrop-blur-sm text-surface hover:bg-fresh",
          )}
        >
          <span>Download</span>
          <span aria-hidden="true">↓</span>
        </button>
      ) : (
      <div className="absolute bottom-3 right-3 flex items-center gap-1.5">
        <button
          type="button"
          onClick={handleToggleCart}
          aria-pressed={inCart}
          aria-label={
            inCart
              ? `Remove ${photo.bib ?? "untagged photo"} from cart`
              : `Add ${photo.bib ?? "untagged photo"} to cart`
          }
          className={cn(
            "inline-flex items-center gap-1 px-3 py-1.5 rounded-full font-display font-bold text-[12px] whitespace-nowrap",
            "shadow-[0_4px_12px_-2px_rgba(0,0,0,0.25)]",
            "transition-colors duration-200",
            "focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
            inCart
              ? "bg-fresh text-surface hover:bg-fresh-deep"
              : "bg-surface/95 backdrop-blur-sm text-ink border border-line-strong hover:bg-surface hover:border-ink",
          )}
        >
          <span aria-hidden="true">{inCart ? "✓" : "+"}</span>
          <span>Cart</span>
        </button>
        <button
          type="button"
          onClick={handleBuyNow}
          aria-label={`Buy ${photo.bib ?? "untagged photo"} now for ₱${photo.price}`}
          className={cn(
            "inline-flex items-center gap-1 px-3 py-1.5 rounded-full font-display font-bold text-[12px] whitespace-nowrap",
            "shadow-[0_4px_12px_-2px_rgba(0,0,0,0.25)]",
            "transition-colors duration-200",
            "focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
            "bg-ink/85 backdrop-blur-sm text-surface hover:bg-fresh",
          )}
        >
          <span>Buy</span>
          <span aria-hidden="true">→</span>
        </button>
      </div>
      )}
    </div>
  );
}
