"use client";

import { useEffect, useState, type MouseEvent } from "react";
import { useCartStore } from "@/store/cart-store";
import { useUiStore } from "@/store/ui-store";
import { cn } from "@/lib/utils";
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
// row-span-1, `default` → row-span-2). Owns its own +cart / buy → buttons that
// fade in on hover; tile click opens the parent's preview via `onOpen`.
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
    thumbnailUrl: "",
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

  const fadeRule = "opacity-100 md:opacity-60 md:group-hover:opacity-100";

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
                imageLoaded ? "opacity-100" : "opacity-0",
              )}
              draggable={false}
            />
          )}
          <span
            aria-hidden="true"
            className="absolute inset-0 flex items-center justify-center bg-ink/0 group-hover:bg-ink/30 transition-colors duration-300"
          >
            <span className="font-mono uppercase tracking-[0.3em] text-[10px] text-bone/0 group-hover:text-bone/95 transition-colors duration-300">
              View →
            </span>
          </span>
        </div>
      </button>
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
            "inline-flex items-center gap-1 px-2.5 py-1.5 rounded-full font-mono uppercase tracking-[0.2em] text-[9px] whitespace-nowrap",
            "shadow-[0_4px_12px_-2px_rgba(0,0,0,0.25)]",
            "transition-all duration-200",
            "focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
            inCart
              ? "bg-fresh text-bone hover:bg-fresh-deep"
              : cn(
                  "bg-bone/90 backdrop-blur-sm text-ink hover:bg-bone hover:scale-105",
                  fadeRule,
                ),
          )}
        >
          <span aria-hidden="true">{inCart ? "✓" : "+"}</span>
          <span>cart</span>
        </button>
        <button
          type="button"
          onClick={handleBuyNow}
          aria-label={`Buy ${photo.bib ?? "untagged photo"} now for ₱${photo.price}`}
          className={cn(
            "inline-flex items-center gap-1 px-2.5 py-1.5 rounded-full font-mono uppercase tracking-[0.2em] text-[9px] whitespace-nowrap",
            "shadow-[0_4px_12px_-2px_rgba(0,0,0,0.25)]",
            "transition-all duration-200",
            "focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
            "bg-bone/90 backdrop-blur-sm text-ink hover:bg-fresh hover:text-bone hover:scale-105",
            fadeRule,
          )}
        >
          <span>buy</span>
          <span aria-hidden="true">→</span>
        </button>
      </div>
    </div>
  );
}
