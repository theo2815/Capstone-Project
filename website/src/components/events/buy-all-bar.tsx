"use client";

import { useState } from "react";
import { MAX_CART_ITEMS, useCartStore } from "@/store/cart-store";
import { FieldError } from "@/components/ui/field-error";
import { cn } from "@/lib/utils";
import type { EventDetail } from "@/types/event";
import type { MockPhoto } from "@/app/events/[slug]/mock-photos";

// Sticky bottom bar shown when a bib filter is active and at least one match
// exists. Adds every visible photo to the cart in one click. FloatingCart
// offsets above this bar on filtered-browse routes (see floating-cart.tsx).
export function BuyAllBar({
  event,
  photos,
  total,
}: {
  event: EventDetail;
  photos: MockPhoto[];
  total: number;
}) {
  const addItem = useCartStore((s) => s.addItem);
  const items = useCartStore((s) => s.items);
  const [pressed, setPressed] = useState(false);
  const [capError, setCapError] = useState<string | null>(null);

  const allInCart =
    photos.length > 0 &&
    photos.every((p) => items.some((i) => i.photoId === p.id));

  const handleBuyAll = () => {
    setCapError(null);
    let skipped = 0;
    for (const p of photos) {
      const ok = addItem({
        photoId: p.id,
        eventId: event.id,
        thumbnailUrl: "",
        price: p.price,
        bib: p.bib,
        eventName: event.name,
        eventSlug: event.slug,
        tone: p.tone,
        time: p.time,
      });
      if (!ok && !items.some((i) => i.photoId === p.id)) {
        skipped += 1;
      }
    }
    if (skipped > 0) {
      setCapError(
        `Cart is full at ${MAX_CART_ITEMS}. ${skipped} photo${
          skipped === 1 ? "" : "s"
        } didn't fit.`,
      );
    }
    setPressed(true);
    setTimeout(() => setPressed(false), 2400);
  };

  if (photos.length === 0) return null;

  return (
    <div className="fixed bottom-0 inset-x-0 px-4 md:px-10 py-3 md:py-4 bg-bone/95 backdrop-blur-md border-t border-line z-30">
      <div className="max-w-7xl mx-auto flex flex-col gap-2">
        {capError && (
          <FieldError
            message={capError}
            id="buyall-cap-error"
            density="tight"
            className="mt-0 text-center sm:text-left"
          />
        )}
        <div className="flex items-center justify-between gap-3 md:gap-4">
          <div className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate hidden sm:flex items-center gap-3">
            <span>
              <span className="tnum text-ink">{photos.length}</span>{" "}
              {photos.length === 1 ? "photo" : "photos"}
            </span>
            <span className="h-3 w-px bg-line" aria-hidden="true" />
            <span>
              Total{" "}
              <span className="tnum text-ink">₱{total.toLocaleString()}</span>
            </span>
          </div>
          <button
            type="button"
            onClick={handleBuyAll}
            disabled={pressed || allInCart}
            aria-describedby={capError ? "buyall-cap-error" : undefined}
            className={cn(
              "ml-auto inline-flex items-center bg-fresh hover:bg-fresh-deep text-bone px-5 md:px-7 py-3 rounded-full font-mono uppercase tracking-[0.2em] text-[13px] min-[400px]:text-[14px] md:text-[12px] transition-colors whitespace-nowrap",
              (pressed || allInCart) && "opacity-90 cursor-default",
            )}
          >
            {pressed || allInCart
              ? `Added · ${photos.length} in cart ✓`
              : photos.length === 1
                ? `Buy 1 · ₱${total.toLocaleString()} →`
                : `Buy all ${photos.length} · ₱${total.toLocaleString()} →`}
          </button>
        </div>
      </div>
    </div>
  );
}
