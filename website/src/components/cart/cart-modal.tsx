"use client";

import { PROTECTED_IMG_CLASS, PROTECTED_IMG_PROPS } from "@/lib/protected-image";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { useCartStore } from "@/store/cart-store";
import { useConfirmation } from "@/hooks/use-confirmation";
import { fetchEventDetail } from "@/lib/api-events";
import { ROUTES } from "@/lib/constants";
import { useScrollLock } from "@/lib/scroll-lock";
import { cn, formatPrice } from "@/lib/utils";
import { BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import type { CartItem } from "@/types/order";
import {
  PhotoPreviewCard,
  type PhotoPreviewItem,
} from "@/components/photos/photo-preview-card";

const TONE_COLORS = [
  "var(--ink)",
  "var(--ink-soft)",
  "var(--slate)",
  "var(--slate-soft)",
];

interface CartModalProps {
  isOpen: boolean;
  onClose: () => void;
  onContinueToCheckout?: () => void;
}

function cartItemToPreview(item: CartItem): PhotoPreviewItem {
  return {
    id: item.photoId,
    bib: item.bib ?? null,
    time: item.time ?? "—",
    tone: item.tone ?? 0,
    price: item.price,
    imageUrl: item.thumbnailUrl || null,
  };
}

export function CartModal({
  isOpen,
  onClose,
  onContinueToCheckout,
}: CartModalProps) {
  const items = useCartStore((s) => s.items);
  const removeItem = useCartStore((s) => s.removeItem);
  const clear = useCartStore((s) => s.clear);
  const total = useCartStore((s) => s.total());
  const syncEnabled = useCartStore((s) => s.syncEnabled);
  const { confirm } = useConfirmation();

  // Guest carts carry the price captured at browse time. A signed-in runner's
  // cart self-corrects — the server owns the row and `GET /me/cart` renders
  // the live `photos.price_php` — but a guest has no `cart_items` row for the
  // BE to correct, while checkout still charges the live price. So an admin
  // re-price between browsing and checking out would bill more than the cart
  // showed. Re-read on open.
  //
  // One GET per distinct event, not per photo: price is an event-level value
  // (admin PATCH re-prices every photo under the event), so the event detail
  // answers for the whole cart. Items with no `eventSlug` — persisted before
  // the field existed — are skipped rather than guessed at.
  useEffect(() => {
    if (!isOpen || syncEnabled) return;
    let cancelled = false;

    const slugs = Array.from(
      new Set(
        useCartStore
          .getState()
          .items.map((i) => i.eventSlug)
          .filter((s): s is string => Boolean(s)),
      ),
    );
    if (slugs.length === 0) return;

    void Promise.all(
      slugs.map(async (slug) => {
        const event = await fetchEventDetail(slug);
        return [slug, event?.pricePerPhoto ?? null] as const;
      }),
    ).then((pairs) => {
      if (cancelled) return;
      const priceBySlug = new Map<string, number>();
      for (const [slug, price] of pairs) {
        if (price !== null) priceBySlug.set(slug, price);
      }
      if (priceBySlug.size === 0) return;

      // Re-read at write time — an add or remove may have landed while the
      // requests were in flight, and setItems replaces the whole array.
      const current = useCartStore.getState().items;
      let changed = false;
      const next = current.map((item) => {
        const price = item.eventSlug
          ? priceBySlug.get(item.eventSlug)
          : undefined;
        if (price === undefined || price === item.price) return item;
        changed = true;
        return { ...item, price };
      });
      if (changed) useCartStore.getState().setItems(next);
    });

    return () => {
      cancelled = true;
    };
  }, [isOpen, syncEnabled]);

  async function handleClearCart() {
    const ok = await confirm({
      title: "Clear cart?",
      message:
        "This removes every photo from your cart. You'll have to add them again to check out.",
      confirmLabel: "Clear cart",
      danger: true,
    });
    if (!ok) return;
    clear();
  }

  const pathname = usePathname() ?? "";
  const router = useRouter();

  const uniqueEvents = useMemo(() => {
    const map = new Map<string, { slug?: string; name?: string }>();
    for (const it of items) {
      if (!map.has(it.eventId)) {
        map.set(it.eventId, { slug: it.eventSlug, name: it.eventName });
      }
    }
    return Array.from(map.entries()).map(([eventId, v]) => ({
      eventId,
      slug: v.slug,
      name: v.name,
    }));
  }, [items]);

  const currentEventSlug = useMemo(() => {
    const m = pathname.match(/^\/events\/([^/?]+)/);
    return m ? m[1] : null;
  }, [pathname]);

  const isOnEventsList = pathname === ROUTES.EVENTS;
  const isOnAnyCartEvent =
    currentEventSlug !== null &&
    uniqueEvents.some((e) => e.slug === currentEventSlug);

  const browseTarget = useMemo<{
    label: string;
    href: string | null;
  }>(() => {
    if (isOnEventsList || isOnAnyCartEvent) {
      return { label: "Keep browsing", href: null };
    }
    if (uniqueEvents.length === 1 && uniqueEvents[0].slug) {
      const e = uniqueEvents[0];
      const labelName = e.name ?? "this event";
      return {
        label: `Continue · ${labelName}`,
        href: `${ROUTES.EVENTS}/${e.slug}?browse=1`,
      };
    }
    return { label: "Browse events", href: ROUTES.EVENTS };
  }, [isOnEventsList, isOnAnyCartEvent, uniqueEvents]);

  const handleKeepBrowsing = () => {
    if (browseTarget.href) {
      router.push(browseTarget.href);
    }
    onClose();
  };

  const [previewIndex, setPreviewIndex] = useState<number | null>(null);
  const previewIndexRef = useRef<number | null>(null);

  // Three-state render machine: keeps the drawer mounted during the slide-out
  // exit animation so close paths (X, backdrop, Esc, checkout-handoff) feel
  // smooth instead of snapping. EXIT_MS must match the keyframe duration.
  const EXIT_MS = 350;
  const [renderState, setRenderState] = useState<"hidden" | "open" | "closing">(
    isOpen ? "open" : "hidden",
  );

  useEffect(() => {
    if (isOpen) {
      setRenderState("open");
      return;
    }
    setRenderState((prev) => (prev === "open" ? "closing" : "hidden"));
  }, [isOpen]);

  useEffect(() => {
    if (renderState !== "closing") return;
    const t = setTimeout(() => setRenderState("hidden"), EXIT_MS);
    return () => clearTimeout(t);
  }, [renderState]);

  useEffect(() => {
    previewIndexRef.current = previewIndex;
  }, [previewIndex]);

  useScrollLock(renderState === "open");

  useEffect(() => {
    if (renderState !== "open") {
      if (renderState === "hidden") setPreviewIndex(null);
      return;
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && previewIndexRef.current === null) onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
    };
  }, [renderState, onClose]);

  useEffect(() => {
    if (previewIndex !== null && previewIndex >= items.length) {
      setPreviewIndex(items.length === 0 ? null : items.length - 1);
    }
  }, [items.length, previewIndex]);

  if (renderState === "hidden") return null;
  const isClosing = renderState === "closing";

  const itemCount = items.length;
  const previewItem =
    previewIndex !== null ? items[previewIndex] ?? null : null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Your cart"
      aria-hidden={isClosing || undefined}
      className={cn(
        "fixed inset-0 z-50",
        isClosing && "pointer-events-none",
      )}
    >
      <button
        type="button"
        onClick={isClosing ? undefined : onClose}
        disabled={isClosing}
        aria-label="Close cart"
        className="absolute inset-0 bg-ink/50 backdrop-blur-sm cursor-default"
        style={{
          animation: isClosing
            ? "fade-out 0.3s ease-in both"
            : "fade-in 0.25s ease-out both",
        }}
      />

      <aside
        className="absolute top-0 right-0 h-full w-full sm:max-w-md flex flex-col bg-bone shadow-[-30px_0_80px_-20px_rgba(0,0,0,0.45)]"
        style={{
          animation: isClosing
            ? "slide-out-right 0.35s cubic-bezier(0.4, 0, 1, 1) both"
            : "slide-in-right 0.35s cubic-bezier(0.16, 1, 0.3, 1) both",
        }}
      >
        <header className="flex items-start justify-between gap-3 px-6 md:px-7 pt-6 pb-5 border-b border-line">
          <div>
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink-soft mb-1.5">
              Your cart
            </p>
            <p className="font-display text-2xl md:text-3xl font-medium text-ink tracking-tight leading-tight">
              {itemCount === 0
                ? "Empty for now."
                : itemCount === 1
                  ? "1 photo."
                  : `${itemCount} photos.`}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={isClosing}
            aria-label="Close cart"
            className="size-9 shrink-0 rounded-full border border-line text-ink hover:bg-bone-deep flex items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh disabled:opacity-50 disabled:cursor-not-allowed"
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
        </header>

        {itemCount === 0 ? (
          <EmptyCart onClose={onClose} />
        ) : (
          <>
            <ul className="flex-1 overflow-y-auto divide-y divide-line">
              {items.map((item, i) => (
                <CartRow
                  key={item.photoId}
                  item={item}
                  index={i}
                  onPreview={() => setPreviewIndex(i)}
                  onRemove={() => removeItem(item.photoId)}
                />
              ))}
            </ul>

            <footer className="border-t border-line bg-bone-deep px-6 md:px-7 py-5 md:py-6">
              <div className="flex items-center justify-between gap-3 mb-4">
                <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink-soft">
                  Subtotal
                </span>
                <span className="font-display text-3xl md:text-4xl font-medium text-ink tracking-tight tnum">
                  {formatPrice(total)}
                </span>
              </div>
              <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft mb-5">
                Watermarked previews are free. Pay once, download forever.
              </p>
              <button
                type="button"
                onClick={() => {
                  if (onContinueToCheckout) {
                    onContinueToCheckout();
                  } else {
                    onClose();
                  }
                }}
                className={cn(BTN_PRIMARY, BTN_SIZE.md, "w-full")}
              >
                Continue to checkout →
              </button>
              <div className="mt-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <button
                  type="button"
                  onClick={handleKeepBrowsing}
                  className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors text-left sm:max-w-[60%] sm:truncate"
                  title={browseTarget.label}
                >
                  ← {browseTarget.label}
                </button>
                <button
                  type="button"
                  onClick={handleClearCart}
                  className="self-start sm:self-auto font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft hover:text-fresh transition-colors shrink-0"
                >
                  Clear cart
                </button>
              </div>
            </footer>
          </>
        )}
      </aside>

      {previewItem && previewIndex !== null && (
        <PhotoPreviewCard
          photo={cartItemToPreview(previewItem)}
          eventName={previewItem.eventName ?? "QuickPitik"}
          index={previewIndex + 1}
          total={items.length}
          inCart
          onClose={() => setPreviewIndex(null)}
          onPrev={
            previewIndex > 0 ? () => setPreviewIndex(previewIndex - 1) : undefined
          }
          onNext={
            previewIndex < items.length - 1
              ? () => setPreviewIndex(previewIndex + 1)
              : undefined
          }
          onToggleCart={() => {
            removeItem(previewItem.photoId);
            setPreviewIndex(null);
          }}
          onBuyNow={() => {
            setPreviewIndex(null);
            onContinueToCheckout?.();
          }}
        />
      )}
    </div>
  );
}

function CartRow({
  item,
  index,
  onPreview,
  onRemove,
}: {
  item: CartItem;
  index: number;
  onPreview: () => void;
  onRemove: () => void;
}) {
  const tone = item.tone ?? 0;
  const colorIdx = tone % TONE_COLORS.length;

  return (
    <li
      className="flex items-start gap-4 px-6 md:px-7 py-5"
      style={{
        animation: `fade-up 0.4s ${Math.min(index * 0.04, 0.4)}s both`,
        opacity: 0,
      }}
    >
      <button
        type="button"
        onClick={onPreview}
        aria-label={`Preview ${item.bib ?? "untagged photo"} from ${item.eventName ?? "QuickPitik"}`}
        aria-haspopup="dialog"
        className="group flex flex-1 min-w-0 items-start gap-4 text-left rounded-md focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
      >
        <div
          className="relative size-20 shrink-0 rounded-md overflow-hidden transition-transform duration-200 group-hover:-translate-y-0.5"
          style={{ backgroundColor: TONE_COLORS[colorIdx] }}
          aria-hidden="true"
        >
          {item.thumbnailUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={item.thumbnailUrl}
              alt=""
              onError={(e) => {
                // Broken URL → hide the img so the solid tone block shows
                // instead of the browser's broken-image icon.
                e.currentTarget.style.display = "none";
              }}
              className={`absolute inset-0 w-full h-full object-cover ${PROTECTED_IMG_CLASS}`}
              {...PROTECTED_IMG_PROPS}
            />
          ) : (
            <span className="absolute inset-0 flex items-center justify-center font-mono uppercase tracking-[0.14em] text-[8px] text-bone/40 rotate-[-18deg]">
              QuickPitik
            </span>
          )}
          <span className="absolute inset-0 bg-ink/0 group-hover:bg-ink/25 transition-colors duration-200 flex items-center justify-center">
            <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-bone/0 group-hover:text-bone/95 transition-colors duration-200">
              View
            </span>
          </span>
        </div>

        <div className="flex-1 min-w-0">
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate truncate">
            {item.eventName ?? "QuickPitik"}
          </p>
          <p className="mt-1 font-display text-lg font-medium text-ink tracking-tight tnum truncate group-hover:text-fresh transition-colors">
            {item.bib ?? "Untagged"}
          </p>
          {item.time ? (
            <p className="mt-1 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft tnum">
              {item.time}
            </p>
          ) : null}
        </div>
      </button>

      <div className="flex flex-col items-end gap-1.5 shrink-0">
        <span className="font-mono text-sm text-ink tnum">
          {formatPrice(item.price)}
        </span>
        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove ${item.bib ?? "untagged photo"} from cart`}
          className="inline-flex items-center gap-1 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft hover:text-fresh transition-colors focus:outline-none focus-visible:text-fresh"
        >
          <span aria-hidden="true">✕</span>
          <span>Remove</span>
        </button>
      </div>
    </li>
  );
}

function EmptyCart({ onClose }: { onClose: () => void }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-8 py-12 text-center">
      <div
        className="size-16 rounded-full border border-line flex items-center justify-center mb-6 text-slate"
        aria-hidden="true"
      >
        <svg viewBox="0 0 24 24" className="size-7" fill="none">
          <path
            d="M3.5 4 H6 L8.5 16 H19 L21 8 H7.2"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <circle cx="9.5" cy="20" r="1.2" fill="currentColor" />
          <circle cx="17" cy="20" r="1.2" fill="currentColor" />
        </svg>
      </div>
      <p className="font-display text-xl font-medium text-ink mb-2">
        Nothing here yet.
      </p>
      <p className="font-sans text-sm text-ink-soft max-w-xs mb-7">
        Open an event, search by bib or selfie, then add the photos you want to
        keep.
      </p>
      <Link
        href={ROUTES.EVENTS}
        onClick={onClose}
        className={cn(
          "inline-flex items-center bg-ink hover:bg-ink-soft text-bone px-6 py-3 rounded-full",
          "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] transition-colors",
        )}
      >
        Browse events →
      </Link>
    </div>
  );
}
