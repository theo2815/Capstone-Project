"use client";

import { usePathname, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useCartStore } from "@/store/cart-store";
import { usePendingPaymentStore } from "@/store/pending-payment-store";
import { useUiStore } from "@/store/ui-store";
import { ROUTES } from "@/lib/constants";
import { cn, formatPrice } from "@/lib/utils";
import { CartModal } from "@/components/cart/cart-modal";
import { CheckoutModal } from "@/components/cart/checkout-modal";

const HIDDEN_ROUTES: string[] = [
  ROUTES.LOGIN,
  ROUTES.REGISTER,
  ROUTES.FORGOT_PASSWORD,
];

export function FloatingCart() {
  const pathname = usePathname() ?? "";
  const searchParams = useSearchParams();
  const itemCount = useCartStore((s) => s.items.length);
  const total = useCartStore((s) => s.total());
  // A live QR Ph payment keeps the pill (and the checkout drawer) mounted even
  // when the cart is empty, and turns the pill into the way back to that QR.
  const pending = usePendingPaymentStore((s) => s.pending);

  const open = useUiStore((s) => s.cartOpen);
  const checkoutOpen = useUiStore((s) => s.checkoutOpen);
  const openCart = useUiStore((s) => s.openCart);
  const closeCart = useUiStore((s) => s.closeCart);
  const openCheckout = useUiStore((s) => s.openCheckout);
  const closeCheckout = useUiStore((s) => s.closeCheckout);
  const expressCheckoutPending = useUiStore((s) => s.expressCheckoutPending);
  const clearExpressCheckout = useUiStore((s) => s.clearExpressCheckout);

  const [mounted, setMounted] = useState(false);
  const [pulseKey, setPulseKey] = useState(0);
  const [minimized, setMinimized] = useState(false);
  const prevCountRef = useRef(0);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) {
      prevCountRef.current = itemCount;
      return;
    }
    if (itemCount > prevCountRef.current) {
      setPulseKey((k) => k + 1);
      if (expressCheckoutPending) {
        openCheckout();
        clearExpressCheckout();
      } else if (!checkoutOpen) {
        // Never close an open checkout on a count change. The reachable
        // trigger isn't a stray add — it's the guest→login resume path:
        // <CheckoutResumeWatcher> re-opens the modal on `?checkout=1` and
        // <AuthHydrator>'s mergeCart lands right after, so any server-side
        // cart item the local cart lacked would slam the modal shut mid-
        // payment-method selection.
        closeCart();
        closeCheckout();
      }
      // New item arriving is a strong signal the user wants the cart visible —
      // pop it back out so they see the count update.
      setMinimized(false);
    }
    prevCountRef.current = itemCount;
  }, [
    itemCount,
    mounted,
    expressCheckoutPending,
    checkoutOpen,
    openCheckout,
    closeCart,
    closeCheckout,
    clearExpressCheckout,
  ]);

  useEffect(() => {
    if (open || checkoutOpen) setMinimized(false);
  }, [open, checkoutOpen]);

  if (!mounted) return null;
  // The drawer lives here, so the host must stay mounted while checkout is
  // open — success clears both the cart and the pending record, and the
  // "All yours." step would unmount with them otherwise.
  if (itemCount === 0 && !pending && !checkoutOpen) return null;

  const isHiddenRoute = HIDDEN_ROUTES.some(
    (r) => pathname === r || pathname.startsWith(r + "/"),
  );
  if (isHiddenRoute && !checkoutOpen) return null;
  const showPill = !isHiddenRoute && (itemCount > 0 || pending !== null);

  // Two surfaces render BuyAllBar on filtered browse:
  //   1. /events/{slug}              → runner-wide gallery + bib filter
  //   2. /{handle}/events/{slug}     → per-photographer gallery + bib filter
  // Both need the cart pill to lift above the buy-all bar so users can still
  // reach the cart without clearing the filter. Admin + dashboard variants
  // share the URL shape but don't render BuyAllBar, so exclude them.
  const isRunnerEventDetail =
    pathname.startsWith(ROUTES.EVENTS + "/") &&
    pathname !== ROUTES.EVENTS;
  const isPhotographerEventGallery =
    /^\/[^/]+\/events\/[^/]+$/.test(pathname) &&
    !pathname.startsWith("/admin/") &&
    !pathname.startsWith("/dashboard/") &&
    !pathname.startsWith(ROUTES.EVENTS + "/");
  const isEventDetail = isRunnerEventDetail || isPhotographerEventGallery;
  const hasBibFilter = Boolean(searchParams?.get("bib"));
  const isFilteredBrowse = isEventDetail && hasBibFilter;

  const display = itemCount > 99 ? "99+" : String(itemCount);
  const showBadge = itemCount > 0;
  const pillAmount = formatPrice(pending ? pending.total : total);
  const pillLabel = pending
    ? `Payment pending · ${pillAmount}`
    : `${itemCount} ${itemCount === 1 ? "photo" : "photos"} · ${pillAmount}`;
  const openPill = () => (pending ? openCheckout() : openCart());

  const hasMobileStickyCta =
    pathname === ROUTES.RUNNERS || pathname === ROUTES.PHOTOGRAPHERS;
  // Mobile bottom = max(designed offset, safe-area inset) so the pill never
  // sits under the iOS home indicator.
  // - Filtered browse (`?bib=...`) sits above BuyAllBar (~68-80px tall) so users
  //   can still reach their cart without clearing the filter.
  // - /runners + /photographers add the inset on top of the 6rem clearance so
  //   the pill clears MobileStickyCta (which itself absorbs the inset via its
  //   own padding-bottom).
  let mobileBottomClass: string;
  let desktopBottomClass: string;
  if (isFilteredBrowse) {
    mobileBottomClass = "bottom-[calc(5.5rem+env(safe-area-inset-bottom))]";
    desktopBottomClass = "md:bottom-[6.5rem]";
  } else if (hasMobileStickyCta) {
    mobileBottomClass = "bottom-[calc(6rem+env(safe-area-inset-bottom))]";
    desktopBottomClass = "md:bottom-8";
  } else {
    mobileBottomClass =
      "bottom-[max(1.25rem,calc(env(safe-area-inset-bottom)+0.75rem))]";
    desktopBottomClass = "md:bottom-8";
  }

  return (
    <>
    {showPill && (<>
    {/* Full pill — slides off when minimized */}
    <div
      className={cn(
        "fixed right-5 md:right-8 z-40 will-change-transform",
        mobileBottomClass,
        desktopBottomClass,
        "transition-transform duration-[420ms] [transition-timing-function:cubic-bezier(0.32,0.72,0,1)]",
        minimized
          ? "translate-x-[calc(100%+2rem)] pointer-events-none"
          : "translate-x-0",
      )}
      style={{ animation: "fade-up 0.45s ease-out both" }}
      aria-hidden={minimized || undefined}
    >
      <div className="relative">
        <button
          type="button"
          onClick={openPill}
          aria-haspopup="dialog"
          aria-expanded={open || checkoutOpen}
          aria-label={`Open ${pending ? "checkout" : "cart"} · ${pillLabel}`}
          className={cn(
            "group relative inline-flex items-center gap-3 pl-4 pr-5 h-14 md:h-16 rounded-full",
            "bg-fresh hover:bg-fresh-deep text-surface",
            "shadow-[0_18px_40px_-12px_rgba(17,17,17,0.5)]",
            "focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
            "transition-transform duration-200 hover:-translate-y-0.5",
          )}
        >
          <span className="relative inline-flex items-center justify-center size-10 md:size-11 rounded-full bg-bone/15">
            <svg
              viewBox="0 0 24 24"
              className="size-5 md:size-[22px]"
              fill="none"
              aria-hidden="true"
            >
              <path
                d="M3.5 4 H6 L8.5 16 H19 L21 8 H7.2"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <circle cx="9.5" cy="20" r="1.4" fill="currentColor" />
              <circle cx="17" cy="20" r="1.4" fill="currentColor" />
            </svg>
            {showBadge && (
              <span
                key={pulseKey}
                className="absolute -top-1.5 -right-1.5 min-w-[1.25rem] h-5 px-1.5 inline-flex items-center justify-center rounded-full bg-ink text-bone font-mono text-[10px] tnum tracking-tight border-2 border-fresh group-hover:border-fresh-deep transition-colors"
                style={{ animation: "count-up 0.5s ease-out both" }}
                aria-hidden="true"
              >
                {display}
              </span>
            )}
          </span>
          <span className="hidden md:flex flex-col items-start leading-tight">
            <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-bone/70">
              {pending ? "Payment pending" : "Cart"}
            </span>
            <span className="font-mono text-[14px] min-[400px]:text-[15px] md:text-[13px] tnum">
              {pillAmount}
            </span>
          </span>
          <span className="md:hidden font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] tnum">
            {pending ? "Pending" : pillAmount}
          </span>
        </button>
        {/* Minimize close-chip — detached so it doesn't crowd the count badge */}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            setMinimized(true);
          }}
          aria-label="Hide cart pill"
          tabIndex={minimized ? -1 : 0}
          className={cn(
            "absolute -top-2 -right-2 size-7 rounded-full",
            "bg-ink hover:bg-ink-soft text-bone",
            "border-2 border-bone shadow-[0_4px_12px_-2px_rgba(0,0,0,0.35)]",
            "inline-flex items-center justify-center transition-colors",
            "focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
          )}
        >
          <svg
            viewBox="0 0 16 16"
            className="size-3"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M3 3 L13 13 M13 3 L3 13"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            />
          </svg>
        </button>
      </div>
    </div>

    {/* Restore handle — half-circle pinned to right edge when minimized */}
    <button
      type="button"
      onClick={() => setMinimized(false)}
      aria-label={`Show ${pending ? "pending payment" : "cart"} · ${pillLabel}`}
      tabIndex={minimized ? 0 : -1}
      aria-hidden={!minimized || undefined}
      className={cn(
        "fixed -right-7 md:-right-8 z-40 will-change-transform group",
        mobileBottomClass,
        desktopBottomClass,
        "size-14 md:size-16 rounded-full",
        "bg-fresh hover:bg-fresh-deep text-surface",
        "shadow-[-8px_18px_40px_-12px_rgba(17,17,17,0.45)]",
        "inline-flex items-center justify-center",
        "transition-transform duration-[420ms] [transition-timing-function:cubic-bezier(0.32,0.72,0,1)]",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
        minimized
          ? "translate-x-0"
          : "translate-x-full pointer-events-none",
      )}
    >
      {/* Cart icon — centered in the full circle so only the left half is visible */}
      <svg
        viewBox="0 0 24 24"
        className="size-6 md:size-[26px]"
        fill="none"
        aria-hidden="true"
      >
        <path
          d="M3.5 4 H6 L8.5 16 H19 L21 8 H7.2"
          stroke="currentColor"
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="9.5" cy="20" r="1.4" fill="currentColor" />
        <circle cx="17" cy="20" r="1.4" fill="currentColor" />
      </svg>
      {/* Count badge — anchored over the upper-left curve of the visible half */}
      {showBadge && (
        <span
          className="absolute top-1 -left-1.5 min-w-[1.25rem] h-5 px-1.5 inline-flex items-center justify-center rounded-full bg-ink text-bone font-mono text-[10px] tnum tracking-tight border-2 border-fresh group-hover:border-fresh-deep transition-colors"
          aria-hidden="true"
        >
          {display}
        </span>
      )}
    </button>
    </>)}

    <CartModal
      isOpen={open}
      onClose={() => closeCart()}
      onContinueToCheckout={() => openCheckout()}
    />
    <CheckoutModal
      isOpen={checkoutOpen}
      onClose={() => closeCheckout()}
      onBackToCart={() => openCart()}
    />
    </>
  );
}
