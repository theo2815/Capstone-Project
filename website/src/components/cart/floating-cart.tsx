"use client";

import { usePathname, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useCartStore } from "@/store/cart-store";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { CartModal } from "@/components/cart/cart-modal";
import { CheckoutModal } from "@/components/cart/checkout-modal";

const HIDDEN_ROUTES: string[] = [
  ROUTES.LOGIN,
  ROUTES.REGISTER,
  ROUTES.FORGOT_PASSWORD,
  ROUTES.CART,
  ROUTES.CHECKOUT,
];

export function FloatingCart() {
  const pathname = usePathname() ?? "";
  const searchParams = useSearchParams();
  const itemCount = useCartStore((s) => s.items.length);
  const total = useCartStore((s) => s.total());

  const [mounted, setMounted] = useState(false);
  const [open, setOpen] = useState(false);
  const [checkoutOpen, setCheckoutOpen] = useState(false);
  const [pulseKey, setPulseKey] = useState(0);
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
      setOpen(false);
      setCheckoutOpen(false);
    }
    prevCountRef.current = itemCount;
  }, [itemCount, mounted]);

  if (!mounted || itemCount === 0) return null;

  const isHiddenRoute = HIDDEN_ROUTES.some(
    (r) => pathname === r || pathname.startsWith(r + "/"),
  );
  if (isHiddenRoute) return null;

  const isEventDetail =
    pathname.startsWith(ROUTES.EVENTS + "/") &&
    pathname !== ROUTES.EVENTS;
  const hasBibFilter = Boolean(searchParams?.get("bib"));
  if (isEventDetail && hasBibFilter) return null;

  const display = itemCount > 99 ? "99+" : String(itemCount);

  const hasMobileStickyCta =
    pathname === ROUTES.RUNNERS || pathname === ROUTES.PHOTOGRAPHERS;
  const bottomOffset = hasMobileStickyCta
    ? "bottom-24 md:bottom-8"
    : "bottom-5 md:bottom-8";

  return (
    <>
    <div
      className={cn("fixed right-5 md:right-8 z-40", bottomOffset)}
      style={{ animation: "fade-up 0.45s ease-out both" }}
    >
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={`Open cart · ${itemCount} ${
          itemCount === 1 ? "photo" : "photos"
        } · ₱${total.toLocaleString()}`}
        className={cn(
          "group relative inline-flex items-center gap-3 pl-4 pr-5 h-14 md:h-16 rounded-full",
          "bg-fresh hover:bg-fresh-deep text-bone",
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
          <span
            key={pulseKey}
            className="absolute -top-1.5 -right-1.5 min-w-[1.25rem] h-5 px-1.5 inline-flex items-center justify-center rounded-full bg-ink text-bone font-mono text-[10px] tnum tracking-tight border-2 border-fresh group-hover:border-fresh-deep transition-colors"
            style={{ animation: "count-up 0.5s ease-out both" }}
            aria-hidden="true"
          >
            {display}
          </span>
        </span>
        <span className="hidden md:flex flex-col items-start leading-tight">
          <span className="font-mono uppercase tracking-[0.25em] text-[9px] text-bone/70">
            Cart
          </span>
          <span className="font-mono text-xs tnum">
            ₱{total.toLocaleString()}
          </span>
        </span>
        <span className="md:hidden font-mono uppercase tracking-[0.2em] text-[10px]">
          ₱{total.toLocaleString()}
        </span>
      </button>
    </div>
    <CartModal
      isOpen={open}
      onClose={() => setOpen(false)}
      onContinueToCheckout={() => {
        setOpen(false);
        setCheckoutOpen(true);
      }}
    />
    <CheckoutModal
      isOpen={checkoutOpen}
      onClose={() => setCheckoutOpen(false)}
      onBackToCart={() => {
        setCheckoutOpen(false);
        setOpen(true);
      }}
    />
    </>
  );
}
