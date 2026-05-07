"use client";

import Link from "next/link";
import { useCart } from "@/hooks/use-cart";
import { useConfirmation } from "@/hooks/use-confirmation";
import { CartItem } from "@/components/cart/cart-item";
import { CartSummary } from "@/components/cart/cart-summary";
import { SiteHeader } from "@/components/layout/site-header";
import { Kicker } from "@/components/ui/kicker";
import { ROUTES } from "@/lib/constants";

export default function CartPage() {
  const { items, clear } = useCart();
  const { confirm } = useConfirmation();

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

  if (items.length === 0) {
    return (
      <main className="bg-bone text-ink min-h-screen">
        <SiteHeader />
        <div className="max-w-7xl mx-auto px-6 md:px-10 py-16 md:py-24">
          <div className="border border-dashed border-line rounded-2xl p-10 md:p-16 text-center">
            <Kicker as="p" className="mb-3">
              Cart empty
            </Kicker>
            <p className="font-display text-3xl md:text-4xl font-medium tracking-tight text-ink">
              Nothing in here yet.
            </p>
            <p className="font-sans text-base text-ink-soft mt-4 max-w-md mx-auto">
              Find your race photos and add them to your cart.
            </p>
            <Link
              href={ROUTES.EVENTS}
              className="mt-8 inline-flex items-center gap-2 font-sans text-base font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-3 px-6 rounded-full transition-colors"
            >
              Browse events
              <span aria-hidden="true">→</span>
            </Link>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="bg-bone text-ink min-h-screen">
      <SiteHeader />
      <div className="max-w-7xl mx-auto px-6 md:px-10 py-10 md:py-14">
        <div className="flex items-baseline justify-between border-b border-line pb-4 mb-8 gap-4 flex-wrap">
          <div className="min-w-0">
            <Kicker as="p" tnum>
              {items.length} in cart
            </Kicker>
            <h1 className="font-display text-3xl md:text-4xl font-medium tracking-tight text-ink mt-2">
              Your cart
            </h1>
          </div>
          <button
            type="button"
            onClick={handleClearCart}
            className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-error hover:text-error transition-colors"
          >
            Clear all
          </button>
        </div>

        <div className="grid gap-8 lg:grid-cols-3">
          <div className="space-y-3 lg:col-span-2">
            {items.map((item) => (
              <CartItem
                key={item.photoId}
                photoId={item.photoId}
                thumbnailUrl={item.thumbnailUrl}
                price={item.price}
              />
            ))}
          </div>
          <CartSummary />
        </div>
      </div>
    </main>
  );
}
