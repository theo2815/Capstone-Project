"use client";

import { useEffect } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useAuthStore } from "@/store/auth-store";
import { useCartStore } from "@/store/cart-store";
import { useUiStore } from "@/store/ui-store";

// Pairs with the redirect built by `<CheckoutModal>`'s `IdentifyStep`. When
// a guest hits "Log in" / "Sign up" from inside the checkout modal, we send
// them to `/login?redirect=<original-page>?checkout=1`. After they auth,
// they land back on the original page with `?checkout=1` in the URL — this
// component spots the flag, re-opens the checkout modal, then strips the
// flag from the URL so a refresh doesn't keep re-opening it.
//
// Cart-empty guard: if the cart was cleared between login redirect and
// return (e.g. user wandered off, second tab cleared it), we still strip
// the flag but skip `openCheckout()` to avoid an empty-cart modal.
export function CheckoutResumeWatcher() {
  const router = useRouter();
  const pathname = usePathname() ?? "/";
  const searchParams = useSearchParams();
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const isLoading = useAuthStore((s) => s.isLoading);
  const cartItemCount = useCartStore((s) => s.items.length);
  const openCheckout = useUiStore((s) => s.openCheckout);

  useEffect(() => {
    if (isLoading) return;
    if (!isAuthenticated) return;
    if (!searchParams || searchParams.get("checkout") !== "1") return;

    const params = new URLSearchParams(searchParams);
    params.delete("checkout");
    const cleanQuery = params.toString();
    router.replace(pathname + (cleanQuery ? `?${cleanQuery}` : ""));

    if (cartItemCount > 0) {
      openCheckout();
    }
  }, [
    isLoading,
    isAuthenticated,
    searchParams,
    pathname,
    router,
    openCheckout,
    cartItemCount,
  ]);

  return null;
}
