"use client";

import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ApiError } from "@/lib/api";
import {
  buildOrderReturnPath,
  classifyExpired,
  fetchPendingStatus,
} from "@/lib/api-orders";
import { formatOrderRef } from "@/lib/utils";
import { useCartStore } from "@/store/cart-store";
import { usePendingPaymentStore } from "@/store/pending-payment-store";
import { useToastStore } from "@/store/toast-store";
import { useUiStore } from "@/store/ui-store";

const POLL_INTERVAL_MS = 5000;
// A record whose QR expired this long ago with no answer is stale — the
// backend has already expired it or the return token is dead. Drop it; the
// receipt email is the backstop either way.
const STALE_AFTER_MS = 5 * 60_000;

// Headless. When the runner leaves the checkout with a QR still live, this
// keeps checking so the confirmation still reaches them on-page — a toast
// with the receipt link — instead of only by email. Idle while the checkout
// modal is open (the modal runs its own poll) and when nothing is pending.
export function PendingPaymentWatcher() {
  const pending = usePendingPaymentStore((s) => s.pending);
  const clearPending = usePendingPaymentStore((s) => s.clear);
  const checkoutOpen = useUiStore((s) => s.checkoutOpen);
  const openCheckout = useUiStore((s) => s.openCheckout);
  const clearCart = useCartStore((s) => s.clear);
  const showToast = useToastStore((s) => s.showToast);
  const queryClient = useQueryClient();

  // "Safe to refresh — we'll pick up where you left off": once per page load,
  // a live QR reopens the checkout on the QR step. Client-side navigation
  // never remounts this, so an intentional Leave stays left.
  const reopened = useRef(false);
  useEffect(() => {
    if (reopened.current || !pending) return;
    reopened.current = true;
    if (!checkoutOpen) openCheckout();
  }, [pending, checkoutOpen, openCheckout]);

  useEffect(() => {
    if (!pending || checkoutOpen) return;
    let cancelled = false;
    let inFlight = false;

    const tick = async () => {
      if (cancelled || inFlight) return;
      inFlight = true;
      try {
        const status = await fetchPendingStatus(pending);
        if (cancelled) return;
        if (status.status === "PAID" || status.status === "FULFILLED") {
          clearCart();
          clearPending();
          queryClient.invalidateQueries({ queryKey: ["me", "orders"] });
          showToast({
            kind: "success",
            message: `Payment confirmed · Ref ${formatOrderRef(pending.orderId)}`,
            link: { label: "View receipt", href: buildOrderReturnPath(pending) },
            duration: 10_000,
          });
        } else if (status.status === "EXPIRED") {
          clearPending();
          showToast({
            kind: "info",
            message:
              classifyExpired(pending.expiresAt) === "failed"
                ? "Your payment didn't go through — nothing was charged."
                : "Your QR code expired before a payment was detected — nothing was charged.",
            duration: 8_000,
          });
        } else if (
          Date.now() - new Date(pending.expiresAt).getTime() > STALE_AFTER_MS
        ) {
          clearPending();
        }
      } catch (err) {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) clearPending();
      } finally {
        inFlight = false;
      }
    };

    void tick();
    const timer = window.setInterval(() => void tick(), POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [pending, checkoutOpen, clearCart, clearPending, queryClient, showToast]);

  return null;
}
