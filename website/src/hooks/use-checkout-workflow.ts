"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useCartStore } from "@/store/cart-store";
import { useAuthStore } from "@/store/auth-store";
import { usePendingPaymentStore } from "@/store/pending-payment-store";
import { useCheckoutActions } from "@/hooks/use-checkout-actions";
import { classifyExpired, type OrderStatusPayload } from "@/lib/api-orders";
import { postCouponPreview, type CouponPreview } from "@/lib/api-coupons";
import { ApiError, formatRetryWait } from "@/lib/api";
import { captureAuthSession } from "@/lib/auth-session";
import { safeUUID } from "@/lib/utils";

// Plain status reads are cheap; provider verification runs after a paid claim.
const STATUS_POLL_MS = 2500;
const VERIFY_EVERY_MS = 20_000;
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export type Step = "identify" | "payment" | "processing" | "qr" | "success";

// Why the runner is back on the pay step. Rendered as one notice above the
// order summary — the first thing a returning customer reads.
export type Outcome =
  | { kind: "expired" }
  | { kind: "failed" }
  | { kind: "cancelled" }
  | { kind: "already_purchased" }
  | { kind: "error"; message: string };

export interface QrPayment {
  orderId: string;
  imageUrl: string;
  expiresAt: string;
  returnToken: string | null;
  testUrl: string | null;
}

// What the success step shows — snapshotted before the cart and the pending
// record are cleared, since both are gone by the time it renders.
export interface Receipt {
  orderId: string;
  returnToken: string | null;
  email: string;
  total: number;
  itemCount: number;
  paidAt: string | null;
}

/** Owns checkout state and effects; the modal owns presentation and confirmation. */
export function useCheckoutWorkflow(isOpen: boolean) {
  const items = useCartStore((s) => s.items);
  const total = useCartStore((s) => s.total());
  const clearCart = useCartStore((s) => s.clear);
  const removeItem = useCartStore((s) => s.removeItem);
  const queryClient = useQueryClient();
  const checkoutActions = useCheckoutActions();

  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const authUser = useAuthStore((s) => s.user);

  const initialStep: Step = isAuthenticated ? "payment" : "identify";
  const [step, setStep] = useState<Step>(initialStep);

  const [email, setEmail] = useState(authUser?.email ?? "");
  const [confirmEmail, setConfirmEmail] = useState("");
  const [errors, setErrors] = useState<{ email?: string; confirm?: string }>({});

  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [qrPayment, setQrPayment] = useState<QrPayment | null>(null);
  const [qrCheckError, setQrCheckError] = useState<string | null>(null);
  const [checkingPayment, setCheckingPayment] = useState(false);
  // Cancel in flight: polling pauses and the drawer locks like `processing`,
  // because the backend is asking PayMongo whether the payment already won.
  const [cancelling, setCancelling] = useState(false);
  // The return token died (35-min guest window) or the order vanished: we
  // can't poll from here any more, but the backend still emails the receipt.
  const [checkUnreachable, setCheckUnreachable] = useState(false);
  const [receipt, setReceipt] = useState<Receipt | null>(null);
  // Bumped by every manual check so the automatic verify timer restarts and
  // the button cools down — keeps verify traffic under the server bucket.
  const [lastManualCheckAt, setLastManualCheckAt] = useState(0);
  const paymentCheckInFlight = useRef(false);

  const pending = usePendingPaymentStore((s) => s.pending);
  const setPending = usePendingPaymentStore((s) => s.set);
  const markPaid = usePendingPaymentStore((s) => s.markPaid);
  const clearPending = usePendingPaymentStore((s) => s.clear);
  const paidClaimedAt = pending?.paidClaimedAt ?? null;
  // Q-008 RESOLVED: one client-generated UUID per checkout attempt.
  const [idempotencyKey, setIdempotencyKey] = useState<string | null>(null);
  // Photographer coupon (V45). `coupon` is the server's priced preview for
  // the current cart — the only source of the discount the modal shows.
  const [coupon, setCoupon] = useState<CouponPreview | null>(null);
  const [couponInput, setCouponInput] = useState("");
  const [couponError, setCouponError] = useState<string | null>(null);
  const [couponBusy, setCouponBusy] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    setConfirmEmail("");
    setErrors({});
    setOutcome(null);
    setQrCheckError(null);
    setCheckingPayment(false);
    setCancelling(false);
    setCheckUnreachable(false);
    setReceipt(null);
    setIdempotencyKey(safeUUID());
    // A coupon was previewed against a specific cart; re-entry may follow a
    // cart edit, so it has to be re-applied.
    setCoupon(null);
    setCouponInput("");
    setCouponError(null);
    // A live QR (refresh, closed tab, accidental dismissal, auth flip) resumes
    // where it left off instead of resetting — the record is the source of
    // truth, not React state.
    const live = usePendingPaymentStore.getState().pending;
    if (live) {
      setEmail(live.email);
      setQrPayment({
        orderId: live.orderId,
        imageUrl: live.imageUrl,
        expiresAt: live.expiresAt,
        returnToken: live.returnToken,
        testUrl: live.testUrl ?? null,
      });
      setStep("qr");
      return;
    }
    setStep(isAuthenticated ? "payment" : "identify");
    setEmail(authUser?.email ?? "");
    setQrPayment(null);
  }, [isOpen, isAuthenticated, authUser?.email]);

  const itemCount = items.length;
  const recipientEmail = isAuthenticated ? authUser?.email ?? email : email;
  const discountTotal = coupon?.discountTotal ?? 0;
  const payable = Math.max(0, Math.round((total - discountTotal) * 100) / 100);
  // While a QR is live the cart may already be empty (resume after refresh),
  // so the amount on screen comes from the record.
  const qrTotal = pending?.total ?? payable;
  const qrEmail = pending?.email ?? recipientEmail ?? "";

  const handleApplyCoupon = async () => {
    const code = couponInput.trim().toUpperCase();
    if (!code) {
      setCouponError("Enter a code.");
      return;
    }
    setCouponBusy(true);
    try {
      const preview = await postCouponPreview({
        code,
        photoIds: items.map((i) => i.photoId),
      });
      setCoupon(preview);
      setCouponError(null);
      // Same reasoning as a payment-method change: a different discount is
      // a different intent and must not dedupe against a prior attempt.
      if (idempotencyKey) setIdempotencyKey(safeUUID());
    } catch (err) {
      setCouponError(
        err instanceof ApiError ? err.message : "Couldn't check that code. Try again.",
      );
    } finally {
      setCouponBusy(false);
    }
  };

  const handleRemoveCoupon = () => {
    setCoupon(null);
    setCouponInput("");
    setCouponError(null);
    if (idempotencyKey) setIdempotencyKey(safeUUID());
  };

  // Auto-apply (2026-09-05): on the pay step the server prices the cart with
  // every photographer's own live coupon — nothing typed. A typed code
  // replaces this quote until it is removed; a cart edit re-quotes.
  const photoIdsKey = items.map((i) => i.photoId).join(",");
  const typedCode = coupon?.code ?? null;
  useEffect(() => {
    if (!isOpen || step !== "payment" || typedCode || !photoIdsKey) return;
    let cancelled = false;
    postCouponPreview({ photoIds: photoIdsKey.split(",") })
      .then((quote) => {
        if (!cancelled) setCoupon(quote);
      })
      .catch(() => {
        // Display only — the charge is priced server-side regardless.
        if (!cancelled) setCoupon(null);
      });
    return () => {
      cancelled = true;
    };
  }, [isOpen, step, typedCode, photoIdsKey]);

  const handleIdentifySubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = email.trim();
    const trimmedConfirm = confirmEmail.trim();
    const next: { email?: string; confirm?: string } = {};
    if (!trimmed) next.email = "Email is required.";
    else if (!EMAIL_REGEX.test(trimmed)) next.email = "Enter a valid email.";
    // The receipt and download links go nowhere else, so a typo here is the
    // one mistake the runner can't recover from — hence the second field.
    if (!trimmedConfirm) next.confirm = "Type your email again to confirm.";
    else if (trimmed.toLowerCase() !== trimmedConfirm.toLowerCase())
      next.confirm = "Emails don't match.";
    setErrors(next);
    if (!next.email && !next.confirm) {
      setEmail(trimmed);
      setStep("payment");
    }
  };

  const handlePay = async () => {
    const isCurrentSession = captureAuthSession();
    if (!idempotencyKey) {
      setOutcome({ kind: "error", message: "Checkout is still loading. Try again." });
      return;
    }
    setOutcome(null);
    setStep("processing");

    try {
      const order = await checkoutActions.createOrder({
        items: items.map((i) => ({ photoId: i.photoId, eventId: i.eventId })),
        paymentMethod: "qrph",
        recipientEmail: isAuthenticated ? undefined : email,
        // Only a typed code travels; automatic coupons are the server's call.
        ...(coupon?.code ? { couponCode: coupon.code } : {}),
        idempotencyKey,
      });
      if (!isCurrentSession()) return;
      if (!order) {
        setStep("payment");
        return;
      }

      if (order.status === "PAID" || order.status === "FULFILLED") {
        // Idempotent replay of an already-paid order (e.g. paid, then the
        // record was lost). Same landing as the polled success.
        setReceipt({
          orderId: order.id,
          returnToken: order.returnToken ?? order.qrPh?.returnToken ?? null,
          email: recipientEmail ?? "",
          total: order.totalAmount,
          itemCount,
          paidAt: null,
        });
        clearCart();
        clearPending();
        queryClient.invalidateQueries({ queryKey: ["me", "orders"] });
        setStep("success");
        return;
      }
      if (!order.qrPh) {
        setOutcome({ kind: "error", message: "The payment provider didn't return a QR code." });
        setStep("payment");
        return;
      }
      const qr: QrPayment = {
        orderId: order.id,
        imageUrl: order.qrPh.imageUrl,
        expiresAt: order.qrPh.expiresAt,
        returnToken: order.qrPh.returnToken ?? null,
        testUrl: order.qrPh.testUrl ?? null,
      };
      setQrPayment(qr);
      setPending({
        ...qr,
        email: recipientEmail ?? "",
        total: order.totalAmount,
        itemCount,
        paidClaimedAt: null,
      });
      setStep("qr");
    } catch (err) {
      // The client usually catches an already-owned photo in the cart, but a
      // guest (no /me/orders) or a photo bought mid-checkout can still trip the
      // server gate — brand that 409 instead of the generic QR-failure copy.
      if (!isCurrentSession()) return;
      if (
        err instanceof ApiError &&
        err.errors.some((e) => e.code === "ALREADY_PURCHASED")
      ) {
        // Refresh owned photos so the just-detected duplicates get flagged in
        // the summary and the pay button gates — the runner sees which to drop.
        queryClient.invalidateQueries({ queryKey: ["me", "orders"] });
        setOutcome({ kind: "already_purchased" });
        setStep("payment");
        return;
      }
      setOutcome({
        kind: "error",
        message:
          err instanceof ApiError
            ? err.message
            : "We couldn't reach the payment provider. Try again in a moment.",
      });
      setStep("payment");
    }
  };

  // A settled status, whichever path delivered it (poll, verify, or a cancel
  // that lost the race): snapshot the receipt before the cart and the record
  // are cleared, then land on success.
  const completeFromStatus = useCallback(
    (status: OrderStatusPayload, qr: QrPayment) => {
      const live = usePendingPaymentStore.getState().pending;
      setReceipt({
        orderId: qr.orderId,
        returnToken: qr.returnToken,
        email: live?.email ?? recipientEmail ?? "",
        total: live?.total ?? payable,
        itemCount: live?.itemCount ?? itemCount,
        paidAt: status.paidAt,
      });
      clearCart();
      clearPending();
      queryClient.invalidateQueries({ queryKey: ["me", "orders"] });
      setStep("success");
    },
    [clearCart, clearPending, itemCount, payable, queryClient, recipientEmail],
  );

  // The QR is finished without a payment (expired, declined, or cancelled):
  // drop the record, mint a fresh key so the next attempt is a new order, and
  // tell the runner why they're back on the pay step.
  const resetToPayment = useCallback(
    (why: Outcome) => {
      clearPending();
      setQrPayment(null);
      setIdempotencyKey(safeUUID());
      setOutcome(why);
      setStep("payment");
    },
    [clearPending],
  );

  const checkPaymentStatus = useCallback(
    async (opts: { verify?: boolean; manual?: boolean } = {}) => {
      if (!qrPayment || paymentCheckInFlight.current) return;
      const isCurrentSession = captureAuthSession();
      paymentCheckInFlight.current = true;
      setCheckingPayment(true);
      try {
        const status = await checkoutActions.checkStatus(qrPayment, { verify: opts.verify });
        if (!isCurrentSession() || !status) return;
        setQrCheckError(null);
        if (status.status === "PAID" || status.status === "FULFILLED") {
          completeFromStatus(status, qrPayment);
        } else if (status.status === "EXPIRED") {
          resetToPayment({ kind: classifyExpired(qrPayment.expiresAt) });
        }
      } catch (err) {
        if (!isCurrentSession()) return;
        if (err instanceof ApiError && err.status === 404) {
          // Token dead or order gone — stop asking; the email is the backstop.
          clearPending();
          setCheckUnreachable(true);
        } else if (err instanceof ApiError && err.status === 429 && opts.manual) {
          setQrCheckError(
            `Too many checks — try again in ${formatRetryWait(err.retryAfterSeconds ?? 20)}. We're still checking automatically.`,
          );
        } else {
          setQrCheckError("We couldn't check yet. Your QR is still safe to use.");
        }
      } finally {
        paymentCheckInFlight.current = false;
        if (isCurrentSession()) setCheckingPayment(false);
      }
    },
    [checkoutActions, clearPending, completeFromStatus, qrPayment, resetToPayment],
  );

  const pollingActive =
    step === "qr" && qrPayment !== null && !checkUnreachable && !cancelling;

  // The modal confirms intent before calling this action. A payment that
  // settles during cancellation still wins and lands on success.
  const handleCancelPayment = async () => {
    if (!qrPayment) return;
    const isCurrentSession = captureAuthSession();
    setCancelling(true);
    setQrCheckError(null);
    try {
      const status = await checkoutActions.cancelPayment(qrPayment);
      if (!isCurrentSession() || !status) return;
      if (status.status === "PAID" || status.status === "FULFILLED") {
        completeFromStatus(status, qrPayment);
      } else {
        resetToPayment({ kind: "cancelled" });
      }
    } catch {
      if (!isCurrentSession()) return;
      setQrCheckError("We couldn't cancel right now. Your QR code is still live — try again.");
    } finally {
      if (isCurrentSession()) setCancelling(false);
    }
  };

  // Cheap DB read on a fixed cadence for as long as the QR is on screen.
  useEffect(() => {
    if (!pollingActive) return;
    void checkPaymentStatus();
    const timer = window.setInterval(() => void checkPaymentStatus(), STATUS_POLL_MS);
    return () => window.clearInterval(timer);
  }, [checkPaymentStatus, pollingActive]);

  // Once the runner says they've paid, also ask PayMongo directly every
  // VERIFY_EVERY_MS. A manual check restarts this timer (dependency below).
  useEffect(() => {
    if (!pollingActive || !paidClaimedAt) return;
    const timer = window.setInterval(
      () => void checkPaymentStatus({ verify: true }),
      VERIFY_EVERY_MS,
    );
    return () => window.clearInterval(timer);
  }, [checkPaymentStatus, pollingActive, paidClaimedAt, lastManualCheckAt]);

  // At the QR's own expiry, one verify makes the backend flip the order to
  // EXPIRED right away instead of on the next minute sweep.
  useEffect(() => {
    if (!pollingActive || !qrPayment) return;
    const delay = Math.max(0, new Date(qrPayment.expiresAt).getTime() - Date.now() + 2000);
    const timer = window.setTimeout(() => void checkPaymentStatus({ verify: true }), delay);
    return () => window.clearTimeout(timer);
  }, [checkPaymentStatus, pollingActive, qrPayment]);

  // Returning from the bank/e-wallet app (tab re-shown) verifies once, actively,
  // so a payment that settled while we were backgrounded flips to success right
  // away instead of on the next poll or the 60 s server sweep. In-app webviews
  // suspend JS timers while backgrounded, so the interval poll alone lags here.
  useEffect(() => {
    if (!pollingActive) return;
    const onVisible = () => {
      if (document.visibilityState === "visible") {
        void checkPaymentStatus({ verify: true });
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
  }, [checkPaymentStatus, pollingActive]);

  const handlePaidClaim = () => {
    markPaid();
    setLastManualCheckAt(Date.now());
    void checkPaymentStatus({ verify: true, manual: true });
  };

  return {
    items, removeItem, clearCart, isAuthenticated, step, setStep,
    email, setEmail, confirmEmail, setConfirmEmail, errors, outcome,
    qrPayment, qrCheckError, checkingPayment, cancelling, checkUnreachable,
    receipt, lastManualCheckAt, paidClaimedAt, itemCount, recipientEmail,
    payable, qrTotal, qrEmail, coupon, couponInput, setCouponInput,
    couponError, setCouponError, couponBusy, handleApplyCoupon,
    handleRemoveCoupon, handleIdentifySubmit, handlePay, handleCancelPayment,
    handlePaidClaim,
  };
}
