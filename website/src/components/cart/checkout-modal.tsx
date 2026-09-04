"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useCartStore } from "@/store/cart-store";
import { useAuthStore } from "@/store/auth-store";
import { ROUTES } from "@/lib/constants";
import { useScrollLock } from "@/lib/scroll-lock";
import { cn, formatPrice, safeUUID } from "@/lib/utils";
import {
  fetchOrderStatus,
  fetchOrderStatusForUser,
  postOrder,
} from "@/lib/api-orders";
import { postCouponPreview, type CouponPreview } from "@/lib/api-coupons";
import { ApiError } from "@/lib/api";
import { Kicker } from "@/components/ui/kicker";
import { BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import type { CartItem } from "@/types/order";

// Build the post-login resume URL: original page + `?checkout=1` flag.
// `<CheckoutResumeWatcher>` reads the flag on mount and re-opens the modal,
// so the user lands back on the page where they invoked checkout instead
// of the legacy `/cart` page.
function buildResumeUrl(pathname: string, sp: URLSearchParams | null): string {
  const params = new URLSearchParams(sp ?? undefined);
  params.set("checkout", "1");
  const queryStr = params.toString();
  return queryStr ? `${pathname}?${queryStr}` : pathname;
}

type Step = "identify" | "payment" | "processing" | "qr" | "success";

interface QrPayment {
  orderId: string;
  imageUrl: string;
  expiresAt: string;
  returnToken?: string | null;
}

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

interface CheckoutModalProps {
  isOpen: boolean;
  onClose: () => void;
  onBackToCart?: () => void;
}

export function CheckoutModal({
  isOpen,
  onClose,
  onBackToCart,
}: CheckoutModalProps) {
  const items = useCartStore((s) => s.items);
  const total = useCartStore((s) => s.total());
  const clearCart = useCartStore((s) => s.clear);
  const queryClient = useQueryClient();

  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const authUser = useAuthStore((s) => s.user);

  const pathname = usePathname() ?? ROUTES.HOME;
  const searchParams = useSearchParams();
  const resumeUrl = buildResumeUrl(pathname, searchParams);

  const initialStep: Step = isAuthenticated ? "payment" : "identify";
  const [step, setStep] = useState<Step>(initialStep);

  const [email, setEmail] = useState(authUser?.email ?? "");
  const [confirmEmail, setConfirmEmail] = useState("");
  const [errors, setErrors] = useState<{ email?: string; confirm?: string }>(
    {},
  );

  const [paymentError, setPaymentError] = useState<string | null>(null);
  const [orderId, setOrderId] = useState<string | null>(null);
  const [qrPayment, setQrPayment] = useState<QrPayment | null>(null);
  const [qrCheckError, setQrCheckError] = useState<string | null>(null);
  const [checkingPayment, setCheckingPayment] = useState(false);
  const paymentCheckInFlight = useRef(false);
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
    setStep(isAuthenticated ? "payment" : "identify");
    setEmail(authUser?.email ?? "");
    setConfirmEmail("");
    setErrors({});
    setPaymentError(null);
    setOrderId(null);
    setQrPayment(null);
    setQrCheckError(null);
    setIdempotencyKey(safeUUID());
    // A coupon was previewed against a specific cart; re-entry may follow a
    // cart edit, so it has to be re-applied.
    setCoupon(null);
    setCouponInput("");
    setCouponError(null);
  }, [isOpen, isAuthenticated, authUser?.email]);

  useScrollLock(isOpen);

  // Esc is gated on `step` for the same reason the backdrop and the header ✕
  // are: closing mid-`processing` drops the in-flight postOrder's orderId (the
  // isOpen reset effect above nulls it on re-entry) and the user is left with
  // a charge they can't see a reference for.
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && step !== "processing") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
    };
  }, [isOpen, step, onClose]);

  const itemCount = items.length;
  const recipientEmail = isAuthenticated ? authUser?.email ?? email : email;
  const discountTotal = coupon?.discountTotal ?? 0;
  const payable = Math.max(0, total - discountTotal);

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

  const handleIdentifySubmit = (e: FormEvent) => {
    e.preventDefault();
    const next: { email?: string; confirm?: string } = {};
    const trimmed = email.trim();
    const trimmedConfirm = confirmEmail.trim();
    if (!trimmed) next.email = "Email is required.";
    else if (!EMAIL_REGEX.test(trimmed)) next.email = "Enter a valid email.";
    if (!trimmedConfirm) next.confirm = "Please confirm your email.";
    else if (trimmed.toLowerCase() !== trimmedConfirm.toLowerCase())
      next.confirm = "Emails don't match.";
    setErrors(next);
    if (Object.keys(next).length === 0) {
      setEmail(trimmed);
      setStep("payment");
    }
  };

  const handlePay = async () => {
    if (!idempotencyKey) {
      setPaymentError("Checkout is still loading. Try again.");
      return;
    }
    setPaymentError(null);
    setStep("processing");

    try {
      const order = await postOrder({
        items: items.map((i) => ({ photoId: i.photoId, eventId: i.eventId })),
        paymentMethod: "qrph",
        recipientEmail: isAuthenticated ? undefined : email,
        couponCode: coupon?.code,
        idempotencyKey,
      });

      setOrderId(order.id);
      if (order.status === "PAID" || order.status === "FULFILLED") {
        clearCart();
        queryClient.invalidateQueries({ queryKey: ["me", "orders"] });
        setStep("success");
        return;
      }
      if (!order.qrPh) {
        setPaymentError("QR code unavailable. Try again.");
        setStep("payment");
        return;
      }
      setQrPayment({ orderId: order.id, ...order.qrPh });
      setStep("qr");
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "Payment failed. Try again in a moment.";
      setPaymentError(message);
      setStep("payment");
    }
  };

  const checkPaymentStatus = useCallback(async () => {
    if (!qrPayment || paymentCheckInFlight.current) return;
    paymentCheckInFlight.current = true;
    setCheckingPayment(true);
    try {
      const status = isAuthenticated
        ? await fetchOrderStatusForUser(qrPayment.orderId)
        : qrPayment.returnToken
          ? await fetchOrderStatus(qrPayment.orderId, qrPayment.returnToken)
          : null;
      if (!status) throw new Error("Missing payment status token");
      setQrCheckError(null);
      if (status.status === "PAID" || status.status === "FULFILLED") {
        clearCart();
        queryClient.invalidateQueries({ queryKey: ["me", "orders"] });
        setStep("success");
      } else if (status.status === "EXPIRED") {
        setQrPayment(null);
        setIdempotencyKey(safeUUID());
        setPaymentError("That QR code expired. Generate a new one to continue.");
        setStep("payment");
      }
    } catch {
      setQrCheckError("We couldn't check yet. Your QR is still safe to use.");
    } finally {
      paymentCheckInFlight.current = false;
      setCheckingPayment(false);
    }
  }, [clearCart, isAuthenticated, qrPayment, queryClient]);

  useEffect(() => {
    if (step !== "qr" || !qrPayment) return;
    void checkPaymentStatus();
    const timer = window.setInterval(() => void checkPaymentStatus(), 2500);
    return () => window.clearInterval(timer);
  }, [checkPaymentStatus, qrPayment, step]);

  const handleSuccessClose = () => {
    clearCart();
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Checkout"
      className="fixed inset-0 z-50"
    >
      <button
        type="button"
        onClick={step === "processing" ? () => {} : onClose}
        aria-label="Close checkout"
        disabled={step === "processing"}
        className="absolute inset-0 bg-ink/55 backdrop-blur-sm cursor-default"
        style={{ animation: "fade-in 0.25s ease-out both" }}
      />

      <aside
        className="absolute top-0 right-0 h-full w-full sm:max-w-md flex flex-col bg-bone shadow-[-30px_0_80px_-20px_rgba(0,0,0,0.45)]"
        style={{ animation: "slide-in-right 0.35s ease-out both" }}
      >
        <header className="flex items-start justify-between gap-3 px-6 md:px-7 pt-6 pb-5 border-b border-line">
          <div className="min-w-0">
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate mb-1.5">
              Checkout
            </p>
            <p className="font-display text-2xl md:text-3xl font-medium text-ink tracking-tight leading-tight">
              {step === "identify"
                ? "Where should we send them?"
                : step === "payment"
                  ? "Pay with QR Ph."
                  : step === "processing"
                    ? "Generating your QR…"
                    : step === "qr"
                      ? "Scan. Pay. Done."
                    : "All yours."}
            </p>
            <StepIndicator step={step} isAuthenticated={isAuthenticated} />
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={step === "processing"}
            aria-label="Close checkout"
            className="size-9 shrink-0 rounded-full border border-line text-ink hover:bg-bone-deep flex items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent"
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

        <div className="flex-1 overflow-y-auto">
          {step === "identify" && (
            <IdentifyStep
              email={email}
              confirmEmail={confirmEmail}
              errors={errors}
              onEmailChange={setEmail}
              onConfirmChange={setConfirmEmail}
              onSubmit={handleIdentifySubmit}
              onBackToCart={onBackToCart}
              resumeUrl={resumeUrl}
            />
          )}
          {step === "payment" && (
            <PaymentStep
              email={recipientEmail ?? ""}
              isAuthenticated={isAuthenticated}
              paymentError={paymentError}
              total={payable}
              itemCount={itemCount}
              items={items}
              coupon={coupon}
              couponInput={couponInput}
              onCouponInputChange={(v) => {
                setCouponInput(v.toUpperCase());
                setCouponError(null);
              }}
              couponError={couponError}
              couponBusy={couponBusy}
              onApplyCoupon={() => void handleApplyCoupon()}
              onRemoveCoupon={handleRemoveCoupon}
              onPay={handlePay}
              onEditEmail={
                isAuthenticated ? undefined : () => setStep("identify")
              }
              onBackToCart={isAuthenticated ? onBackToCart : undefined}
            />
          )}
          {step === "processing" && <ProcessingStep />}
          {step === "qr" && qrPayment && (
            <QrPaymentStep
              payment={qrPayment}
              total={payable}
              checking={checkingPayment}
              checkError={qrCheckError}
              onCheck={() => void checkPaymentStatus()}
            />
          )}
          {step === "success" && (
            <SuccessStep
              email={recipientEmail ?? ""}
              orderId={orderId}
              itemCount={itemCount}
              total={payable}
              onDone={handleSuccessClose}
            />
          )}
        </div>
      </aside>
    </div>
  );
}

function StepIndicator({
  step,
  isAuthenticated,
}: {
  step: Step;
  isAuthenticated: boolean;
}) {
  const steps = isAuthenticated
    ? (["payment", "qr", "success"] as const)
    : (["identify", "payment", "qr", "success"] as const);
  const currentIdx = steps.findIndex((s) =>
    step === "processing" ? s === "payment" : s === step,
  );
  return (
    <div className="mt-4 flex items-center gap-1.5" aria-hidden="true">
      {steps.map((s, i) => (
        <span
          key={s}
          className={cn(
            "h-[3px] flex-1 max-w-[28px] rounded-full transition-colors duration-300",
            i <= currentIdx ? "bg-fresh" : "bg-line",
          )}
        />
      ))}
    </div>
  );
}

function IdentifyStep({
  email,
  confirmEmail,
  errors,
  onEmailChange,
  onConfirmChange,
  onSubmit,
  onBackToCart,
  resumeUrl,
}: {
  email: string;
  confirmEmail: string;
  errors: { email?: string; confirm?: string };
  onEmailChange: (v: string) => void;
  onConfirmChange: (v: string) => void;
  onSubmit: (e: FormEvent) => void;
  onBackToCart?: () => void;
  resumeUrl: string;
}) {
  return (
    <form
      onSubmit={onSubmit}
      className="px-6 md:px-7 py-6 flex flex-col gap-7"
      noValidate
    >
      <p className="font-sans text-sm text-ink-soft leading-relaxed -mt-1">
        We&apos;ll send your purchase receipt and full-resolution download links
        to this email.
      </p>

      <Field
        id="checkout-email"
        label="Email"
        type="email"
        autoComplete="email"
        value={email}
        onChange={onEmailChange}
        error={errors.email}
        placeholder="you@email.com"
        autoFocus
      />
      <Field
        id="checkout-confirm-email"
        label="Confirm email"
        type="email"
        autoComplete="off"
        value={confirmEmail}
        onChange={onConfirmChange}
        error={errors.confirm}
        placeholder="you@email.com"
      />

      <button
        type="submit"
        className="inline-flex w-full items-center justify-center gap-1.5 bg-fresh hover:bg-fresh-deep text-surface px-6 py-3.5 rounded-full font-display font-bold text-[15px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
      >
        Continue →
      </button>

      <div className="relative">
        <div className="absolute inset-0 flex items-center">
          <span className="w-full border-t border-line" />
        </div>
        <div className="relative flex justify-center">
          <span className="bg-bone px-3 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
            Have an account?
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <Link
          href={`${ROUTES.LOGIN}?redirect=${encodeURIComponent(resumeUrl)}`}
          className="inline-flex items-center justify-center border border-ink hover:bg-ink hover:text-bone text-ink px-4 py-3 rounded-full font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] transition-colors"
        >
          Log in
        </Link>
        <Link
          href={`${ROUTES.REGISTER}?redirect=${encodeURIComponent(resumeUrl)}`}
          className="inline-flex items-center justify-center border border-line hover:bg-bone-deep text-ink px-4 py-3 rounded-full font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] transition-colors"
        >
          Sign up
        </Link>
      </div>

      {onBackToCart && (
        <button
          type="button"
          onClick={onBackToCart}
          className="self-start font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors"
        >
          ← Back to cart
        </button>
      )}
    </form>
  );
}

function PaymentStep({
  email,
  isAuthenticated,
  paymentError,
  total,
  itemCount,
  items,
  coupon,
  couponInput,
  onCouponInputChange,
  couponError,
  couponBusy,
  onApplyCoupon,
  onRemoveCoupon,
  onPay,
  onEditEmail,
  onBackToCart,
}: {
  email: string;
  isAuthenticated: boolean;
  paymentError: string | null;
  /** What will be charged — already net of any applied coupon. */
  total: number;
  itemCount: number;
  items: CartItem[];
  coupon: CouponPreview | null;
  couponInput: string;
  onCouponInputChange: (v: string) => void;
  couponError: string | null;
  couponBusy: boolean;
  onApplyCoupon: () => void;
  onRemoveCoupon: () => void;
  onPay: () => void;
  onEditEmail?: () => void;
  onBackToCart?: () => void;
}) {
  const discountFor = (photoId: string) =>
    coupon?.items.find((c) => c.photoId === photoId)?.discount ?? null;

  return (
    <div className="px-6 md:px-7 py-6 flex flex-col gap-7">
      <section>
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate mb-2">
          Order summary
        </p>
        {coupon ? (
          // With a code applied the summary opens up per photo, so the runner
          // never has to guess which pictures the discount reached.
          <ul className="rounded-xl border border-line bg-bone-deep divide-y divide-line">
            {items.map((item) => {
              const discount = discountFor(item.photoId);
              return (
                <li key={item.photoId} className="flex items-center gap-3 px-4 py-3">
                  {item.thumbnailUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={item.thumbnailUrl}
                      alt=""
                      className="size-10 rounded-md object-cover shrink-0 bg-line"
                    />
                  ) : (
                    <span aria-hidden="true" className="size-10 rounded-md bg-line shrink-0" />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="font-sans text-sm text-ink truncate">
                      {item.bib ? `Bib ${item.bib}` : "Untagged photo"}
                      {item.eventName ? (
                        <span className="text-slate-soft"> · {item.eventName}</span>
                      ) : null}
                    </p>
                    <Kicker as="p" tone="soft" tnum className="truncate">
                      {discount != null
                        ? `${coupon.code} · −${formatPrice(discount)}`
                        : "Not covered by this code"}
                    </Kicker>
                  </div>
                  <div className="text-right shrink-0">
                    {discount != null ? (
                      <>
                        <p className="font-mono tnum text-sm text-slate-soft line-through">
                          {formatPrice(item.price)}
                        </p>
                        <p className="font-mono tnum font-semibold text-ink">
                          {formatPrice(item.price - discount)}
                        </p>
                      </>
                    ) : (
                      <p className="font-mono tnum font-semibold text-ink">
                        {formatPrice(item.price)}
                      </p>
                    )}
                  </div>
                </li>
              );
            })}
            <li className="flex items-baseline justify-between gap-3 px-4 py-3">
              <span className="font-sans text-sm text-ink-soft">
                Total · <span className="tnum text-ink">{itemCount}</span>{" "}
                {itemCount === 1 ? "photo" : "photos"}
              </span>
              <span className="font-display text-2xl md:text-3xl font-medium text-ink tnum">
                {formatPrice(total)}
              </span>
            </li>
          </ul>
        ) : (
          <div className="rounded-xl border border-line bg-bone-deep px-5 py-4 flex items-baseline justify-between gap-3">
            <span className="font-sans text-sm text-ink-soft">
              <span className="tnum text-ink">{itemCount}</span>{" "}
              {itemCount === 1 ? "photo" : "photos"} · full resolution
            </span>
            <span className="font-display text-2xl md:text-3xl font-medium text-ink tnum">
              {formatPrice(total)}
            </span>
          </div>
        )}
      </section>

      <section>
        {coupon ? (
          <div className="rounded-xl border border-line bg-bone px-5 py-4 flex items-center justify-between gap-3">
            <div className="min-w-0">
              <p className="font-mono font-semibold tnum text-ink truncate">{coupon.code}</p>
              <p className="font-sans text-sm text-ink-soft mt-0.5">
                <span className="tnum">{coupon.percentOff}%</span> off{" "}
                <span className="tnum">{coupon.eligibleCount}</span> of{" "}
                <span className="tnum">{itemCount}</span>{" "}
                {itemCount === 1 ? "photo" : "photos"}
                {coupon.photographerName ? ` · photos by ${coupon.photographerName}` : ""}
              </p>
            </div>
            <button
              type="button"
              onClick={onRemoveCoupon}
              className={cn(BTN_GHOST, BTN_SIZE.sm, "shrink-0")}
            >
              Remove
            </button>
          </div>
        ) : (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              onApplyCoupon();
            }}
            className="flex items-end gap-3"
          >
            <div className="flex-1 min-w-0">
              <Field
                id="coupon-code"
                label="Coupon code · optional"
                value={couponInput}
                onChange={onCouponInputChange}
                error={couponError ?? undefined}
                placeholder="From a photographer's photo card"
                autoComplete="off"
              />
            </div>
            <button
              type="submit"
              disabled={couponBusy || couponInput.trim().length === 0}
              className={cn(BTN_SECONDARY, BTN_SIZE.sm, "shrink-0")}
            >
              {couponBusy ? "Checking…" : "Apply"}
            </button>
          </form>
        )}
      </section>

      <section>
        <div className="flex items-baseline justify-between gap-3 mb-2">
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
            Send to
          </p>
          {onEditEmail && (
            <button
              type="button"
              onClick={onEditEmail}
              className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft hover:text-fresh transition-colors"
            >
              Edit
            </button>
          )}
        </div>
        <p className="font-sans text-sm text-ink truncate" title={email}>
          {email || "—"}
        </p>
        {isAuthenticated && (
          <p className="mt-1 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
            Signed-in account
          </p>
        )}
      </section>

      <section>
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate mb-3">
          Payment method
        </p>
        <div className="rounded-xl border border-fresh bg-bone-deep px-5 py-4 flex items-start gap-4">
          <span
            className="mt-0.5 size-9 rounded-full bg-fresh text-surface shrink-0 inline-flex items-center justify-center font-mono text-sm font-bold"
            aria-hidden="true"
          >
            QR
          </span>
          <div>
            <p className="font-display text-base font-medium text-ink">QR Ph</p>
            <p className="mt-1 font-sans text-sm leading-relaxed text-ink-soft">
              Pay from any participating bank or e-wallet app that can scan QR Ph codes.
            </p>
          </div>
        </div>
        {paymentError && (
          <p
            role="alert"
            className="mt-3 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-error"
          >
            {paymentError}
          </p>
        )}
      </section>

      <button
        type="button"
        onClick={onPay}
        className="inline-flex w-full items-center justify-center gap-1.5 bg-fresh hover:bg-fresh-deep text-surface px-6 py-3.5 rounded-full font-display font-bold text-[15px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
      >
        Generate QR to pay <span className="tnum">{formatPrice(total)}</span> →
      </button>

      <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft text-center -mt-3">
        Secure transaction · Watermark removed on download
      </p>

      {onBackToCart && (
        <button
          type="button"
          onClick={onBackToCart}
          className="self-start font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors"
        >
          ← Back to cart
        </button>
      )}
    </div>
  );
}

function QrPaymentStep({
  payment,
  total,
  checking,
  checkError,
  onCheck,
}: {
  payment: QrPayment;
  total: number;
  checking: boolean;
  checkError: string | null;
  onCheck: () => void;
}) {
  const expiresAt = new Intl.DateTimeFormat("en-PH", {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(payment.expiresAt));

  return (
    <div className="px-6 md:px-7 py-6 flex flex-col gap-6">
      <div className="text-center">
        <Kicker as="p" tone="soft" tnum>
          Amount due · {formatPrice(total)}
        </Kicker>
        <div className="mx-auto mt-3 w-fit rounded-2xl border border-line bg-white p-3 shadow-sm">
          {/* PayMongo returns the QR as a trusted image data URI. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={payment.imageUrl}
            alt={`QR Ph payment code for ${formatPrice(total)}`}
            className="size-[248px] max-w-full object-contain"
          />
        </div>
        <p className="mt-3 font-mono uppercase tracking-[0.14em] text-[13px] text-slate-soft tnum">
          Valid until {expiresAt}
        </p>
      </div>

      <a
        href={payment.imageUrl}
        download={`quickpitik-qrph-${payment.orderId}.png`}
        className={cn(BTN_SECONDARY, BTN_SIZE.md, "w-full justify-center")}
      >
        Save QR code
      </a>

      <div
        aria-live="polite"
        className="rounded-xl border border-fresh/40 bg-fresh/10 px-5 py-4 flex items-start gap-3"
      >
        <span className="mt-1.5 size-2 rounded-full bg-fresh animate-pulse shrink-0" aria-hidden="true" />
        <div>
          <p className="font-display text-base font-medium text-ink">
            {checking ? "Checking payment…" : "Waiting for payment"}
          </p>
          <p className="mt-1 font-sans text-sm leading-relaxed text-ink-soft">
            Keep this checkout open after paying. Confirmation can take a few seconds.
          </p>
        </div>
      </div>

      <section className="rounded-xl border border-line bg-bone-deep px-5 py-4">
        <p className="font-mono uppercase tracking-[0.14em] text-[13px] text-slate mb-3">
          How to pay
        </p>
        <ol className="space-y-3 font-sans text-sm leading-relaxed text-ink-soft">
          <li className="flex gap-3">
            <span className="font-mono tnum text-fresh">01</span>
            Open a participating bank or e-wallet app and choose Scan QR.
          </li>
          <li className="flex gap-3">
            <span className="font-mono tnum text-fresh">02</span>
            Scan this code, or save it and open the image on another device.
          </li>
          <li className="flex gap-3">
            <span className="font-mono tnum text-fresh">03</span>
            Confirm the exact amount in your app, then return here. We&apos;ll detect the payment automatically.
          </li>
        </ol>
      </section>

      {checkError && (
        <p role="alert" className="font-sans text-sm text-error">
          {checkError}
        </p>
      )}

      <button
        type="button"
        onClick={onCheck}
        disabled={checking}
        className="inline-flex w-full items-center justify-center bg-ink hover:bg-ink-soft text-bone px-6 py-3.5 rounded-full font-mono uppercase tracking-[0.14em] text-[13px] transition-colors disabled:opacity-60"
      >
        {checking ? "Checking…" : "I've paid · Check status"}
      </button>

      <Kicker as="p" tone="soft" tnum className="text-center break-all">
        Reference · {payment.orderId}
      </Kicker>
    </div>
  );
}

function ProcessingStep() {
  return (
    <div className="flex-1 min-h-[60vh] flex flex-col items-center justify-center px-8 py-12 text-center">
      <span
        className="size-10 rounded-full border-2 border-line border-t-fresh animate-spin mb-7"
        aria-hidden="true"
      />
      <p className="font-display text-xl font-medium text-ink mb-2">
        Creating your QR Ph code…
      </p>
      <p className="font-sans text-sm text-ink-soft max-w-xs">
        Hold tight — this usually takes a few seconds.
      </p>
    </div>
  );
}

function SuccessStep({
  email,
  orderId,
  itemCount,
  total,
  onDone,
}: {
  email: string;
  orderId: string | null;
  itemCount: number;
  total: number;
  onDone: () => void;
}) {
  return (
    <div className="px-6 md:px-7 py-6 flex flex-col gap-7">
      <div className="rounded-2xl bg-bone-deep border border-line p-6">
        <div className="flex items-start gap-4">
          <span
            className="size-10 rounded-full bg-fresh text-surface shrink-0 inline-flex items-center justify-center"
            aria-hidden="true"
          >
            <svg viewBox="0 0 16 16" className="size-4" fill="none">
              <path
                d="M3 8 L7 12 L13 4"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
          <div className="min-w-0">
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate mb-1">
              Order placed
            </p>
            <p className="font-display text-xl font-medium text-ink tracking-tight leading-tight">
              <span className="tnum">{itemCount}</span>{" "}
              {itemCount === 1 ? "photo" : "photos"} ·{" "}
              <span className="tnum">{formatPrice(total)}</span>
            </p>
          </div>
        </div>
        {orderId && (
          <p className="mt-5 pt-4 border-t border-line font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
            Reference · <span className="text-ink tnum">{orderId}</span>
          </p>
        )}
      </div>

      <div>
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate mb-2">
          Download links
        </p>
        <p className="font-sans text-sm text-ink-soft leading-relaxed">
          We&apos;ve emailed full-resolution download links to{" "}
          <span className="text-ink break-all">{email}</span>. Links don&apos;t
          expire — re-download anytime from{" "}
          <Link
            href={ROUTES.ORDERS}
            onClick={onDone}
            className="text-ink underline decoration-fresh underline-offset-4 hover:text-fresh transition-colors"
          >
            Orders
          </Link>
          .
        </p>
      </div>

      <button
        type="button"
        onClick={onDone}
        className="inline-flex w-full items-center justify-center bg-ink hover:bg-ink-soft text-bone px-6 py-3.5 rounded-full font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] transition-colors"
      >
        Done
      </button>
    </div>
  );
}

function Field({
  id,
  label,
  type = "text",
  value,
  onChange,
  error,
  placeholder,
  autoComplete,
  autoFocus,
}: {
  id: string;
  label: string;
  type?: string;
  value: string;
  onChange: (v: string) => void;
  error?: string;
  placeholder?: string;
  autoComplete?: string;
  autoFocus?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <label htmlFor={id} className="block">
      <span className="block font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate mb-2">
        {label}
      </span>
      <input
        ref={inputRef}
        id={id}
        type={type}
        autoComplete={autoComplete}
        autoFocus={autoFocus}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        aria-invalid={Boolean(error)}
        aria-describedby={error ? `${id}-err` : undefined}
        className={cn(
          "block w-full bg-transparent outline-none font-sans text-base py-2.5 placeholder:text-slate-soft text-ink border-b transition-colors",
          error
            ? "border-error focus:border-error"
            : "border-line focus:border-fresh",
        )}
      />
      {error && (
        <span
          id={`${id}-err`}
          className="mt-2 block font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-error"
        >
          {error}
        </span>
      )}
    </label>
  );
}
