"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useCartStore } from "@/store/cart-store";
import { useAuthStore } from "@/store/auth-store";
import { useOrdersStore } from "@/store/orders-store";
import { ROUTES } from "@/lib/constants";
import { useScrollLock } from "@/lib/scroll-lock";
import { cn } from "@/lib/utils";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import { postOrder } from "@/lib/api-orders";
import { ApiError } from "@/lib/api";
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

type Step = "identify" | "payment" | "processing" | "success";
type PaymentMethod = "gcash" | "maya" | "card";

const PAYMENT_OPTIONS: {
  id: PaymentMethod;
  label: string;
  hint: string;
}[] = [
  { id: "gcash", label: "GCash", hint: "Open the GCash app to confirm." },
  { id: "maya", label: "Maya", hint: "Open the Maya app to confirm." },
  { id: "card", label: "Credit / Debit card", hint: "Visa, Mastercard, or JCB." },
];

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
  const addOrder = useOrdersStore((s) => s.addOrder);
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

  const [paymentMethod, setPaymentMethod] = useState<PaymentMethod | null>(
    null,
  );
  const [paymentError, setPaymentError] = useState<string | null>(null);
  const [orderId, setOrderId] = useState<string | null>(null);
  // Q-008 RESOLVED: client-generated UUID per payment-method selection.
  // Re-generates on method change or modal re-entry; backend dedupes for 24 h.
  const [idempotencyKey, setIdempotencyKey] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    setStep(isAuthenticated ? "payment" : "identify");
    setEmail(authUser?.email ?? "");
    setConfirmEmail("");
    setErrors({});
    setPaymentMethod(null);
    setPaymentError(null);
    setOrderId(null);
    setIdempotencyKey(null);
  }, [isOpen, isAuthenticated, authUser?.email]);

  useScrollLock(isOpen);

  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
    };
  }, [isOpen, onClose]);

  const itemCount = items.length;
  const recipientEmail = isAuthenticated ? authUser?.email ?? email : email;

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
    if (!paymentMethod || !idempotencyKey) {
      setPaymentError("Pick a payment method.");
      return;
    }
    setPaymentError(null);
    setStep("processing");

    if (BACKEND_LIVE) {
      try {
        const order = await postOrder({
          items: items.map((i) => ({ photoId: i.photoId, eventId: i.eventId })),
          paymentMethod,
          recipientEmail: isAuthenticated ? undefined : email,
          idempotencyKey,
        });
        // Race log derives from saved-events ∪ orders — invalidate both so the
        // /profile race log picks up the new purchase.
        queryClient.invalidateQueries({ queryKey: ["me", "orders"] });
        queryClient.invalidateQueries({ queryKey: ["me", "saved-events"] });
        setOrderId(order.id);
        setStep("success");
      } catch (err) {
        const message =
          err instanceof ApiError
            ? err.message
            : "Payment failed. Try again in a moment.";
        setPaymentError(message);
        setStep("payment");
      }
      return;
    }

    // Mock mode — synthetic split per event so Race Log can derive counts
    // without a backend round-trip. CheckoutModal is the only mock-mode order
    // creator; live mode uses postOrder() above.
    setTimeout(() => {
      const id = `QP-${Date.now().toString(36).toUpperCase().slice(-6)}`;
      const method = paymentMethod;
      const paidAt = new Date().toISOString();
      const itemsByEvent = new Map<string, CartItem[]>();
      for (const item of items) {
        const list = itemsByEvent.get(item.eventId) ?? [];
        list.push(item);
        itemsByEvent.set(item.eventId, list);
      }
      let suffix = 0;
      for (const [eventId, eventItems] of itemsByEvent) {
        const splitId =
          itemsByEvent.size === 1 ? id : `${id}-${++suffix}`;
        addOrder({
          id: splitId,
          eventId,
          photoIds: eventItems.map((i) => i.photoId),
          total: eventItems.reduce((sum, i) => sum + i.price, 0),
          paymentMethod: method,
          paidAt,
        });
      }
      setOrderId(id);
      setStep("success");
    }, 1600);
  };

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
            <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate mb-1.5">
              Checkout
            </p>
            <p className="font-display text-2xl md:text-3xl font-medium text-ink tracking-tight leading-tight">
              {step === "identify"
                ? "Where should we send them?"
                : step === "payment"
                  ? "Pick how to pay."
                  : step === "processing"
                    ? "Confirming…"
                    : "All yours."}
            </p>
            <StepIndicator step={step} isAuthenticated={isAuthenticated} />
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={step === "processing"}
            aria-label="Close checkout"
            className="size-9 shrink-0 rounded-full border border-line text-ink hover:bg-bone-deep flex items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh disabled:opacity-40 disabled:hover:bg-transparent"
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
              paymentMethod={paymentMethod}
              onPaymentChange={(m) => {
                setPaymentMethod(m);
                setPaymentError(null);
                // Q-008: regenerate idempotencyKey on method change. Different
                // method = different intent, must not dedupe against prior.
                setIdempotencyKey(crypto.randomUUID());
              }}
              paymentError={paymentError}
              total={total}
              itemCount={itemCount}
              onPay={handlePay}
              onEditEmail={
                isAuthenticated ? undefined : () => setStep("identify")
              }
              onBackToCart={isAuthenticated ? onBackToCart : undefined}
            />
          )}
          {step === "processing" && <ProcessingStep />}
          {step === "success" && (
            <SuccessStep
              email={recipientEmail ?? ""}
              orderId={orderId}
              itemCount={itemCount}
              total={total}
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
    ? (["payment", "success"] as const)
    : (["identify", "payment", "success"] as const);
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
        className="inline-flex w-full items-center justify-center bg-fresh hover:bg-fresh-deep text-bone px-6 py-3.5 rounded-full font-mono uppercase tracking-[0.2em] text-[13px] min-[400px]:text-[14px] md:text-[12px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
      >
        Continue →
      </button>

      <div className="relative">
        <div className="absolute inset-0 flex items-center">
          <span className="w-full border-t border-line" />
        </div>
        <div className="relative flex justify-center">
          <span className="bg-bone px-3 font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
            Have an account?
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <Link
          href={`${ROUTES.LOGIN}?redirect=${encodeURIComponent(resumeUrl)}`}
          className="inline-flex items-center justify-center border border-ink hover:bg-ink hover:text-bone text-ink px-4 py-3 rounded-full font-mono uppercase tracking-[0.2em] text-[13px] min-[400px]:text-[14px] md:text-[12px] transition-colors"
        >
          Log in
        </Link>
        <Link
          href={`${ROUTES.REGISTER}?redirect=${encodeURIComponent(resumeUrl)}`}
          className="inline-flex items-center justify-center border border-line hover:bg-bone-deep text-ink px-4 py-3 rounded-full font-mono uppercase tracking-[0.2em] text-[13px] min-[400px]:text-[14px] md:text-[12px] transition-colors"
        >
          Sign up
        </Link>
      </div>

      {onBackToCart && (
        <button
          type="button"
          onClick={onBackToCart}
          className="self-start font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors"
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
  paymentMethod,
  onPaymentChange,
  paymentError,
  total,
  itemCount,
  onPay,
  onEditEmail,
  onBackToCart,
}: {
  email: string;
  isAuthenticated: boolean;
  paymentMethod: PaymentMethod | null;
  onPaymentChange: (m: PaymentMethod) => void;
  paymentError: string | null;
  total: number;
  itemCount: number;
  onPay: () => void;
  onEditEmail?: () => void;
  onBackToCart?: () => void;
}) {
  return (
    <div className="px-6 md:px-7 py-6 flex flex-col gap-7">
      <section>
        <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate mb-2">
          Order summary
        </p>
        <div className="rounded-xl border border-line bg-bone-deep px-5 py-4 flex items-baseline justify-between gap-3">
          <span className="font-sans text-sm text-ink-soft">
            <span className="tnum text-ink">{itemCount}</span>{" "}
            {itemCount === 1 ? "photo" : "photos"} · full resolution
          </span>
          <span className="font-display text-2xl md:text-3xl font-medium text-ink tnum">
            ₱{total.toLocaleString()}
          </span>
        </div>
      </section>

      <section>
        <div className="flex items-baseline justify-between gap-3 mb-2">
          <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate">
            Send to
          </p>
          {onEditEmail && (
            <button
              type="button"
              onClick={onEditEmail}
              className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft hover:text-fresh transition-colors"
            >
              Edit
            </button>
          )}
        </div>
        <p className="font-sans text-sm text-ink truncate" title={email}>
          {email || "—"}
        </p>
        {isAuthenticated && (
          <p className="mt-1 font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
            Signed-in account
          </p>
        )}
      </section>

      <section>
        <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate mb-3">
          Payment method
        </p>
        <ul className="space-y-2.5">
          {PAYMENT_OPTIONS.map((opt) => {
            const active = paymentMethod === opt.id;
            return (
              <li key={opt.id}>
                <button
                  type="button"
                  onClick={() => onPaymentChange(opt.id)}
                  aria-pressed={active}
                  className={cn(
                    "w-full text-left rounded-xl border px-5 py-4 transition-colors flex items-start gap-4",
                    active
                      ? "border-fresh bg-bone-deep"
                      : "border-line bg-bone hover:bg-bone-deep",
                  )}
                >
                  <span
                    className={cn(
                      "mt-0.5 size-4 rounded-full border-2 shrink-0 transition-colors",
                      active
                        ? "border-fresh bg-fresh"
                        : "border-line bg-bone",
                    )}
                    aria-hidden="true"
                  />
                  <span className="flex-1 min-w-0">
                    <span className="block font-display text-base font-medium text-ink leading-snug">
                      {opt.label}
                    </span>
                    <span className="block mt-0.5 font-sans text-sm text-ink-soft">
                      {opt.hint}
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
        {paymentError && (
          <p
            role="alert"
            className="mt-3 font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-error"
          >
            {paymentError}
          </p>
        )}
      </section>

      <button
        type="button"
        onClick={onPay}
        className="inline-flex w-full items-center justify-center bg-fresh hover:bg-fresh-deep text-bone px-6 py-3.5 rounded-full font-mono uppercase tracking-[0.2em] text-[13px] min-[400px]:text-[14px] md:text-[12px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
      >
        Pay ₱{total.toLocaleString()} →
      </button>

      <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft text-center -mt-3">
        Secure transaction · Watermark removed on download
      </p>

      {onBackToCart && (
        <button
          type="button"
          onClick={onBackToCart}
          className="self-start font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors"
        >
          ← Back to cart
        </button>
      )}
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
        Confirming your purchase…
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
            className="size-10 rounded-full bg-fresh text-bone shrink-0 inline-flex items-center justify-center"
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
            <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate mb-1">
              Order placed
            </p>
            <p className="font-display text-xl font-medium text-ink tracking-tight leading-tight">
              <span className="tnum">{itemCount}</span>{" "}
              {itemCount === 1 ? "photo" : "photos"} · ₱
              <span className="tnum">{total.toLocaleString()}</span>
            </p>
          </div>
        </div>
        {orderId && (
          <p className="mt-5 pt-4 border-t border-line font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
            Reference · <span className="text-ink tnum">{orderId}</span>
          </p>
        )}
      </div>

      <div>
        <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate mb-2">
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
        className="inline-flex w-full items-center justify-center bg-ink hover:bg-ink-soft text-bone px-6 py-3.5 rounded-full font-mono uppercase tracking-[0.2em] text-[13px] min-[400px]:text-[14px] md:text-[12px] transition-colors"
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
      <span className="block font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate mb-2">
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
          className="mt-2 block font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-error"
        >
          {error}
        </span>
      )}
    </label>
  );
}
