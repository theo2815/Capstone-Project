"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useCartStore } from "@/store/cart-store";
import { useAuthStore } from "@/store/auth-store";
import { useConfirmationStore } from "@/store/confirmation-store";
import { usePendingPaymentStore } from "@/store/pending-payment-store";
import { useConfirmation } from "@/hooks/use-confirmation";
import { ROUTES } from "@/lib/constants";
import { useScrollLock } from "@/lib/scroll-lock";
import { cn, formatOrderRef, formatPrice, safeUUID } from "@/lib/utils";
import {
  buildOrderReturnPath,
  cancelPendingPayment,
  classifyExpired,
  fetchPendingStatus,
  postOrder,
  type OrderStatusPayload,
} from "@/lib/api-orders";
import { postCouponPreview, type CouponPreview } from "@/lib/api-coupons";
import { ApiError, formatRetryWait } from "@/lib/api";
import { Kicker } from "@/components/ui/kicker";
import {
  BTN_GHOST,
  BTN_PRIMARY,
  BTN_SECONDARY,
  BTN_SIZE,
} from "@/components/ui/button-styles";
import { FieldError } from "@/components/ui/field-error";
import type { CartItem } from "@/types/order";

// Poll cadence while a QR is on screen. The plain read is cheap (one row);
// `verify` asks PayMongo and is throttled server-side, so it only runs on
// the runner's word ("I've paid" / "Check again") and every VERIFY_EVERY_MS
// after that.
const STATUS_POLL_MS = 2500;
const VERIFY_EVERY_MS = 20_000;
const MANUAL_COOLDOWN_MS = 20_000;
// After this long past "I've paid" with no confirmation the copy switches to
// "taking longer than usual" — banks usually report back within 10–30 s.
const SLOW_AFTER_MS = 60_000;
// The expiry countdown turns amber under this much time left.
const EXPIRY_WARN_MS = 5 * 60_000;

// "h:mm am/pm" in the runner's locale — QR validity, confirm dialog, paid-at.
const formatClock = (iso: string) =>
  new Intl.DateTimeFormat("en-PH", { hour: "numeric", minute: "2-digit" }).format(
    new Date(iso),
  );

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

// Why the runner is back on the pay step. Rendered as one notice above the
// order summary — the first thing a returning customer reads.
type Outcome =
  | { kind: "expired" }
  | { kind: "failed" }
  | { kind: "cancelled" }
  | { kind: "error"; message: string };

interface QrPayment {
  orderId: string;
  imageUrl: string;
  expiresAt: string;
  returnToken: string | null;
  testUrl: string | null;
}

// What the success step shows — snapshotted before the cart and the pending
// record are cleared, since both are gone by the time it renders.
interface Receipt {
  orderId: string;
  returnToken: string | null;
  email: string;
  total: number;
  itemCount: number;
  paidAt: string | null;
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
  const confirmActive = useConfirmationStore((s) => s.active !== null);
  const { confirm } = useConfirmation();
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

  useScrollLock(isOpen);

  const itemCount = items.length;
  const recipientEmail = isAuthenticated ? authUser?.email ?? email : email;
  const discountTotal = coupon?.discountTotal ?? 0;
  const payable = Math.max(0, total - discountTotal);
  // While a QR is live the cart may already be empty (resume after refresh),
  // so the amount on screen comes from the record.
  const qrTotal = pending?.total ?? payable;
  const qrEmail = pending?.email ?? recipientEmail ?? "";

  // Every dismissal path (backdrop, ✕, Esc) routes through here so the policy
  // lives in one place: free before payment and after success, locked while
  // the order is being created, and a real question while a QR is live —
  // money may already have moved, so leaving is intentional, never a slip.
  // Leaving keeps the pending record; the cart pill brings the runner back.
  const requestClose = useCallback(async () => {
    if (step === "processing" || cancelling) return;
    if (step === "qr" && qrPayment && !checkUnreachable) {
      const ok = await confirm({
        title: "Leave checkout?",
        message: (
          <>
            Your QR code stays valid until{" "}
            <span className="text-ink tnum">{formatClock(qrPayment.expiresAt)}</span>.
            If you&rsquo;ve already paid, your photos and receipt are safe
            &mdash; we&rsquo;ll email them to{" "}
            <span className="text-ink break-all">{qrEmail}</span>. You can come
            back to this payment from the cart.
          </>
        ),
        confirmLabel: "Leave",
        cancelLabel: "Stay",
      });
      if (!ok) return;
    }
    onClose();
  }, [step, cancelling, qrPayment, checkUnreachable, confirm, qrEmail, onClose]);

  // Gated while the confirmation dialog is up so one Esc closes only the
  // topmost layer (ui-pitfalls 2026-05-06 stacked-modal rule).
  useEffect(() => {
    if (!isOpen || confirmActive) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") void requestClose();
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
    };
  }, [isOpen, confirmActive, requestClose]);

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
    if (!idempotencyKey) {
      setOutcome({ kind: "error", message: "Checkout is still loading. Try again." });
      return;
    }
    setOutcome(null);
    setStep("processing");

    try {
      const order = await postOrder({
        items: items.map((i) => ({ photoId: i.photoId, eventId: i.eventId })),
        paymentMethod: "qrph",
        recipientEmail: isAuthenticated ? undefined : email,
        couponCode: coupon?.code,
        idempotencyKey,
      });

      if (order.status === "PAID" || order.status === "FULFILLED") {
        // Idempotent replay of an already-paid order (e.g. paid, then the
        // record was lost). Same landing as the polled success.
        setReceipt({
          orderId: order.id,
          returnToken: order.qrPh?.returnToken ?? null,
          email: recipientEmail ?? "",
          total: payable,
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
        total: payable,
        itemCount,
        paidClaimedAt: null,
      });
      setStep("qr");
    } catch (err) {
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
      paymentCheckInFlight.current = true;
      setCheckingPayment(true);
      try {
        const status = await fetchPendingStatus(qrPayment, { verify: opts.verify });
        setQrCheckError(null);
        if (status.status === "PAID" || status.status === "FULFILLED") {
          completeFromStatus(status, qrPayment);
        } else if (status.status === "EXPIRED") {
          resetToPayment({ kind: classifyExpired(qrPayment.expiresAt) });
        }
      } catch (err) {
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
        setCheckingPayment(false);
      }
    },
    [clearPending, completeFromStatus, qrPayment, resetToPayment],
  );

  const pollingActive =
    step === "qr" && qrPayment !== null && !checkUnreachable && !cancelling;

  // Cancel is a real question — money may already be moving — and then a
  // server round-trip: the backend asks PayMongo once more, so a payment
  // that landed a moment ago wins and we show success instead of "cancelled".
  const handleCancelPayment = async () => {
    if (!qrPayment) return;
    const ok = await confirm({
      title: "Cancel this payment?",
      message: (
        <>
          The QR code stops being tied to this order and nothing is charged.
          Already paid? Choose <span className="text-ink">Keep waiting</span>{" "}
          &mdash; we&rsquo;ll confirm it in a moment.
        </>
      ),
      confirmLabel: "Cancel payment",
      cancelLabel: "Keep waiting",
      danger: true,
    });
    if (!ok) return;
    setCancelling(true);
    setQrCheckError(null);
    try {
      const status = await cancelPendingPayment(qrPayment);
      if (status.status === "PAID" || status.status === "FULFILLED") {
        completeFromStatus(status, qrPayment);
      } else {
        resetToPayment({ kind: "cancelled" });
      }
    } catch {
      setQrCheckError("We couldn't cancel right now. Your QR code is still live — try again.");
    } finally {
      setCancelling(false);
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

  const handlePaidClaim = () => {
    markPaid();
    setLastManualCheckAt(Date.now());
    void checkPaymentStatus({ verify: true, manual: true });
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
        onClick={() => void requestClose()}
        aria-label="Close checkout"
        disabled={step === "processing" || cancelling}
        className="absolute inset-0 bg-ink/55 backdrop-blur-sm cursor-default"
        style={{ animation: "fade-in 0.25s ease-out both" }}
      />

      <aside
        className="absolute top-0 right-0 h-full w-full sm:max-w-md flex flex-col bg-bone shadow-[-30px_0_80px_-20px_rgba(0,0,0,0.45)]"
        style={{ animation: "slide-in-right 0.35s ease-out both" }}
      >
        <header className="flex items-start justify-between gap-3 px-6 md:px-7 pt-6 pb-5 border-b border-line">
          <div className="min-w-0">
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink-soft mb-1.5">
              Checkout
            </p>
            <p className="font-display text-2xl md:text-3xl font-medium text-ink tracking-tight leading-tight">
              {step === "identify"
                ? "Where should we send your photos?"
                : step === "payment"
                  ? "Review & pay"
                  : step === "processing"
                    ? "Creating your QR code…"
                    : step === "qr"
                      ? checkUnreachable
                        ? "Check your inbox."
                        : paidClaimedAt
                          ? "Confirming your payment."
                          : "Scan to pay."
                    : "All yours."}
            </p>
            <StepIndicator step={step} isAuthenticated={isAuthenticated} />
          </div>
          <button
            type="button"
            onClick={() => void requestClose()}
            disabled={step === "processing" || cancelling}
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
              outcome={outcome}
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
          {step === "qr" && qrPayment && checkUnreachable && (
            <UnreachableNotice
              email={qrEmail}
              orderId={qrPayment.orderId}
              onDone={onClose}
            />
          )}
          {step === "qr" && qrPayment && !checkUnreachable && (
            <QrPaymentStep
              payment={qrPayment}
              total={qrTotal}
              email={qrEmail}
              paidClaimedAt={paidClaimedAt}
              lastManualCheckAt={lastManualCheckAt}
              checking={checkingPayment}
              cancelling={cancelling}
              checkError={qrCheckError}
              onPaid={handlePaidClaim}
              onCheck={handlePaidClaim}
              onCancel={() => void handleCancelPayment()}
              onLeaveForEmail={onClose}
            />
          )}
          {step === "success" && receipt && (
            <SuccessStep
              receipt={receipt}
              hasAccount={isAuthenticated}
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
  // Order carries meaning here — the runner really is walking a sequence —
  // so the rail is numbered and named, not just coloured bars.
  const steps = isAuthenticated
    ? (["payment", "qr", "success"] as const)
    : (["identify", "payment", "qr", "success"] as const);
  const currentIdx = steps.findIndex((s) =>
    step === "processing" ? s === "payment" : s === step,
  );
  return (
    <div className="mt-4 flex items-center gap-3">
      <div className="flex items-center gap-1.5" aria-hidden="true">
        {steps.map((s, i) => (
          <span
            key={s}
            className={cn(
              "h-[3px] w-6 rounded-full transition-colors duration-300",
              i <= currentIdx ? "bg-fresh" : "bg-line",
            )}
          />
        ))}
      </div>
      <Kicker as="p" tone="soft" tnum>
        Step {currentIdx + 1} of {steps.length} · {STEP_LABEL[steps[currentIdx]]}
      </Kicker>
    </div>
  );
}

const STEP_LABEL: Record<Exclude<Step, "processing">, string> = {
  identify: "Contact",
  payment: "Review & pay",
  qr: "Pay",
  success: "Done",
};

// The one shape every payment outcome takes, wherever it shows: dot, state
// kicker, headline, one sentence, at most one action. Tones stay inside the
// existing palette — nothing new for the runner to learn.
type NoticeTone = "waiting" | "slow" | "neutral" | "error";

const NOTICE_TONE: Record<NoticeTone, { box: string; dot: string }> = {
  waiting: { box: "border-fresh/40 bg-fresh/10", dot: "bg-fresh animate-pulse" },
  slow: { box: "border-warning/50 bg-warning/10", dot: "bg-warning animate-pulse" },
  neutral: { box: "border-line bg-bone-deep", dot: "bg-slate-soft" },
  error: { box: "border-error/40 bg-error/10", dot: "bg-error" },
};

function PaymentNotice({
  tone,
  kicker,
  title,
  children,
  role,
}: {
  tone: NoticeTone;
  kicker?: string;
  title: string;
  children?: ReactNode;
  role?: "status" | "alert";
}) {
  const t = NOTICE_TONE[tone];
  return (
    <div
      role={role}
      aria-live={role === "alert" ? undefined : "polite"}
      className={cn("rounded-xl border px-5 py-4 flex items-start gap-3", t.box)}
    >
      <span className={cn("mt-1.5 size-2 rounded-full shrink-0", t.dot)} aria-hidden="true" />
      <div className="min-w-0">
        {kicker && (
          <Kicker as="p" tone="soft" tnum className="mb-1">
            {kicker}
          </Kicker>
        )}
        <p className="font-display text-base font-medium text-ink">{title}</p>
        {children}
      </div>
    </div>
  );
}

const OUTCOME_COPY: Record<Outcome["kind"], { tone: NoticeTone; kicker: string; title: string; body: string }> = {
  expired: {
    tone: "neutral",
    kicker: "Nothing was charged",
    title: "Your QR code expired",
    body: "No payment was detected before it ran out. Generate a new code to pay.",
  },
  failed: {
    tone: "error",
    kicker: "Nothing was charged",
    title: "Payment didn't go through",
    body: "Your bank or e-wallet didn't complete it. Try again, or use a different app.",
  },
  cancelled: {
    tone: "neutral",
    kicker: "Nothing was charged",
    title: "Payment cancelled",
    body: "Your cart is unchanged. Generate a new code whenever you're ready.",
  },
  error: {
    tone: "error",
    kicker: "Try again",
    title: "Couldn't create a QR code",
    body: "",
  },
};

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
        Your receipt and full-resolution download links go to this email and
        nowhere else — so we ask for it twice.
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
        placeholder="Type it again"
      />

      <button type="submit" className={cn(BTN_PRIMARY, BTN_SIZE.md, "w-full")}>
        Continue →
      </button>

      <p className="text-center font-sans text-sm text-ink-soft">
        Have an account?{" "}
        <Link
          href={`${ROUTES.LOGIN}?redirect=${encodeURIComponent(resumeUrl)}`}
          className="font-semibold text-ink underline decoration-line-strong underline-offset-4 transition-colors hover:decoration-fresh"
        >
          Log in
        </Link>{" "}
        or{" "}
        <Link
          href={`${ROUTES.REGISTER}?redirect=${encodeURIComponent(resumeUrl)}`}
          className="font-semibold text-ink underline decoration-line-strong underline-offset-4 transition-colors hover:decoration-fresh"
        >
          Sign up
        </Link>
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
    </form>
  );
}

function PaymentStep({
  email,
  isAuthenticated,
  outcome,
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
  outcome: Outcome | null;
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
      {outcome && (
        <PaymentNotice
          tone={OUTCOME_COPY[outcome.kind].tone}
          kicker={OUTCOME_COPY[outcome.kind].kicker}
          title={OUTCOME_COPY[outcome.kind].title}
          role={OUTCOME_COPY[outcome.kind].tone === "error" ? "alert" : "status"}
        >
          <p className="mt-1 font-sans text-sm leading-relaxed text-ink-soft">
            {outcome.kind === "error" ? outcome.message : OUTCOME_COPY[outcome.kind].body}
          </p>
        </PaymentNotice>
      )}

      <section>
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink-soft mb-2">
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
          >
            <label
              htmlFor="coupon-code"
              className="block font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink-soft mb-2"
            >
              Coupon code · optional
            </label>
            <div className="flex items-stretch gap-2">
              <input
                id="coupon-code"
                value={couponInput}
                onChange={(e) => onCouponInputChange(e.target.value)}
                placeholder="From a photographer's photo card"
                autoComplete="off"
                aria-invalid={Boolean(couponError)}
                aria-describedby={couponError ? "coupon-code-err" : undefined}
                className="h-11 min-w-0 flex-1 rounded-full border border-line bg-surface px-4 font-sans text-sm text-ink outline-none transition-colors placeholder:text-slate-soft focus:border-fresh"
              />
              <button
                type="submit"
                disabled={couponBusy || couponInput.trim().length === 0}
                className={cn(BTN_SECONDARY, "h-11 shrink-0 px-5 text-sm")}
              >
                {couponBusy ? "Checking…" : "Apply"}
              </button>
            </div>
            <FieldError
              id="coupon-code-err"
              message={couponError}
              density="tight"
            />
          </form>
        )}
      </section>

      <section>
        <div className="flex items-baseline justify-between gap-3 mb-2">
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink-soft">
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
        <p className="font-sans text-sm font-semibold text-ink truncate" title={email}>
          {email || "—"}
        </p>
        {isAuthenticated && (
          <p className="mt-1 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
            Signed-in account
          </p>
        )}
      </section>

      <section>
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink-soft mb-3">
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
      </section>

      <button
        type="button"
        onClick={onPay}
        className={cn(BTN_PRIMARY, BTN_SIZE.md, "w-full")}
      >
        {outcome && outcome.kind !== "error" ? "Generate a new QR" : "Generate QR to pay"}{" "}
        <span className="tnum">{formatPrice(total)}</span> →
      </button>

      <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft text-center -mt-3">
        Nothing is charged until you scan · Watermark removed on download
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

type WaitMode = "awaiting" | "confirming" | "slow";

function formatElapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function QrPaymentStep({
  payment,
  total,
  email,
  paidClaimedAt,
  lastManualCheckAt,
  checking,
  cancelling,
  checkError,
  onPaid,
  onCheck,
  onCancel,
  onLeaveForEmail,
}: {
  payment: QrPayment;
  total: number;
  email: string;
  paidClaimedAt: string | null;
  lastManualCheckAt: number;
  checking: boolean;
  cancelling: boolean;
  checkError: string | null;
  onPaid: () => void;
  onCheck: () => void;
  onCancel: () => void;
  onLeaveForEmail: () => void;
}) {
  // 1 s clock for the whole QR step — drives the expiry countdown, and once
  // the runner says they've paid, the elapsed kicker, the slow-tier switch,
  // and the manual-check cooldown.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const elapsedMs = paidClaimedAt ? now - new Date(paidClaimedAt).getTime() : 0;
  const mode: WaitMode = !paidClaimedAt
    ? "awaiting"
    : elapsedMs < SLOW_AFTER_MS
      ? "confirming"
      : "slow";
  const cooldownLeft = Math.max(0, lastManualCheckAt + MANUAL_COOLDOWN_MS - now);
  const remainingMs = new Date(payment.expiresAt).getTime() - now;
  const expiringSoon = remainingMs < EXPIRY_WARN_MS;
  const ref = formatOrderRef(payment.orderId);

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
        <p
          className={cn(
            "mt-3 font-mono uppercase tracking-[0.14em] text-[13px] tnum",
            expiringSoon ? "text-warning" : "text-slate-soft",
          )}
        >
          {remainingMs > 0
            ? `Expires in ${formatElapsed(remainingMs)}`
            : `Expired ${formatClock(payment.expiresAt)}`}{" "}
          · Ref {ref}
        </p>
      </div>

      {payment.testUrl && <TestModePanel testUrl={payment.testUrl} />}

      <a
        href={payment.imageUrl}
        download={`quickpitik-qrph-${payment.orderId}.png`}
        className={cn(BTN_SECONDARY, BTN_SIZE.md, "w-full justify-center")}
      >
        Save QR code
      </a>

      <PaymentNotice
        tone={mode === "slow" ? "slow" : "waiting"}
        kicker={
          mode === "awaiting"
            ? undefined
            : `${mode === "slow" ? "Still confirming" : "Confirming"} · ${formatElapsed(elapsedMs)}`
        }
        title={
          mode === "awaiting"
            ? "Waiting for your payment"
            : mode === "confirming"
              ? "Confirming your payment…"
              : "Taking longer than usual"
        }
      >
        {mode === "awaiting" && (
          <p className="mt-1 font-sans text-sm leading-relaxed text-ink-soft">
            Scan with your bank or e-wallet app. We check automatically every
            few seconds — no need to refresh.
          </p>
        )}
        {mode === "confirming" && (
          <>
            <p className="mt-1 font-sans text-sm leading-relaxed text-ink-soft">
              Banks usually take <span className="tnum">10–30</span> seconds to
              report back. <span className="font-semibold text-ink">Don&rsquo;t pay again</span>{" "}
              — your <span className="tnum text-ink">{formatPrice(total)}</span> is tied to
              Ref <span className="tnum text-ink">{ref}</span>.
            </p>
            <p className="mt-2 font-sans text-sm leading-relaxed text-ink-soft">
              Safe to refresh or close — we&rsquo;ll pick up where you left off,
              and your receipt goes to{" "}
              <span className="text-ink break-all">{email}</span> either way.
            </p>
          </>
        )}
        {mode === "slow" && (
          <p className="mt-1 font-sans text-sm leading-relaxed text-ink-soft">
            Your payment is still safe. If it went through, this page will
            confirm it and we&rsquo;ll email your download links to{" "}
            <span className="text-ink break-all">{email}</span> — even if you
            leave. Please don&rsquo;t pay a second time.
          </p>
        )}
      </PaymentNotice>

      {mode === "awaiting" && (
        <section className="rounded-xl border border-line bg-bone-deep px-5 py-4">
          <p className="font-mono uppercase tracking-[0.14em] text-[13px] text-ink-soft mb-3">
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
              <span>
                Confirm the exact amount in your app, then tap{" "}
                <span className="text-ink">I&rsquo;ve paid</span> below. We
                detect the payment automatically either way.
              </span>
            </li>
          </ol>
        </section>
      )}

      {checkError && (
        <p role="alert" className="font-sans text-sm text-error">
          {checkError}
        </p>
      )}

      {mode === "awaiting" ? (
        <button
          type="button"
          onClick={onPaid}
          disabled={checking}
          className={cn(BTN_PRIMARY, BTN_SIZE.md, "w-full justify-center")}
        >
          I&rsquo;ve paid
        </button>
      ) : (
        <button
          type="button"
          onClick={onCheck}
          disabled={checking || cooldownLeft > 0}
          className={cn(BTN_SECONDARY, BTN_SIZE.md, "w-full justify-center tnum")}
        >
          {checking
            ? "Checking…"
            : cooldownLeft > 0
              ? `Checking again in ${Math.ceil(cooldownLeft / 1000)}s`
              : "Check again"}
        </button>
      )}

      {mode === "slow" && (
        <button
          type="button"
          onClick={onLeaveForEmail}
          className={cn(BTN_GHOST, BTN_SIZE.sm, "self-center")}
        >
          I&rsquo;ll wait for the email →
        </button>
      )}

      {/* The way out. Quiet on purpose — it sits under the payment actions,
          never beside them — but always there, so a pending screen is never
          a trap. Hidden mid-check so it can't race our own poll. */}
      {!checking && (
        <button
          type="button"
          onClick={onCancel}
          disabled={cancelling}
          className={cn(BTN_GHOST, BTN_SIZE.sm, "self-center")}
        >
          {cancelling ? "Cancelling…" : "Cancel payment"}
        </button>
      )}
    </div>
  );
}

// Dev only: PayMongo returns this link solely in test mode, and the backend
// forwards it solely on an sk_test_ key, so this panel cannot render in
// production. In test mode the QR itself is still a *real* QR Ph code —
// scanning it moves real money — which is exactly why the panel exists.
function TestModePanel({ testUrl }: { testUrl: string }) {
  return (
    <section className="rounded-xl border border-dashed border-line-strong bg-bone px-5 py-4">
      <Kicker as="p" tnum className="mb-1">
        Test mode · PayMongo sandbox
      </Kicker>
      <p className="font-sans text-sm leading-relaxed text-ink-soft">
        Don&rsquo;t scan this code with a real app — it would charge you. Use
        the simulator to authorize or fail this payment, then come back and tap{" "}
        <span className="text-ink">I&rsquo;ve paid</span>.
      </p>
      <a
        href={testUrl}
        target="_blank"
        rel="noopener noreferrer"
        className={cn(BTN_SECONDARY, BTN_SIZE.sm, "mt-3 inline-flex")}
      >
        Open PayMongo simulator ↗
      </a>
    </section>
  );
}

// The return token is dead (35-minute guest window) or the order vanished:
// polling is over, but the backend still emails the receipt on settlement.
function UnreachableNotice({
  email,
  orderId,
  onDone,
}: {
  email: string;
  orderId: string;
  onDone: () => void;
}) {
  return (
    <div className="px-6 md:px-7 py-6 flex flex-col gap-6">
      <div className="rounded-xl border border-line bg-bone-deep px-5 py-4">
        <p className="font-display text-base font-medium text-ink">
          We can&rsquo;t check this payment from here any more.
        </p>
        <p className="mt-1 font-sans text-sm leading-relaxed text-ink-soft">
          If you paid, your receipt and download links are on their way to{" "}
          <span className="text-ink break-all">{email}</span>. If you
          didn&rsquo;t, nothing was charged and you can start a fresh checkout.
        </p>
        <Kicker as="p" tone="soft" tnum className="mt-4">
          Ref · {formatOrderRef(orderId)}
        </Kicker>
      </div>
      <button
        type="button"
        onClick={onDone}
        className={cn(BTN_SECONDARY, BTN_SIZE.md, "w-full justify-center")}
      >
        Done
      </button>
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
  receipt,
  hasAccount,
  onDone,
}: {
  receipt: Receipt;
  hasAccount: boolean;
  onDone: () => void;
}) {
  const { email, itemCount, total, orderId, paidAt } = receipt;
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
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink-soft mb-1">
              Payment confirmed{paidAt ? ` · ${formatClock(paidAt)}` : ""}
            </p>
            <p className="font-display text-xl font-medium text-ink tracking-tight leading-tight">
              <span className="tnum">{itemCount}</span>{" "}
              {itemCount === 1 ? "photo" : "photos"} ·{" "}
              <span className="tnum">{formatPrice(total)}</span>
            </p>
          </div>
        </div>
        <p className="mt-5 pt-4 border-t border-line font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
          Ref · <span className="text-ink tnum">{formatOrderRef(orderId)}</span>
        </p>
      </div>

      <div>
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink-soft mb-2">
          Download links
        </p>
        <p className="font-sans text-sm text-ink-soft leading-relaxed">
          Your receipt with full-resolution download links is on its way to{" "}
          <span className="text-ink break-all">{email}</span> — usually within
          a minute.
          {hasAccount ? (
            <>
              {" "}
              Re-download anytime from{" "}
              <Link
                href={ROUTES.ORDERS}
                onClick={onDone}
                className="text-ink underline decoration-fresh underline-offset-4 hover:text-fresh transition-colors"
              >
                Orders
              </Link>
              .
            </>
          ) : null}
        </p>
      </div>

      <Link
        href={buildOrderReturnPath(receipt)}
        onClick={onDone}
        className={cn(BTN_PRIMARY, BTN_SIZE.md, "w-full justify-center")}
      >
        View receipt &amp; download →
      </Link>

      <button
        type="button"
        onClick={onDone}
        className={cn(BTN_SECONDARY, BTN_SIZE.md, "w-full justify-center")}
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
      <span className="block font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink-soft mb-2">
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
