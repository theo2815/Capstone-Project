"use client";

import Link from "next/link";
import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import { SiteHeader } from "@/components/layout/site-header";
import { Kicker } from "@/components/ui/kicker";
import { useAuthStore } from "@/store/auth-store";
import { useCartStore } from "@/store/cart-store";
import { ApiError } from "@/lib/api";
import {
  buildOrderBundleUrl,
  fetchOrderDetail,
  fetchGuestOrderDetail,
  type OrderDetail,
} from "@/lib/api-orders";
import type { OrderPhotoDetail } from "@/types/order";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";

// /orders/return — landing page PayMongo redirects to after hosted checkout.
// Polls the order until status flips to PAID, then renders an editorial
// receipt: photo cards on the left, receipt details + actions on the right
// (mobile stacks photo-first). Guests get a "create account" upsell so
// they keep their photos beyond the email link's 7-day window.

const POLL_INTERVAL_MS = 2000;
const POLL_MAX_ATTEMPTS = 30; // 60s

type PollState = "polling" | "paid" | "failed" | "timeout";

export default function OrdersReturnPage() {
  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col">
      <SiteHeader />
      {/* `ReturnBody` reads `?orderId=` + `?token=` via useSearchParams(). Without
          this boundary Next bails out of static prerender for the whole route and
          `next build` fails — the same reason /reset-password wraps its form. The
          fallback is the polling state at attempt 0 because that is what the page
          renders the instant the params resolve on the happy path. */}
      <Suspense fallback={<PollingState attempt={0} />}>
        <ReturnBody />
      </Suspense>
    </main>
  );
}

function ReturnBody() {
  const searchParams = useSearchParams();
  const orderId = searchParams?.get("orderId") ?? null;
  const token = searchParams?.get("token") ?? null;
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const clearCart = useCartStore((s) => s.clear);
  const queryClient = useQueryClient();

  const [pollState, setPollState] = useState<PollState>(
    orderId ? "polling" : "failed",
  );
  const [detail, setDetail] = useState<OrderDetail | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!orderId) return;
    let cancelled = false;
    let attempts = 0;

    const tick = async () => {
      if (cancelled) return;
      attempts += 1;
      setAttempt(attempts);
      try {
        const d = await loadDetail(orderId, token, isAuthenticated);
        if (cancelled) return;
        if (d && (d.status === "PAID" || d.status === "FULFILLED")) {
          clearCart();
          queryClient.invalidateQueries({ queryKey: ["me", "orders"] });
          queryClient.invalidateQueries({ queryKey: ["me", "saved-events"] });
          setDetail(d);
          setPollState("paid");
          return;
        }
        if (d?.status === "REFUNDED") {
          setPollState("failed");
          return;
        }
      } catch (err) {
        // 404 means the orderId/token combo is genuinely wrong (fat-fingered
        // return URL, expired guest token) — fail fast instead of spinning
        // for 60s and landing on the friendly-but-misleading TimeoutState.
        // Other errors (network blip, BE hiccup, token not visible yet)
        // stay transient and keep polling.
        if (err instanceof ApiError && err.status === 404) {
          if (!cancelled) setPollState("failed");
          return;
        }
      }
      if (attempts >= POLL_MAX_ATTEMPTS) {
        if (!cancelled) setPollState("timeout");
        return;
      }
      setTimeout(tick, POLL_INTERVAL_MS);
    };
    tick();
    return () => {
      cancelled = true;
    };
  }, [orderId, token, isAuthenticated, clearCart, queryClient]);

  if (!orderId) return <MissingOrder />;
  if (pollState === "paid" && detail) {
    return <PaidState detail={detail} hasAccount={isAuthenticated} />;
  }
  if (pollState === "failed") return <FailedState />;
  if (pollState === "timeout")
    return <TimeoutState orderId={orderId} token={token} />;
  return <PollingState attempt={attempt} />;
}

async function loadDetail(
  orderId: string,
  token: string | null,
  isAuthenticated: boolean,
): Promise<OrderDetail | null> {
  if (token) return fetchGuestOrderDetail(orderId, token);
  if (isAuthenticated) return fetchOrderDetail(orderId);
  return null;
}

/* ─────────────── PAID — editorial receipt ─────────────── */

function PaidState({
  detail,
  hasAccount,
}: {
  detail: OrderDetail;
  hasAccount: boolean;
}) {
  const reference = detail.id.slice(0, 8).toUpperCase();
  const paidAt = formatPaidAt(detail.paidAt);
  const photoCount = detail.photos?.length ?? 0;
  const photoWord = photoCount === 1 ? "photo" : "photos";
  const eventName = detail.eventName ?? "QuickPitik";
  const recipientEmail = detail.recipientEmail ?? "";

  // Download-all hits the BE bundle endpoint which streams a ZIP. Token-only
  // auth on the BE — works for runners AND guests since <a download> can't
  // carry a JWT. Hidden when shareToken is missing (legacy orders before
  // V18 backfill, or a BE that omits the field).
  const bundleUrl = detail.shareToken
    ? buildOrderBundleUrl(detail.id, detail.shareToken)
    : null;
  const showDownloadAll = photoCount >= 2 && bundleUrl != null;
  const [bulkState, setBulkState] = useState<"idle" | "preparing">("idle");

  const handleDownloadAll = () => {
    if (bulkState === "preparing" || !bundleUrl) return;
    setBulkState("preparing");
    // Programmatic <a download> click — same-tab, no navigation. Server's
    // Content-Disposition makes the browser save the response as a file.
    const a = document.createElement("a");
    a.href = bundleUrl;
    a.rel = "noopener";
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    // Reset state after the browser has had time to pick up the download.
    // The BE may take a few seconds to assemble the first byte (S3 reads +
    // ZIP entry headers); 5s leaves the spinner visible long enough to
    // signal "we heard you" without holding the button hostage if the
    // request silently fails.
    window.setTimeout(() => setBulkState("idle"), 5000);
  };

  // When the master Download-all owns the fresh slot, per-photo CTAs demote
  // to ghost so the one-fresh-per-viewport rule holds. Single-photo orders
  // keep per-photo as the primary fresh action — the master button hides.
  const perPhotoVariant: PhotoCardVariant = showDownloadAll ? "ghost" : "primary";

  return (
    <section className="flex-1 px-6 md:px-10 py-10 md:py-16">
      <div className="max-w-6xl mx-auto">
        {/* Event meta */}
        <div
          className="mb-8 md:mb-12"
          style={{ animation: "fade-up 0.55s 0.05s both", opacity: 0 }}
        >
          <Kicker as="p" className="mb-1.5">
            {eventName}
          </Kicker>
          <Kicker as="p" tone="soft" tnum>
            {paidAt}
          </Kicker>
        </div>

        {/* 2-column editorial layout */}
        <div className="grid md:grid-cols-2 gap-10 md:gap-14 lg:gap-20 items-start">
          {/* LEFT — photo card stack */}
          <div
            className="space-y-6 md:space-y-7 order-1"
            style={{ animation: "fade-up 0.6s 0.15s both", opacity: 0 }}
          >
            {showDownloadAll && (
              <DownloadAllButton
                photoCount={photoCount}
                state={bulkState}
                onClick={handleDownloadAll}
              />
            )}
            {detail.photos.length > 0 ? (
              detail.photos.map((photo, i) => (
                <PhotoCard
                  key={photo.id}
                  photo={photo}
                  index={i}
                  variant={perPhotoVariant}
                />
              ))
            ) : (
              <PhotoCardSkeleton />
            )}
          </div>

          {/* RIGHT — receipt panel */}
          <div
            className="order-2 md:pt-2"
            style={{ animation: "fade-up 0.6s 0.25s both", opacity: 0 }}
          >
            <Kicker as="p" tone="active" className="mb-3">
              ── Payment received
            </Kicker>
            <h1 className="font-hero text-ink text-5xl md:text-6xl lg:text-7xl mb-5">
              All yours.
            </h1>
            <p className="font-sans text-base md:text-lg text-ink-soft leading-relaxed max-w-md mb-9">
              <span className="tnum text-ink">{photoCount}</span> {photoWord}{" "}
              from <span className="text-ink">{eventName}</span> for{" "}
              <span className="text-ink tnum">
                ₱{detail.total.toLocaleString()}
              </span>
              .
            </p>

            {/* Receipt meta block */}
            <div className="border-t border-line py-5 space-y-3">
              <ReceiptRow label="Ref" value={reference} mono tnum />
              <ReceiptRow label="Paid" value={paidAt} mono />
              {recipientEmail && (
                <ReceiptRow label="To" value={recipientEmail} />
              )}
            </div>

            <div className="border-t border-line pt-5 pb-9">
              <p className="font-sans text-sm text-ink-soft leading-relaxed">
                {recipientEmail ? (
                  <>
                    We&rsquo;ve also emailed these links to{" "}
                    <span className="text-ink">{recipientEmail}</span>. They
                    work for <span className="tnum">7</span> days.
                  </>
                ) : (
                  <>The download links above work for 7 days.</>
                )}
              </p>
            </div>

            {/* CTAs */}
            <div className="flex flex-col gap-4">
              {hasAccount && (
                <Link
                  href={ROUTES.ORDERS}
                  className="inline-flex w-full sm:w-auto items-center justify-center bg-ink hover:bg-ink-soft text-bone px-7 py-3.5 rounded-full font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
                >
                  View your orders →
                </Link>
              )}

              <Link
                href={ROUTES.EVENTS}
                className={cn(
                  "inline-flex w-full sm:w-auto items-center justify-center px-7 py-3.5 rounded-full font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
                  hasAccount
                    ? "border border-line hover:bg-bone-deep text-ink"
                    : "bg-ink hover:bg-ink-soft text-bone",
                )}
              >
                Browse more events →
              </Link>

              {!hasAccount && (
                <UpsellCard recipientEmail={recipientEmail} />
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

type PhotoCardVariant = "primary" | "ghost";

function PhotoCard({
  photo,
  index,
  variant,
}: {
  photo: OrderPhotoDetail;
  index: number;
  variant: PhotoCardVariant;
}) {
  // Disposition + filename are baked into the signed downloadUrl by the
  // backend (appending params client-side breaks the SigV4 signature on R2).
  const downloadUrl = photo.downloadUrl ?? null;
  const label = photo.bib ? `BIB ${photo.bib}` : "Untagged";
  const previewSrc = photo.previewUrl || photo.thumbnailUrl;

  return (
    <article className="space-y-4">
      <div className="relative aspect-[4/5] w-full rounded-2xl overflow-hidden bg-bone-deep border border-line shadow-[0_24px_60px_-30px_rgba(17,17,17,0.25)]">
        {previewSrc ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewSrc}
            alt={photo.bib ? `Race photo ${photo.bib}` : "Untagged race photo"}
            className="absolute inset-0 w-full h-full object-cover"
            draggable={false}
          />
        ) : (
          <span className="absolute inset-0 flex items-center justify-center font-mono uppercase tracking-[0.14em] text-[10px] text-bone/40 -rotate-[18deg]">
            QuickPitik
          </span>
        )}

        {/* Sequence badge — only show for multi-photo orders */}
        <div className="absolute top-3 left-3 bg-bone/85 backdrop-blur-sm rounded-full px-2.5 py-1">
          <Kicker tone="soft" tnum>
            {label}
          </Kicker>
        </div>
      </div>

      {downloadUrl ? (
        <a
          href={downloadUrl}
          download
          className={cn(
            "group inline-flex w-full items-center justify-center gap-2 px-6 py-3.5 rounded-full font-display font-bold text-[15px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
            variant === "primary"
              ? "bg-fresh hover:bg-fresh-deep text-surface"
              : "border border-ink text-ink hover:bg-ink hover:text-surface",
          )}
        >
          <span
            aria-hidden="true"
            className="transition-transform group-hover:translate-y-0.5"
          >
            ↓
          </span>
          <span>Download photo {index > 0 ? `${index + 1}` : ""}</span>
        </a>
      ) : (
        <button
          disabled
          className="inline-flex w-full items-center justify-center gap-2 bg-bone-deep text-slate-soft px-6 py-3.5 rounded-full font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] cursor-not-allowed"
        >
          Preparing · {label}
        </button>
      )}
    </article>
  );
}

function DownloadAllButton({
  photoCount,
  state,
  onClick,
}: {
  photoCount: number;
  state: "idle" | "preparing";
  onClick: () => void;
}) {
  const isPreparing = state === "preparing";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={isPreparing}
      aria-live="polite"
      className={cn(
        "group inline-flex w-full items-center justify-center gap-2.5 bg-fresh hover:bg-fresh-deep text-surface px-7 py-4 rounded-full font-display font-bold text-[15px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
        isPreparing && "opacity-90 cursor-wait hover:bg-fresh",
      )}
    >
      {isPreparing ? (
        <>
          <span
            aria-hidden="true"
            className="size-3.5 rounded-full border-2 border-bone/40 border-t-bone animate-spin"
          />
          <span>Preparing your zip…</span>
        </>
      ) : (
        <>
          <span
            aria-hidden="true"
            className="transition-transform group-hover:translate-y-0.5"
          >
            ↓
          </span>
          <span>
            Download all <span className="tnum">{photoCount}</span> photos (.zip)
          </span>
        </>
      )}
    </button>
  );
}

function PhotoCardSkeleton() {
  return (
    <article className="space-y-4">
      <div className="aspect-[4/5] w-full rounded-2xl bg-bone-deep border border-line animate-pulse" />
      <div className="h-12 w-full rounded-full bg-bone-deep animate-pulse" />
    </article>
  );
}

function ReceiptRow({
  label,
  value,
  mono = false,
  tnum = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
  tnum?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-6">
      <Kicker tone="soft" className="shrink-0">
        {label}
      </Kicker>
      <span
        className={cn(
          "text-right break-all text-sm text-ink",
          mono ? "font-mono uppercase tracking-[0.18em]" : "font-sans",
          tnum && "tnum",
        )}
      >
        {value}
      </span>
    </div>
  );
}

function UpsellCard({ recipientEmail }: { recipientEmail: string }) {
  const prefill = recipientEmail
    ? `?prefill=${encodeURIComponent(recipientEmail)}`
    : "";
  return (
    <div className="rounded-2xl border border-line bg-bone-deep px-5 py-5">
      <Kicker as="p" className="mb-2">
        Want them forever?
      </Kicker>
      <p className="font-sans text-sm md:text-base text-ink leading-relaxed mb-4">
        Create an account to keep this purchase in your library and skip
        checkout next time.
      </p>
      <Link
        href={`${ROUTES.REGISTER}${prefill}`}
        className="inline-flex items-center font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-fresh hover:text-fresh-deep transition-colors"
      >
        Create account →
      </Link>
    </div>
  );
}

/* ─────────────── POLLING / TIMEOUT / FAILED / MISSING ─────────────── */

function PollingState({ attempt }: { attempt: number }) {
  return (
    <div className="flex-1 flex items-center justify-center px-6 md:px-10 py-16">
      <div className="max-w-md w-full text-center">
        <span
          className="size-12 rounded-full border-2 border-line border-t-fresh animate-spin mx-auto mb-8"
          aria-hidden="true"
        />
        <Kicker as="p" className="mb-3">
          Confirming · attempt {attempt}
        </Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-tight mb-4">
          Sealing your photos.
        </h1>
        <p className="font-sans text-base text-ink-soft">
          We&rsquo;re waiting on PayMongo to confirm the charge. Usually a few
          seconds &mdash; keep this tab open.
        </p>
      </div>
    </div>
  );
}

function TimeoutState({
  orderId,
  token,
}: {
  orderId: string;
  token: string | null;
}) {
  return (
    <div className="flex-1 flex items-center justify-center px-6 md:px-10 py-16">
      <div className="max-w-md w-full text-center">
        <Kicker as="p" className="mb-3">
          Still processing
        </Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-tight mb-4">
          Your bank is taking a moment.
        </h1>
        <p className="font-sans text-base text-ink-soft mb-8">
          PayMongo confirmed your payment, but the receipt hasn&rsquo;t reached
          us yet. Check your email in a minute &mdash; we&rsquo;ll send the
          download links the moment it lands.
        </p>
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="inline-flex items-center border border-line hover:bg-bone-deep text-ink px-7 py-3.5 rounded-full font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] transition-colors"
        >
          Check again ↻
        </button>
        <p className="mt-6 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
          Reference ·{" "}
          <span className="text-ink tnum">{orderId.slice(0, 8).toUpperCase()}</span>
          {token ? " · Guest link" : ""}
        </p>
      </div>
    </div>
  );
}

function FailedState() {
  return (
    <div className="flex-1 flex items-center justify-center px-6 md:px-10 py-16">
      <div className="max-w-md w-full text-center">
        <Kicker as="p" className="mb-3">
          Something went sideways
        </Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-tight mb-4">
          We couldn&rsquo;t confirm this order.
        </h1>
        <p className="font-sans text-base text-ink-soft mb-8">
          If you were charged, the receipt will still arrive by email.
          Otherwise nothing was taken &mdash; try again any time.
        </p>
        <Link
          href={ROUTES.EVENTS}
          className="inline-flex items-center bg-ink hover:bg-ink-soft text-bone px-7 py-3.5 rounded-full font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] transition-colors"
        >
          Browse events →
        </Link>
      </div>
    </div>
  );
}

function MissingOrder() {
  return (
    <div className="flex-1 flex items-center justify-center px-6 md:px-10 py-16">
      <div className="max-w-md w-full text-center">
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-tight mb-4">
          Nothing to see here.
        </h1>
        <p className="font-sans text-base text-ink-soft mb-8">
          This page lands you after a payment &mdash; but we don&rsquo;t have
          an order reference in the URL.
        </p>
        <Link
          href={ROUTES.EVENTS}
          className="inline-flex items-center bg-ink hover:bg-ink-soft text-bone px-7 py-3.5 rounded-full font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] transition-colors"
        >
          Browse events →
        </Link>
      </div>
    </div>
  );
}

/* ─────────────── HELPERS ─────────────── */

function formatPaidAt(iso: string | undefined | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  const datePart = d
    .toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      timeZone: "Asia/Manila",
    })
    .toUpperCase();
  const timePart = d
    .toLocaleString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
      timeZone: "Asia/Manila",
    })
    .toUpperCase();
  return `${datePart} · ${timePart} PHT`;
}
