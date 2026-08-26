"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import { ProtectedRoute } from "@/components/auth/protected-route";
import { SiteHeader } from "@/components/layout/site-header";
import {
  IdentityRail,
  Slab,
  type JumpSection,
} from "@/components/profile-shell";
import { useAuth } from "@/hooks/use-auth";
import { useOrdersList, useOrderDetail } from "@/hooks/use-orders";
import {
  buildOrderBundleUrl,
  withdrawDispute,
  type RunnerDispute,
} from "@/lib/api-orders";
import {
  appendDownloadDisposition,
  buildPhotoDownloadFilename,
} from "@/lib/download-helpers";
import { type MockOrder } from "@/store/orders-store";
import { useToast } from "@/hooks/use-toast";
import { useConfirmation } from "@/hooks/use-confirmation";
import {
  PhotoPreviewCard,
  type PhotoPreviewItem,
} from "@/components/photos/photo-preview-card";
import { Kicker } from "@/components/ui/kicker";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { RefundModal } from "@/components/orders/refund-modal";
import { RefundTimeline } from "@/components/orders/refund-timeline";
import {
  getOrderRefundStatus,
  type OrderRefundStatus,
} from "@/lib/refund-helpers";
import { getEventById } from "@/lib/event-catalog";
import { ROUTES } from "@/lib/constants";
import { PAGE_SIZE } from "@/lib/pagination-config";
import {
  formatMemberSince,
  formatMonthYear,
  formatPaidAt,
} from "@/lib/format";
import { ApiError } from "@/lib/api";
import { cn, formatPrice } from "@/lib/utils";
import { BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";

// Programmatic anchor click — direct hit to the presigned S3 URL avoids the
// CORS-on-fetch trap. Same idiom for per-photo + bundle.
function triggerDownload(url: string, filename?: string) {
  if (typeof document === "undefined") return;
  const a = document.createElement("a");
  a.href = url;
  if (filename) a.download = filename;
  a.rel = "noopener";
  a.click();
}

const JUMP_SECTIONS: ReadonlyArray<JumpSection> = [
  { id: "spend", label: "Spend" },
  { id: "receipts", label: "Receipts" },
];

export default function OrdersPage() {
  return (
    <ProtectedRoute>
      <OrdersBody />
    </ProtectedRoute>
  );
}

function OrdersBody() {
  const { user } = useAuth();
  const [refundOrder, setRefundOrder] = useState<MockOrder | null>(null);
  const searchParams = useSearchParams();
  const router = useRouter();

  // ?expand={orderId} deep-link from the runner notification inbox. We
  // mirror the param into local state so the URL can be cleaned
  // immediately (avoids re-triggering on refresh) but the ReceiptRow
  // children still see a stable value to react to. Updates when the URL
  // param changes (mid-session click from the inbox while already on
  // /orders) — the matching row then expands + scrolls.
  const expandFromUrl = searchParams.get("expand");
  const [pendingExpand, setPendingExpand] = useState<string | null>(
    expandFromUrl,
  );

  useEffect(() => {
    if (expandFromUrl && expandFromUrl !== pendingExpand) {
      setPendingExpand(expandFromUrl);
    }
    if (expandFromUrl) {
      // Clear the param so a refresh doesn't re-trigger. `scroll: false`
      // keeps the user's current scroll position while the matching row
      // handles its own scrollIntoView.
      router.replace(ROUTES.ORDERS, { scroll: false });
    }
    // pendingExpand intentionally omitted — including it would loop the
    // effect every time we update local state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expandFromUrl, router]);

  if (!user) return null;

  const memberSince = formatMemberSince(user.createdAt);
  const refundEvent = refundOrder
    ? getEventById(refundOrder.eventId)
    : undefined;
  // Live event name lives on the order itself (BE-hydrated). Catalog
  // fallback only matters for locally-pushed CheckoutModal orders that
  // haven't been refetched yet.
  const refundEventName =
    refundOrder?.eventName ?? refundEvent?.name ?? "—";

  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col scroll-smooth">
      <SiteHeader />
      <div className="flex-1 max-w-7xl mx-auto w-full px-6 md:px-10 flex flex-col">
        <div className="md:grid md:grid-cols-[15rem_1fr] md:gap-12 lg:gap-20 flex-1">
          <IdentityRail
            user={user}
            kicker={
              <>
                Profile · Receipts
                <span className="text-slate-soft"> · </span>
                <span className="tnum">Since {memberSince}</span>
              </>
            }
            headline="Orders."
            subline={
              <span className="block max-w-xs">
                Every receipt and every photo you&apos;ve kept, in one place.
              </span>
            }
            jumpSections={JUMP_SECTIONS}
            currentPath={ROUTES.ORDERS}
          />
          <div className="stagger-children min-w-0 pb-8 md:pb-20 md:border-l md:border-line md:-ml-6 lg:-ml-10 md:pl-6 lg:pl-10">
            <SpendSlab />
            <ReceiptsSlab
              onRefundRequest={setRefundOrder}
              pendingExpand={pendingExpand}
            />
          </div>
        </div>
      </div>

      {refundOrder && (
        <RefundModal
          mode="request"
          isOpen
          onClose={() => setRefundOrder(null)}
          order={refundOrder}
          eventName={refundEventName}
        />
      )}
    </main>
  );
}

function SpendSlab() {
  const { orders } = useOrdersList();
  const stats = useMemo(() => computeSpendStats(orders), [orders]);

  if (orders.length === 0) {
    return (
      <Slab id="spend" number="01" title="Spend">
        <p className="font-sans text-base text-slate max-w-md">
          Your purchase totals will land here once you keep your first photo.
        </p>
      </Slab>
    );
  }

  return (
    <Slab
      id="spend"
      number="01"
      title="Spend"
      caption="Lifetime totals"
    >
      <div className="grid grid-cols-3 gap-4 md:gap-8">
        <Stat value={`₱${stats.total.toLocaleString()}`} label="spent" accent />
        <Stat
          value={String(stats.orderCount)}
          label={stats.orderCount === 1 ? "order" : "orders"}
        />
        <Stat
          value={String(stats.photoCount)}
          label={stats.photoCount === 1 ? "photo kept" : "photos kept"}
        />
      </div>
      {stats.firstPurchase && (
        <Kicker as="p" tone="soft" className="mt-8">
          Since {stats.firstPurchase}
        </Kicker>
      )}
    </Slab>
  );
}

function Stat({
  value,
  label,
  accent,
}: {
  value: string;
  label: string;
  accent?: boolean;
}) {
  return (
    <div className="border-l border-line pl-4 md:pl-6 first:border-0 first:pl-0">
      <p
        className={cn(
          "font-display font-extrabold tracking-tight tnum text-3xl md:text-5xl leading-none",
          accent ? "text-fresh" : "text-ink",
        )}
      >
        {value}
      </p>
      <Kicker as="p" className="mt-3">
        {label}
      </Kicker>
    </div>
  );
}

function ReceiptsSlab({
  onRefundRequest,
  pendingExpand,
}: {
  onRefundRequest: (order: MockOrder) => void;
  pendingExpand: string | null;
}) {
  const { orders } = useOrdersList();
  const sorted = useMemo(
    () =>
      [...orders].sort((a, b) =>
        (b.paidAt ?? "").localeCompare(a.paidAt ?? ""),
      ),
    [orders],
  );
  const trailing = `${sorted.length} receipt${sorted.length === 1 ? "" : "s"}`;
  const [loadedCount, setLoadedCount] = useState(PAGE_SIZE.RECEIPT_INITIAL);
  // If the deep-link target sits past the initial page, bump the visible
  // window so the row mounts and the expand effect fires. Without this a
  // notification for the 11th receipt would silently no-op.
  useEffect(() => {
    if (!pendingExpand) return;
    const idx = sorted.findIndex((o) => o.id === pendingExpand);
    if (idx >= 0 && idx >= loadedCount) {
      setLoadedCount(idx + 1);
    }
  }, [pendingExpand, sorted, loadedCount]);
  const visibleSlice = sorted.slice(0, loadedCount);

  return (
    <Slab
      id="receipts"
      number="02"
      title="Receipts"
      trailing={sorted.length > 0 ? trailing : undefined}
    >
      {sorted.length === 0 ? (
        <ReceiptsEmpty />
      ) : (
        <>
          <ul className="border-y border-line divide-y divide-line">
            {visibleSlice.map((order) => (
              <li key={order.id}>
                <ReceiptRow
                  order={order}
                  onRefundRequest={onRefundRequest}
                  pendingExpand={pendingExpand}
                />
              </li>
            ))}
          </ul>
          <LoadMoreButton
            shown={visibleSlice.length}
            total={sorted.length}
            increment={PAGE_SIZE.RECEIPT_INCREMENT}
            onLoadMore={() =>
              setLoadedCount((n) => n + PAGE_SIZE.RECEIPT_INCREMENT)
            }
          />
        </>
      )}
    </Slab>
  );
}

function ReceiptRow({
  order,
  onRefundRequest,
  pendingExpand,
}: {
  order: MockOrder;
  onRefundRequest: (order: MockOrder) => void;
  pendingExpand: string | null;
}) {
  // Prefer the BE-hydrated event fields on the order itself. The local
  // EVENT_CATALOG seed is empty, so getEventById() returns undefined for any
  // real backend order and we'd render "Event archived" for live data.
  const catalogEvent = getEventById(order.eventId);
  const eventName = order.eventName ?? catalogEvent?.name ?? null;
  const eventSlug = order.eventSlug ?? catalogEvent?.slug ?? null;
  const [expanded, setExpanded] = useState(pendingExpand === order.id);
  const [previewIndex, setPreviewIndex] = useState<number | null>(null);
  const [withdrawingId, setWithdrawingId] = useState<string | null>(null);
  const { showToast } = useToast();
  const { confirm } = useConfirmation();
  const queryClient = useQueryClient();
  const rowRef = useRef<HTMLDivElement>(null);
  const { detail } = useOrderDetail(expanded ? order.id : null);

  // Notification deep-link: when the runner clicks a refund notification,
  // the inbox routes to /orders?expand={orderId}. OrdersBody mirrors the
  // param into pendingExpand and clears the URL; this effect handles the
  // expansion + scroll regardless of whether the row was already expanded.
  useEffect(() => {
    if (pendingExpand && pendingExpand === order.id) {
      setExpanded(true);
      // Defer until layout settles so scrollIntoView lands on the right spot.
      const t = setTimeout(() => {
        rowRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 80);
      return () => clearTimeout(t);
    }
  }, [pendingExpand, order.id]);

  // Disputes live on the order payload itself (BE-embedded). Detail payload
  // is the more current copy when the row is expanded — prefer it so a
  // server-side resolution arriving while the user has the row open shows up.
  const disputes = (detail?.disputes ?? order.disputes ?? []) as RunnerDispute[];
  const orderForStatus = useMemo(
    () => ({ ...order, disputes }),
    [order, disputes],
  );
  const refundStatus = useMemo<OrderRefundStatus>(
    () => getOrderRefundStatus(orderForStatus),
    [orderForStatus],
  );
  const canRequest =
    refundStatus.kind === "none" || refundStatus.kind === "rejected";
  // The runner can only cancel a request before admin engages. ESCALATED is
  // admin's territory; RESOLVED / DENIED / WITHDRAWN are terminal.
  const cancellableDispute = disputes.find((d) => d.status === "open") ?? null;

  // Defensive coalesce — backend payloads can ship partial fields and the
  // renderer must not crash on a single bad row. Mock data is always complete;
  // these defaults only matter post-backend.
  const photoIds = order.photoIds ?? [];
  const total = order.total ?? 0;
  const paymentMethod = order.paymentMethod ?? "";
  const orderLabel = order.id ?? "—";
  const photoCount = photoIds.length;

  // Build PhotoPreviewItem[] for the lightbox. Owned mode hides watermark, in-cart
  // pill, and price-bearing CTAs. Live-mode `detail.photos` carries `previewUrl`
  // (server-baked watermark) — feed it into PhotoPreviewItem.imageUrl so the
  // lightbox renders real images instead of the placeholder geometry. Deps
  // key off `order.photoIds` directly to keep the array reference stable.
  const previewItems = useMemo<ReadonlyArray<PhotoPreviewItem>>(() => {
    if (detail?.photos && detail.photos.length > 0) {
      return detail.photos.map((p, i) => ({
        id: p.id,
        bib: p.bib,
        time: p.time,
        tone: p.tone ?? i,
        price: 0,
        imageUrl: p.previewUrl ?? null,
      }));
    }
    return (order.photoIds ?? []).map((id, i) => ({
      id,
      bib: null,
      time: "—",
      tone: i,
      price: 0,
    }));
  }, [detail?.photos, order.photoIds]);

  // BE no longer pre-stamps a presigned bundle URL — it hands us a per-order
  // shareToken and we point a top-level <a> at the streaming bundle endpoint.
  // Same path runs the email button + the guest /orders/return flow.
  const bundleUrl =
    detail?.shareToken && detail?.id
      ? buildOrderBundleUrl(detail.id, detail.shareToken)
      : null;

  function handleDownloadAll() {
    if (!bundleUrl) return;
    // Server sets the real Content-Disposition (single-photo orders come
    // back as image/jpeg, multi as application/zip). The hint here only
    // matters if the response somehow omits the header — keep it honest
    // to the count so a fallback save doesn't misname the file.
    const hintExt = photoCount === 1 ? "jpg" : "zip";
    triggerDownload(bundleUrl, `${order.id}.${hintExt}`);
    showToast({
      kind: "success",
      message: `Preparing ${photoCount} photo${photoCount === 1 ? "" : "s"}…`,
    });
  }

  async function handleCancelRequest(disputeId: string) {
    const ok = await confirm({
      title: "Cancel refund request?",
      message:
        "You can submit a new request for these photos later if you change your mind.",
      confirmLabel: "Cancel request",
      cancelLabel: "Keep request",
      danger: true,
    });
    if (!ok) return;
    setWithdrawingId(disputeId);
    try {
      await withdrawDispute(disputeId);
      await queryClient.invalidateQueries({ queryKey: ["me", "orders"] });
      showToast({
        kind: "success",
        message: "Refund request cancelled.",
        duration: 4000,
      });
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "We couldn't cancel that request. Try again in a moment.";
      showToast({ kind: "error", message });
    } finally {
      setWithdrawingId(null);
    }
  }

  function handleDownloadOne(id: string) {
    const photo = detail?.photos.find((p) => p.id === id);
    if (!photo?.downloadUrl) return;
    // appendDownloadDisposition flips the response to
    // Content-Disposition: attachment so mobile Safari (which ignores
    // cross-origin `<a download>`) and desktop both save instead of
    // navigating to the image. Same plumbing as the /orders/return cards.
    const filename = buildPhotoDownloadFilename(photo);
    const url = appendDownloadDisposition(photo.downloadUrl, filename);
    triggerDownload(url, filename);
    showToast({
      kind: "success",
      message: `Downloading ${id.replace(/^mock-/, "")}…`,
    });
  }

  const photoCountLabel = photoCount === 1 ? "photo" : "photos";

  return (
    <div ref={rowRef} className="py-6 md:py-7 scroll-mt-24">
      <div className="flex flex-col md:flex-row md:items-baseline md:justify-between gap-3 md:gap-6">
        <div className="flex-1 min-w-0">
          <Kicker as="p" tnum>
            {formatPaidAt(order.paidAt)}
          </Kicker>
          {eventName && eventSlug ? (
            <Link
              href={`/events/${eventSlug}`}
              className="font-display text-xl md:text-2xl font-bold tracking-tight text-ink hover:text-fresh underline decoration-line-strong decoration-2 underline-offset-[6px] hover:decoration-fresh transition-colors mt-2 inline-block max-w-full truncate"
            >
              {eventName}
            </Link>
          ) : eventName ? (
            <p className="font-display text-xl md:text-2xl font-bold tracking-tight text-ink mt-2">
              {eventName}
            </p>
          ) : (
            <p className="font-display text-xl md:text-2xl font-bold tracking-tight text-slate mt-2">
              Event archived
            </p>
          )}
          <p className="font-sans text-sm text-slate mt-2">
            <span className="font-mono tnum">{photoCount}</span>{" "}
            {photoCountLabel}
            <span className="text-slate-soft"> · </span>
            {labelForPaymentMethod(paymentMethod)}
            <span className="text-slate-soft"> · </span>
            <span className="font-mono">{orderLabel}</span>
          </p>
          <RefundStatusChip
            status={refundStatus}
            photoCount={photoCount}
          />
        </div>
        <div className="flex items-baseline justify-between md:flex-col md:items-end gap-3 md:gap-2 shrink-0">
          <p className="font-mono tnum font-medium text-ink text-xl md:text-2xl">
            ₱{total.toLocaleString()}
          </p>
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
            aria-controls={`receipt-${orderLabel}-photos`}
            className="font-sans text-sm font-medium text-ink hover:text-fresh transition-colors inline-flex items-center gap-1.5 group"
          >
            <span className="underline decoration-line-strong underline-offset-4 decoration-1 group-hover:decoration-fresh">
              {expanded ? "Hide photos" : "View photos"}
            </span>
            <span
              aria-hidden="true"
              className={cn(
                "transition-transform",
                expanded ? "rotate-90" : "",
              )}
            >
              →
            </span>
          </button>
        </div>
      </div>

      {expanded && (
        <div
          id={`receipt-${orderLabel}-photos`}
          className="mt-6 pt-6 border-t border-line/60 animate-fade-in"
        >
          {photoCount === 0 ? (
            <p className="font-sans text-sm text-slate">
              Photo details aren&apos;t available for this order.
            </p>
          ) : (
            <PhotoStrip
              photoIds={photoIds}
              thumbnails={detail?.photos.map((p) => p.thumbnailUrl ?? null)}
              onSelect={(i) => setPreviewIndex(i)}
            />
          )}
          <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-3">
            {photoCount > 0 && bundleUrl && (
              <button
                type="button"
                onClick={handleDownloadAll}
                className="font-display text-base font-bold border border-ink text-ink hover:bg-ink hover:text-surface py-3 px-6 rounded-full transition-colors inline-flex items-center gap-2"
              >
                {photoCount === 1 ? "Download photo" : "Download all"}
                <span aria-hidden="true">↓</span>
              </button>
            )}
            {canRequest ? (
              <button
                type="button"
                onClick={() => onRefundRequest(order)}
                className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-error hover:text-error transition-colors"
              >
                Request a refund
              </button>
            ) : (
              <span className="font-sans text-sm text-slate-soft">
                {refundStatus.kind === "approved"
                  ? "Refund issued · cannot resubmit"
                  : "Refund pending review"}
              </span>
            )}
            {cancellableDispute && (
              <button
                type="button"
                onClick={() => handleCancelRequest(cancellableDispute.id)}
                disabled={withdrawingId === cancellableDispute.id}
                className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink hover:text-ink transition-colors disabled:opacity-50 disabled:hover:text-slate disabled:hover:decoration-line"
              >
                {withdrawingId === cancellableDispute.id
                  ? "Cancelling…"
                  : "Cancel refund request"}
              </button>
            )}
          </div>
          {disputes.length > 0 && (
            <RefundTimeline disputes={disputes} className="mt-6" />
          )}
        </div>
      )}

      {previewIndex !== null && previewItems[previewIndex] && (
        <PhotoPreviewCard
          mode="owned"
          photo={previewItems[previewIndex]}
          eventName={eventName ?? "Order"}
          index={previewIndex + 1}
          total={previewItems.length}
          onClose={() => setPreviewIndex(null)}
          onPrev={
            previewIndex > 0
              ? () => setPreviewIndex(previewIndex - 1)
              : undefined
          }
          onNext={
            previewIndex < previewItems.length - 1
              ? () => setPreviewIndex(previewIndex + 1)
              : undefined
          }
          onDownload={() =>
            handleDownloadOne(previewItems[previewIndex].id)
          }
        />
      )}
    </div>
  );
}

function PhotoStrip({
  photoIds,
  thumbnails,
  onSelect,
}: {
  photoIds: ReadonlyArray<string>;
  // Aligned with photoIds — `thumbnails[i]` is the URL for `photoIds[i]`,
  // `null` when missing (mock-mode falls back to ID-chip rendering).
  thumbnails?: ReadonlyArray<string | null>;
  onSelect: (index: number) => void;
}) {
  const max = 5;
  const visible = photoIds.slice(0, max);
  const overflow = Math.max(0, photoIds.length - max);

  return (
    <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
      {visible.map((id, i) => {
        const thumb = thumbnails?.[i] ?? null;
        return (
          <button
            key={id}
            type="button"
            onClick={() => onSelect(i)}
            aria-label={`Preview ${id.replace(/^mock-/, "")}`}
            className="aspect-[4/3] bg-bone-deep border border-line rounded-md flex items-center justify-center overflow-hidden hover:border-ink hover:shadow-[var(--shadow-card)] transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
          >
            {thumb ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={thumb}
                alt={id.replace(/^mock-/, "")}
                className="size-full object-cover"
                loading="lazy"
              />
            ) : (
              <Kicker tnum className="px-2 truncate">
                {id.replace(/^mock-/, "")}
              </Kicker>
            )}
          </button>
        );
      })}
      {overflow > 0 && (
        <button
          type="button"
          onClick={() => onSelect(max)}
          aria-label={`View ${overflow} more photo${overflow === 1 ? "" : "s"}`}
          className="aspect-[4/3] bg-bone-deep border border-line rounded-md flex items-center justify-center overflow-hidden hover:border-ink hover:shadow-[var(--shadow-card)] transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
        >
          <span className="font-mono text-[14px] min-[400px]:text-[15px] md:text-[13px] tracking-[0.18em] text-ink uppercase tnum">
            +{overflow}
          </span>
        </button>
      )}
    </div>
  );
}

function ReceiptsEmpty() {
  return (
    <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
      <p className="font-display text-2xl md:text-3xl font-extrabold tracking-tight text-ink">
        No purchases yet.
      </p>
      <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
        Find your photos and pick the ones worth keeping.
      </p>
      <Link href={ROUTES.EVENTS} className={cn(BTN_PRIMARY, BTN_SIZE.md, "mt-6")}>
        Browse races
        <span aria-hidden="true">→</span>
      </Link>
    </div>
  );
}

function computeSpendStats(orders: ReadonlyArray<MockOrder>): {
  total: number;
  orderCount: number;
  photoCount: number;
  firstPurchase: string | null;
} {
  if (orders.length === 0) {
    return { total: 0, orderCount: 0, photoCount: 0, firstPurchase: null };
  }
  let total = 0;
  let photoCount = 0;
  let earliest: string | null = null;
  for (const o of orders) {
    total += o.total ?? 0;
    photoCount += (o.photoIds ?? []).length;
    if (o.paidAt && (earliest === null || o.paidAt < earliest)) {
      earliest = o.paidAt;
    }
  }
  return {
    total,
    orderCount: orders.length,
    photoCount,
    firstPurchase: earliest ? formatMonthYear(earliest) : null,
  };
}

function labelForPaymentMethod(method: string): string {
  const map: Record<string, string> = {
    gcash: "GCash",
    card: "Card",
    paymaya: "PayMaya",
    grabpay: "GrabPay",
  };
  return map[method.toLowerCase()] ?? method;
}

function RefundStatusChip({
  status,
  photoCount,
}: {
  status: OrderRefundStatus;
  photoCount: number;
}) {
  if (status.kind === "none") return null;

  const label = (() => {
    switch (status.kind) {
      case "pending":
        return `Refund pending · ${status.pendingCount} of ${photoCount} photo${photoCount === 1 ? "" : "s"}`;
      case "partial":
        return `Refund in review · ${status.totalDisputed} of ${photoCount} photo${photoCount === 1 ? "" : "s"}`;
      case "approved":
        return `Refund approved · ${formatPrice(status.refundAmount)}`;
      case "rejected":
        return `Refund declined`;
      default:
        return null;
    }
  })();

  if (!label) return null;

  return (
    <div className="mt-3 space-y-1.5">
      <Kicker as="p" tnum className="inline-flex items-center gap-2">
        <span
          aria-hidden="true"
          className={cn(
            "size-1.5 rounded-full",
            status.kind === "approved"
              ? "bg-fresh"
              : status.kind === "rejected"
                ? "bg-error"
                : "bg-slate",
          )}
        />
        {label}
      </Kicker>
      {status.kind === "rejected" && status.rejectedNote && (
        <p className="font-sans text-sm text-ink-soft max-w-md">
          <span className="text-slate">Admin note · </span>
          {status.rejectedNote}
        </p>
      )}
    </div>
  );
}
