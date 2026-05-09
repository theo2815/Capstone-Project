"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { ProtectedRoute } from "@/components/auth/protected-route";
import { SiteHeader } from "@/components/layout/site-header";
import {
  IdentityRail,
  Slab,
  type JumpSection,
} from "@/components/profile-shell";
import { useAuth } from "@/hooks/use-auth";
import { useOrdersList, useOrderDetail } from "@/hooks/use-orders";
import { type MockOrder } from "@/store/orders-store";
import { useToast } from "@/hooks/use-toast";
import {
  PhotoPreviewCard,
  type PhotoPreviewItem,
} from "@/components/photos/photo-preview-card";
import { Kicker } from "@/components/ui/kicker";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { RefundModal } from "@/components/orders/refund-modal";
import {
  useAdminDisputeStore,
  getEffectiveDisputes,
} from "@/store/admin-dispute-store";
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
import { cn, formatPrice } from "@/lib/utils";

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
  if (!user) return null;

  const memberSince = formatMemberSince(user.createdAt);
  const runnerHandle = deriveRunnerHandle(user.email);
  const refundEvent = refundOrder
    ? getEventById(refundOrder.eventId)
    : undefined;

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
            <ReceiptsSlab onRefundRequest={setRefundOrder} />
          </div>
        </div>
      </div>

      {refundOrder && (
        <RefundModal
          mode="request"
          isOpen
          onClose={() => setRefundOrder(null)}
          order={refundOrder}
          eventName={refundEvent?.name ?? "—"}
          photographerHandle=""
          runnerHandle={runnerHandle}
        />
      )}
    </main>
  );
}

// Mock-only handle derivation for runner-submitted disputes. Backend ships
// a real `handle` field on User in Phase B; until then we slugify the email
// local part so the admin queue has something readable.
function deriveRunnerHandle(email: string): string {
  const local = email.split("@")[0] ?? "";
  return local.split(".")[0].toLowerCase() || "runner";
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
          "font-display font-medium tracking-tight tnum text-3xl md:text-5xl leading-none",
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
}: {
  onRefundRequest: (order: MockOrder) => void;
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
}: {
  order: MockOrder;
  onRefundRequest: (order: MockOrder) => void;
}) {
  const event = getEventById(order.eventId);
  const [expanded, setExpanded] = useState(false);
  const [previewIndex, setPreviewIndex] = useState<number | null>(null);
  const { showToast } = useToast();
  const { detail } = useOrderDetail(expanded ? order.id : null);

  const overrides = useAdminDisputeStore((s) => s.overrides);
  const submissions = useAdminDisputeStore((s) => s.submissions);
  const refundStatus = useMemo<OrderRefundStatus>(
    () => getOrderRefundStatus(order, getEffectiveDisputes(overrides, submissions)),
    [order, overrides, submissions],
  );
  const canRequest =
    refundStatus.kind === "none" || refundStatus.kind === "rejected";

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

  function handleDownloadAll() {
    if (detail?.downloadBundleUrl) {
      triggerDownload(detail.downloadBundleUrl, `${order.id}.zip`);
    }
    showToast({
      kind: "success",
      message: `Preparing ${photoCount} photo${photoCount === 1 ? "" : "s"}…`,
    });
  }

  function handleDownloadOne(id: string) {
    const photo = detail?.photos.find((p) => p.id === id);
    if (photo?.downloadUrl) {
      triggerDownload(photo.downloadUrl);
    }
    showToast({
      kind: "success",
      message: `Downloading ${id.replace(/^mock-/, "")}…`,
    });
  }

  const photoCountLabel = photoCount === 1 ? "photo" : "photos";

  return (
    <div className="py-6 md:py-7">
      <div className="flex flex-col md:flex-row md:items-baseline md:justify-between gap-3 md:gap-6">
        <div className="flex-1 min-w-0">
          <Kicker as="p" tnum>
            {formatPaidAt(order.paidAt)}
          </Kicker>
          {event ? (
            <Link
              href={`/events/${event.slug}`}
              className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink hover:text-fresh transition-colors mt-2 inline-block max-w-full truncate"
            >
              {event.name}
            </Link>
          ) : (
            <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-slate mt-2">
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
            className="font-sans text-sm text-slate hover:text-ink transition-colors inline-flex items-center gap-1.5 group"
          >
            <span className="underline decoration-line underline-offset-4 decoration-1 group-hover:decoration-ink">
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
            {photoCount > 0 && (
              <button
                type="button"
                onClick={handleDownloadAll}
                className="font-sans text-base font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-3 px-6 rounded-full transition-colors inline-flex items-center gap-2"
              >
                Download all
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
          </div>
        </div>
      )}

      {previewIndex !== null && previewItems[previewIndex] && (
        <PhotoPreviewCard
          mode="owned"
          photo={previewItems[previewIndex]}
          eventName={event?.name ?? "Order"}
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
            className="aspect-[4/3] bg-bone-deep border border-line rounded-md flex items-center justify-center overflow-hidden hover:border-ink/40 hover:bg-bone transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
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
          className="aspect-[4/3] bg-bone-deep border border-line rounded-md flex items-center justify-center overflow-hidden hover:border-ink/40 hover:bg-bone transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
        >
          <span className="font-mono text-[13px] min-[400px]:text-[14px] md:text-[12px] tracking-[0.18em] text-ink uppercase tnum">
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
      <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
        No purchases yet.
      </p>
      <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
        Find your photos and pick the ones worth keeping.
      </p>
      <Link
        href={ROUTES.EVENTS}
        className="mt-6 inline-block font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
      >
        Browse races
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
    <Kicker
      as="p"
      tnum
      className="mt-3 inline-flex items-center gap-2"
    >
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
  );
}
