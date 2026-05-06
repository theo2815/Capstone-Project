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
import { useOrdersStore, type MockOrder } from "@/store/orders-store";
import { useToast } from "@/hooks/use-toast";
import {
  PhotoPreviewCard,
  type PhotoPreviewItem,
} from "@/components/photos/photo-preview-card";
import { getEventById } from "@/lib/event-catalog";
import { ROUTES } from "@/lib/constants";
import {
  formatMemberSince,
  formatMonthYear,
  formatPaidAt,
} from "@/lib/format";
import { cn } from "@/lib/utils";

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
  if (!user) return null;

  const memberSince = formatMemberSince(user.createdAt);

  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col scroll-smooth">
      <SiteHeader />
      <div className="flex-1 max-w-7xl mx-auto w-full px-6 md:px-10">
        <div className="md:grid md:grid-cols-[15rem_1fr] md:gap-12 lg:gap-20">
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
          <div className="stagger-children min-w-0 pb-8 md:pb-20">
            <SpendSlab />
            <ReceiptsSlab />
          </div>
        </div>
      </div>
    </main>
  );
}

function SpendSlab() {
  const orders = useOrdersStore((s) => s.orders);
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
        <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft mt-8">
          Since {stats.firstPurchase}
        </p>
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
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate mt-3">
        {label}
      </p>
    </div>
  );
}

function ReceiptsSlab() {
  const orders = useOrdersStore((s) => s.orders);
  const sorted = useMemo(
    () => [...orders].sort((a, b) => b.paidAt.localeCompare(a.paidAt)),
    [orders],
  );
  const trailing = `${sorted.length} receipt${sorted.length === 1 ? "" : "s"}`;

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
        <ul className="border-y border-line divide-y divide-line">
          {sorted.map((order) => (
            <li key={order.id}>
              <ReceiptRow order={order} />
            </li>
          ))}
        </ul>
      )}
    </Slab>
  );
}

function ReceiptRow({ order }: { order: MockOrder }) {
  const event = getEventById(order.eventId);
  const [expanded, setExpanded] = useState(false);
  const [previewIndex, setPreviewIndex] = useState<number | null>(null);
  const { showToast } = useToast();

  // Build PhotoPreviewItem[] for the lightbox. Owned mode hides watermark, in-cart
  // pill, and price-bearing CTAs, so bib/time/price are placeholders here.
  // TODO(backend): hydrate `imageUrl` (and real bib/time) from `/me/orders/{id}`
  // once Spring Boot Phase E lands.
  const previewItems = useMemo<ReadonlyArray<PhotoPreviewItem>>(
    () =>
      order.photoIds.map((id, i) => ({
        id,
        bib: null,
        time: "—",
        tone: i,
        price: 0,
      })),
    [order.photoIds],
  );

  function handleDownloadAll() {
    // TODO(backend): swap for a presigned-bundle fetch once Spring Boot Phase E
    // exposes `/me/orders/${order.id}/download-bundle`.
    showToast({
      kind: "success",
      message: `Preparing ${order.photoIds.length} photo${order.photoIds.length === 1 ? "" : "s"}…`,
    });
  }

  function handleDownloadOne(id: string) {
    // TODO(backend): swap for a presigned single-photo fetch on
    // `/me/orders/${order.id}/photos/${id}/download`.
    showToast({
      kind: "success",
      message: `Downloading ${id.replace(/^mock-/, "")}…`,
    });
  }

  const photoCountLabel = order.photoIds.length === 1 ? "photo" : "photos";

  return (
    <div className="py-6 md:py-7">
      <div className="flex flex-col md:flex-row md:items-baseline md:justify-between gap-3 md:gap-6">
        <div className="flex-1 min-w-0">
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate tnum">
            {formatPaidAt(order.paidAt)}
          </p>
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
            <span className="font-mono tnum">{order.photoIds.length}</span>{" "}
            {photoCountLabel}
            <span className="text-slate-soft"> · </span>
            {labelForPaymentMethod(order.paymentMethod)}
            <span className="text-slate-soft"> · </span>
            <span className="font-mono">{order.id}</span>
          </p>
        </div>
        <div className="flex items-baseline justify-between md:flex-col md:items-end gap-3 md:gap-2 shrink-0">
          <p className="font-mono tnum font-medium text-ink text-xl md:text-2xl">
            ₱{order.total.toLocaleString()}
          </p>
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
            aria-controls={`receipt-${order.id}-photos`}
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
          id={`receipt-${order.id}-photos`}
          className="mt-6 pt-6 border-t border-line/60 animate-fade-in"
        >
          <PhotoStrip
            photoIds={order.photoIds}
            onSelect={(i) => setPreviewIndex(i)}
          />
          <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-3">
            <button
              type="button"
              onClick={handleDownloadAll}
              className="font-sans text-base font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-3 px-6 rounded-full transition-colors inline-flex items-center gap-2"
            >
              Download all
              <span aria-hidden="true">↓</span>
            </button>
            <a
              href={`mailto:support@quickpitik.com?subject=Receipt ${order.id}`}
              className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink hover:text-ink transition-colors"
            >
              Need help with this order?
            </a>
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
  onSelect,
}: {
  photoIds: ReadonlyArray<string>;
  onSelect: (index: number) => void;
}) {
  // TODO(backend): replace abstract tiles with real thumbnail <img> tags once
  // Spring Boot returns presigned S3 URLs on `/me/orders/{id}` with each photo's
  // small variant. Until then, show ID chips so the slot has presence.
  const max = 5;
  const visible = photoIds.slice(0, max);
  const overflow = Math.max(0, photoIds.length - max);

  return (
    <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
      {visible.map((id, i) => (
        <button
          key={id}
          type="button"
          onClick={() => onSelect(i)}
          aria-label={`Preview ${id.replace(/^mock-/, "")}`}
          className="aspect-[4/3] bg-bone-deep border border-line rounded-md flex items-center justify-center overflow-hidden hover:border-ink/40 hover:bg-bone transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
        >
          <span className="font-mono text-[10px] tracking-[0.2em] text-slate uppercase tnum px-2 truncate">
            {id.replace(/^mock-/, "")}
          </span>
        </button>
      ))}
      {overflow > 0 && (
        <button
          type="button"
          onClick={() => onSelect(max)}
          aria-label={`View ${overflow} more photo${overflow === 1 ? "" : "s"}`}
          className="aspect-[4/3] bg-bone-deep border border-line rounded-md flex items-center justify-center overflow-hidden hover:border-ink/40 hover:bg-bone transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
        >
          <span className="font-mono text-[10px] tracking-[0.2em] text-ink uppercase tnum">
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
    total += o.total;
    photoCount += o.photoIds.length;
    if (earliest === null || o.paidAt < earliest) earliest = o.paidAt;
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
