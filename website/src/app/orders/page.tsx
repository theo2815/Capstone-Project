"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { ProtectedRoute } from "@/components/auth/protected-route";
import { SiteHeader } from "@/components/layout/site-header";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { useAuth } from "@/hooks/use-auth";
import { useOrdersStore, type MockOrder } from "@/store/orders-store";
import { useToast } from "@/hooks/use-toast";
import {
  PhotoPreviewCard,
  type PhotoPreviewItem,
} from "@/components/photos/photo-preview-card";
import { getEventById } from "@/lib/event-catalog";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { Role, User } from "@/types/user";

const JUMP_SECTIONS: ReadonlyArray<{ id: string; label: string }> = [
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

  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col scroll-smooth">
      <SiteHeader />
      <div className="flex-1 max-w-5xl mx-auto w-full px-6 md:px-10">
        <div className="md:grid md:grid-cols-[15rem_1fr] md:gap-12 lg:gap-20">
          <IdentityRail user={user} />
          <div className="stagger-children min-w-0 pb-8 md:pb-20">
            <SpendSlab />
            <ReceiptsSlab />
          </div>
        </div>
      </div>
      <Footer />
    </main>
  );
}

function IdentityRail({ user }: { user: User }) {
  const router = useRouter();
  const { logout } = useAuth();
  const activeId = useActiveSection(JUMP_SECTIONS.map((s) => s.id));
  const memberSince = formatMemberSince(user.createdAt);
  const moreLinks = moreLinksForRole(user.role, ROUTES.ORDERS);

  function handleSignOut() {
    logout();
    router.replace(ROUTES.HOME);
  }

  return (
    <aside className="md:sticky md:top-24 md:self-start pt-10 md:pt-20 pb-8 md:pb-12 border-b md:border-0 border-line">
      <div className="flex items-start gap-5 md:block">
        <AvatarDisc name={user.name} size="md" />
        <div className="flex-1 min-w-0 md:mt-7">
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
            Profile · Receipts
            <span className="text-slate-soft"> · </span>
            <span className="tnum">Since {memberSince}</span>
          </p>
          <h1 className="font-display text-3xl md:text-4xl font-medium tracking-tight leading-[1.05] mt-3">
            Orders.
          </h1>
          <p className="font-sans text-sm text-slate mt-3 max-w-xs">
            Every receipt and every photo you&apos;ve kept, in one place.
          </p>
        </div>
      </div>

      <nav aria-label="Jump to section" className="hidden md:block mt-12">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
          Jump to
        </p>
        <ul className="mt-4 space-y-3">
          {JUMP_SECTIONS.map((section) => {
            const isActive = activeId === section.id;
            return (
              <li key={section.id}>
                <a
                  href={`#${section.id}`}
                  aria-current={isActive ? "true" : undefined}
                  className={cn(
                    "group flex items-center gap-3 font-display text-base transition-colors",
                    isActive ? "text-ink" : "text-slate hover:text-ink",
                  )}
                >
                  <span
                    aria-hidden="true"
                    className={cn(
                      "size-1.5 rounded-full transition-colors",
                      isActive ? "bg-fresh" : "bg-line group-hover:bg-slate",
                    )}
                  />
                  <span>{section.label}</span>
                </a>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="mt-8 md:mt-12">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
          More
        </p>
        <ul className="mt-4 space-y-3">
          {moreLinks.map((link) => (
            <li key={link.href}>
              <Link
                href={link.href}
                className="font-display text-base text-slate hover:text-ink transition-colors"
              >
                {link.label}
              </Link>
            </li>
          ))}
          <li>
            <button
              type="button"
              onClick={handleSignOut}
              className="font-display text-base text-slate hover:text-ink transition-colors"
            >
              Sign out
            </button>
          </li>
        </ul>
      </div>
    </aside>
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

function formatPaidAt(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
  const day = d.getDate().toString().padStart(2, "0");
  const year = d.getFullYear();
  return `${month} ${day} · ${year}`;
}

function formatMonthYear(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
  const year = d.getFullYear();
  return `${month} ${year}`;
}

function formatMemberSince(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
  const year = d.getFullYear();
  return `${month} ${year}`;
}

function Slab({
  id,
  number,
  title,
  caption,
  trailing,
  children,
}: {
  id?: string;
  number: string;
  title: string;
  caption?: string;
  trailing?: string;
  children: React.ReactNode;
}) {
  return (
    <section
      id={id}
      className="border-t border-line py-12 md:py-16 scroll-mt-24 first:border-0 first:pt-10 md:first:pt-20"
    >
      <div className="flex items-baseline justify-between gap-6 mb-8 md:mb-10">
        <div className="flex items-baseline gap-4 min-w-0">
          <span className="font-mono text-[10px] tracking-[0.15em] text-slate-soft tnum">
            {number}
          </span>
          <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-ink shrink-0">
            {title}
          </p>
          {caption && (
            <p className="hidden md:block font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft truncate">
              {caption}
            </p>
          )}
        </div>
        {trailing && (
          <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft tnum shrink-0">
            {trailing}
          </p>
        )}
      </div>
      {children}
    </section>
  );
}

function Footer() {
  return (
    <footer className="px-6 md:px-10 py-8 mt-12 flex flex-col md:flex-row items-center justify-between gap-4 border-t border-line bg-bone">
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        QuickPitik &middot; Cebu, Philippines
      </p>
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        © 2026
      </p>
    </footer>
  );
}

function useActiveSection(ids: ReadonlyArray<string>): string | null {
  const [active, setActive] = useState<string | null>(ids[0] ?? null);
  useEffect(() => {
    const elements = ids
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => el !== null);
    if (elements.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible.length > 0) {
          setActive(visible[0].target.id);
        }
      },
      {
        rootMargin: "-25% 0px -55% 0px",
        threshold: [0, 0.1, 0.5, 1],
      },
    );

    elements.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [ids]);
  return active;
}

function moreLinksForRole(
  role: Role,
  current: string,
): ReadonlyArray<{ label: string; href: string }> {
  const all: Array<{ label: string; href: string }> = [
    { label: "Profile", href: ROUTES.PROFILE },
    { label: "Account", href: ROUTES.ACCOUNT },
  ];
  if (role === "RUNNER") all.push({ label: "Orders", href: ROUTES.ORDERS });
  if (role === "PHOTOGRAPHER")
    all.push({ label: "Dashboard", href: ROUTES.DASHBOARD });
  if (role === "ADMIN") all.push({ label: "Admin", href: ROUTES.ADMIN });
  return all.filter((l) => l.href !== current);
}
