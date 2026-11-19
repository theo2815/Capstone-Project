import { useAdminKpis } from "@/hooks/use-admin-data";
import type { AdminKpis } from "@/lib/api-admin";
import { ADMIN_FLAGS_ENABLED, ROUTES } from "@/lib/constants";

// The one list of admin queues. Its order is the priority order: the rail
// counts, the phone strip, the KPI strip and the single breathing attention
// dot (one-fresh-per-viewport) all read it. Every count comes from
// GET /admin/kpis; a field an older backend does not send gives null (no badge).
// Adding a queue means one entry here plus its Inbox content.

export type AdminQueueKey =
  | "disputes"
  | "flags"
  | "verifications"
  | "sellerDetails"
  | "eventRequests"
  | "payouts"
  | "payoutReports"
  | "relevance"
  | "removals";

export interface AdminQueue {
  key: AdminQueueKey;
  label: string;
  singularLabel: string;
  inboxType: string;
  href: string;
  count: (kpis: AdminKpis) => number | null;
}

const ADMIN_QUEUES_ALL: readonly AdminQueue[] = [
  { key: "disputes", label: "Disputes", singularLabel: "Dispute", inboxType: "disputes", href: ROUTES.ADMIN_DISPUTES,
    count: (k) => k.disputesNeedingAdmin ?? k.openDisputes },
  { key: "flags", label: "Flags", singularLabel: "Flag", inboxType: "flags", href: ROUTES.ADMIN_FLAGS,
    count: (k) => k.openFlags },
  { key: "verifications", label: "Verifications", singularLabel: "Verification", inboxType: "verifications",
    href: ROUTES.ADMIN_VERIFICATIONS, count: (k) => k.pendingVerifications },
  { key: "sellerDetails", label: "Seller details", singularLabel: "Seller details", inboxType: "seller-details",
    href: `${ROUTES.ADMIN_VERIFICATIONS}#seller-details`, count: (k) => k.pendingSellerDetails ?? null },
  { key: "eventRequests", label: "Event requests", singularLabel: "Event request", inboxType: "events",
    href: ROUTES.ADMIN_EVENT_REQUESTS, count: (k) => k.pendingEventRequests },
  // The page's Review tab (PENDING), not pendingPayouts (PENDING + SCHEDULED).
  { key: "payouts", label: "Payouts", singularLabel: "Payout", inboxType: "payouts", href: ROUTES.ADMIN_PAYOUTS,
    count: (k) => k.payoutsAwaitingReview ?? null },
  { key: "payoutReports", label: "Payout reports", singularLabel: "Payout report", inboxType: "payout-reports",
    href: `${ROUTES.ADMIN_PAYOUTS}#payout-reports`, count: (k) => k.openPayoutReports ?? null },
  // Cases a person must decide; the Hidden / Can't tell / Appeals totals are for the page's tab bar.
  { key: "relevance", label: "Photo Relevance", singularLabel: "Photo Relevance", inboxType: "relevance",
    href: ROUTES.ADMIN_RELEVANCE, count: (k) => k.relevanceNeedsPerson ?? null },
  { key: "removals", label: "Photo removals", singularLabel: "Photo removal", inboxType: "removals",
    href: ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS, count: (k) => k.removalsNeedingAction ?? null },
];

// Same gate as the rail's Flags row.
export const ADMIN_QUEUES: readonly AdminQueue[] = ADMIN_FLAGS_ENABLED
  ? ADMIN_QUEUES_ALL
  : ADMIN_QUEUES_ALL.filter((q) => q.key !== "flags");

/** The page a queue href lives on, without its in-page anchor. */
export function queuePath(href: string): string {
  return href.split("#")[0];
}

/** The sum of the present counts, or null when none is present. */
export function sumPresent(counts: readonly (number | null)[]): number | null {
  const present = counts.filter((c): c is number => c !== null);
  return present.length ? present.reduce((a, b) => a + b, 0) : null;
}

export interface AdminQueueCounts {
  byKey: Record<AdminQueueKey, number | null>;
  /** The sum of the present counts; null while none is present (loading, or an old backend). */
  total: number | null;
}

export function useAdminQueueCounts(): AdminQueueCounts {
  const kpis = useAdminKpis();
  const byKey = Object.fromEntries(
    ADMIN_QUEUES_ALL.map((q) => [q.key, kpis && ADMIN_QUEUES.includes(q) ? q.count(kpis) : null]),
  ) as Record<AdminQueueKey, number | null>;
  return { byKey, total: sumPresent(Object.values(byKey)) };
}

export interface AdminAttentionTarget {
  key: AdminQueueKey;
  href: string;
  count: number;
  label: string;
}

/** The first queue in the order with work waiting, or null. */
export function useAdminAttentionTarget(): AdminAttentionTarget | null {
  const { byKey } = useAdminQueueCounts();
  for (const q of ADMIN_QUEUES) {
    const count = byKey[q.key];
    if (count !== null && count > 0) return { key: q.key, href: q.href, count, label: q.label };
  }
  return null;
}
