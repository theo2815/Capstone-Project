import { useAdminUserStore } from "@/store/admin-user-store";
import { useAdminDisputeStore, getEffectiveDisputes } from "@/store/admin-dispute-store";
import { useAdminFlagStore, getEffectiveFlags } from "@/store/admin-flag-store";
import { useAdminPayoutStore, getEffectivePayouts } from "@/store/admin-payout-store";
import { ROUTES, ADMIN_FLAGS_ENABLED } from "@/lib/constants";

// One-fresh-per-viewport rule: at most ONE rail row carries the breathing
// fresh attention dot. Priority order:
//   1. Disputes  — money + customer-facing
//   2. Flags     — content moderation, time-sensitive
//   3. Verifications — photographer-onboarding-blocking
//   4. Payouts   — cycle-based, lowest urgency
// First non-zero queue wins.

export interface AdminAttentionTarget {
  href: string;
  count: number;
  label: "Disputes" | "Flags" | "Verifications" | "Payouts";
}

export function getAdminAttentionTarget(): AdminAttentionTarget | null {
  // Disputes — read effective state via store overrides + seed + submissions
  const disputeState = useAdminDisputeStore.getState();
  const openDisputes = getEffectiveDisputes(
    disputeState.overrides,
    disputeState.submissions,
  ).filter((d) => d.status === "open").length;
  if (openDisputes > 0) {
    return {
      href: ROUTES.ADMIN_DISPUTES,
      count: openDisputes,
      label: "Disputes",
    };
  }

  // Flags — skipped entirely when the v1 feature gate is off.
  if (ADMIN_FLAGS_ENABLED) {
    const flagOverrides = useAdminFlagStore.getState().overrides;
    const openFlags = getEffectiveFlags(flagOverrides).filter(
      (f) => f.status === "open",
    ).length;
    if (openFlags > 0) {
      return {
        href: ROUTES.ADMIN_FLAGS,
        count: openFlags,
        label: "Flags",
      };
    }
  }

  // Verifications — attention dot lights up the Inbox row, since Inbox is
  // the new daily home for the verifications queue (Phase 1 admin redesign).
  // /admin/verifications still exists as a deep-link route but no longer
  // appears in the rail.
  const pendingVerifications =
    useAdminUserStore.getState().getPendingPhotographers().length;
  if (pendingVerifications > 0) {
    return {
      href: ROUTES.ADMIN_INBOX,
      count: pendingVerifications,
      label: "Verifications",
    };
  }

  // Payouts
  const payoutOverrides = useAdminPayoutStore.getState().overrides;
  const pendingPayouts = getEffectivePayouts(payoutOverrides).filter(
    (p) => p.status === "pending_review",
  ).length;
  if (pendingPayouts > 0) {
    return {
      href: ROUTES.ADMIN_PAYOUTS,
      count: pendingPayouts,
      label: "Payouts",
    };
  }

  return null;
}

export interface AdminQueueCounts {
  inbox: number;
  disputes: number;
  flags: number;
  payouts: number;
  total: number;
}

// Per-queue counts for the editorial-data rail. Consumed by <AdminRail>
// to render the row-level count and the "QUEUES · N open / all clear"
// summary line. Non-reactive (matches getAdminAttentionTarget): the rail
// re-renders on pathname change which refreshes these on navigation.
export function getAdminQueueCounts(): AdminQueueCounts {
  const disputeState = useAdminDisputeStore.getState();
  const payoutOverrides = useAdminPayoutStore.getState().overrides;

  const disputes = getEffectiveDisputes(
    disputeState.overrides,
    disputeState.submissions,
  ).filter((d) => d.status === "open").length;
  const flags = ADMIN_FLAGS_ENABLED
    ? getEffectiveFlags(useAdminFlagStore.getState().overrides).filter(
        (f) => f.status === "open",
      ).length
    : 0;
  const payouts = getEffectivePayouts(payoutOverrides).filter(
    (p) => p.status === "pending_review",
  ).length;
  const inbox = useAdminUserStore.getState().getPendingPhotographers().length;

  return {
    inbox,
    disputes,
    flags,
    payouts,
    total: inbox + disputes + flags + payouts,
  };
}
