import { useAdminDisputeStore, getEffectiveDisputes } from "@/store/admin-dispute-store";
import {
  useAdminPayoutStore,
  mergePayoutsWithOverrides,
} from "@/store/admin-payout-store";
import {
  useAdminFlags,
  useAdminKpis,
  useAdminPayouts,
  EMPTY_PAYOUTS,
} from "@/hooks/use-admin-data";
import { getAdminUsersDataSnapshot } from "@/lib/admin-users-data";
import { ROUTES } from "@/lib/constants";

function countPendingPhotographers(): number {
  let count = 0;
  for (const row of getAdminUsersDataSnapshot()) {
    if (
      row.role === "PHOTOGRAPHER" &&
      row.verificationStatus === "pending" &&
      row.suspendedAt === null
    ) {
      count += 1;
    }
  }
  return count;
}

// One-fresh-per-viewport rule: at most ONE rail row carries the breathing
// fresh attention dot. Priority order:
//   1. Disputes  — money + customer-facing
//   2. Flags     — content moderation, time-sensitive
//   3. Verifications — photographer-onboarding-blocking
//   4. Event requests — a photographer's event is invisible until decided (V46)
//   5. Payouts   — request-based, lowest urgency
// First non-zero queue wins.
//
// useAdminAttentionTarget / useAdminQueueCounts are hooks because the
// payouts count now hydrates from BE via useAdminPayouts (React Query).
// The other queues still read snapshot state from Zustand — the React
// Query call is the only reactive piece that forces this to be a hook.

export interface AdminAttentionTarget {
  href: string;
  count: number;
  label: "Disputes" | "Flags" | "Verifications" | "Event requests" | "Payouts";
}

export function useAdminAttentionTarget(): AdminAttentionTarget | null {
  const serverPayouts = useAdminPayouts() ?? EMPTY_PAYOUTS;
  // BE-hydrated; useAdminFlags is a no-op when the feature gate is off.
  const openFlags = useAdminFlags({ status: "open" })?.total ?? 0;
  const pendingEventRequests = useAdminKpis()?.pendingEventRequests ?? 0;

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

  if (openFlags > 0) {
    return {
      href: ROUTES.ADMIN_FLAGS,
      count: openFlags,
      label: "Flags",
    };
  }

  // Verifications — attention dot lights up the Inbox row, since Inbox is
  // the new daily home for the verifications queue (Phase 1 admin redesign).
  // /admin/verifications still exists as a deep-link route but no longer
  // appears in the rail.
  const pendingVerifications = countPendingPhotographers();
  if (pendingVerifications > 0) {
    return {
      href: ROUTES.ADMIN_INBOX,
      count: pendingVerifications,
      label: "Verifications",
    };
  }

  if (pendingEventRequests > 0) {
    return {
      href: ROUTES.ADMIN_EVENT_REQUESTS,
      count: pendingEventRequests,
      label: "Event requests",
    };
  }

  // Payouts — BE-hydrated count; mock seed is empty in the new flow.
  const payoutOverrides = useAdminPayoutStore.getState().overrides;
  const pendingPayouts = mergePayoutsWithOverrides(
    serverPayouts,
    payoutOverrides,
  ).filter((p) => p.status === "pending_review").length;
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
  events: number;
  payouts: number;
  total: number;
}

// Per-queue counts for the editorial-data rail. Consumed by <AdminRail>
// to render the row-level count and the "QUEUES · N open / all clear"
// summary line.
export function useAdminQueueCounts(): AdminQueueCounts {
  const serverPayouts = useAdminPayouts() ?? EMPTY_PAYOUTS;
  const flags = useAdminFlags({ status: "open" })?.total ?? 0;
  const events = useAdminKpis()?.pendingEventRequests ?? 0;
  const disputeState = useAdminDisputeStore.getState();
  const payoutOverrides = useAdminPayoutStore.getState().overrides;

  const disputes = getEffectiveDisputes(
    disputeState.overrides,
    disputeState.submissions,
  ).filter((d) => d.status === "open").length;
  const payouts = mergePayoutsWithOverrides(
    serverPayouts,
    payoutOverrides,
  ).filter((p) => p.status === "pending_review").length;
  const inbox = countPendingPhotographers();

  return {
    inbox,
    disputes,
    flags,
    events,
    payouts,
    total: inbox + disputes + flags + events + payouts,
  };
}
