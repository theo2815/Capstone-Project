import { useAdminPayouts, EMPTY_PAYOUTS } from "@/hooks/use-admin-payouts";
import {
  useAdminFlags,
  useAdminKpis,
} from "@/hooks/use-admin-data";
import { usePendingVerificationsCount } from "@/hooks/use-admin-users";
import { ROUTES } from "@/lib/constants";

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
// Dispute/event counts come from backend KPIs; verification counts subscribe
// to the canonical user directory.

export interface AdminAttentionTarget {
  href: string;
  count: number;
  label: "Disputes" | "Flags" | "Verifications" | "Event requests" | "Payouts";
}

export function useAdminAttentionTarget(): AdminAttentionTarget | null {
  const pendingVerifications = usePendingVerificationsCount();
  const serverPayouts = useAdminPayouts().data ?? EMPTY_PAYOUTS;
  // BE-hydrated; useAdminFlags is a no-op when the feature gate is off.
  const openFlags = useAdminFlags({ status: "open" })?.total ?? 0;
  const kpis = useAdminKpis();
  const pendingEventRequests = kpis?.pendingEventRequests ?? 0;

  const openDisputes = kpis?.openDisputes ?? 0;
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

  // Payouts — count the canonical backend rows.
  const pendingPayouts = serverPayouts.filter((p) => p.status === "pending_review").length;
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
  const serverPayouts = useAdminPayouts().data ?? EMPTY_PAYOUTS;
  const flags = useAdminFlags({ status: "open" })?.total ?? 0;
  const kpis = useAdminKpis();
  const events = kpis?.pendingEventRequests ?? 0;

  const disputes = kpis?.openDisputes ?? 0;
  const payouts = serverPayouts.filter((p) => p.status === "pending_review").length;
  const inbox = usePendingVerificationsCount();

  return {
    inbox,
    disputes,
    flags,
    events,
    payouts,
    total: inbox + disputes + flags + events + payouts,
  };
}
