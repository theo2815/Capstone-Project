"use client";

import { useMemo } from "react";
import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { ROUTES, ADMIN_FLAGS_ENABLED } from "@/lib/constants";
import { useAdminUsersData } from "@/lib/admin-users-data";
import {
  useAdminDisputeStore,
  getEffectiveDisputes,
} from "@/store/admin-dispute-store";
import {
  useAdminFlagStore,
  getEffectiveFlags,
} from "@/store/admin-flag-store";
import {
  useAdminPayoutStore,
  getEffectivePayouts,
} from "@/store/admin-payout-store";
import { cn } from "@/lib/utils";
import { useAdminKpis } from "@/hooks/use-admin-data";

// Compact at-a-glance metrics row. Sits above the inbox queue (Phase 1)
// and any other admin route that wants a one-line read of every queue.
// Each metric is a Link into /admin/inbox with the matching ?type= chip
// preselected — so clicks from the inbox swap the chip in-place, and
// clicks from elsewhere navigate to the inbox already filtered.
//
// The highest priority non-zero queue carries a single breathing fresh
// dot, honoring the one-fresh-per-viewport rule across the page (Inbox
// header has no fresh accent of its own — the priority dot owns it).
//
// Priority order matches getAdminAttentionTarget: Disputes > Flags >
// Verifications > Payouts. First non-zero wins.

interface KpiEntry {
  readonly count: number;
  readonly label: string;
  readonly singularLabel: string;
  readonly href: string;
  readonly priorityRank: number; // lower number = higher priority
}

export function AdminKpiStrip() {
  const { rows: userRows } = useAdminUsersData();
  const disputeOverrides = useAdminDisputeStore((s) => s.overrides);
  const disputeSubmissions = useAdminDisputeStore((s) => s.submissions);
  const flagOverrides = useAdminFlagStore((s) => s.overrides);
  const payoutOverrides = useAdminPayoutStore((s) => s.overrides);
  // Live-mode preference: when GET /admin/kpis returns, override the
  // 4 counts the strip surfaces. Falls back to derivation from the merged
  // admin users hook while the kpis endpoint is still loading.
  const liveKpis = useAdminKpis();

  const entries = useMemo<readonly KpiEntry[]>(() => {
    let pendingVerifications = 0;
    for (const eff of userRows) {
      if (
        eff.role === "PHOTOGRAPHER" &&
        eff.verificationStatus === "pending" &&
        eff.suspendedAt === null
      ) {
        pendingVerifications += 1;
      }
    }

    let openDisputes = getEffectiveDisputes(
      disputeOverrides,
      disputeSubmissions,
    ).filter((d) => d.status === "open").length;
    let openFlags = getEffectiveFlags(flagOverrides).filter(
      (f) => f.status === "open",
    ).length;
    let pendingPayouts = getEffectivePayouts(payoutOverrides).filter(
      (p) => p.status === "pending_review",
    ).length;

    if (liveKpis) {
      pendingVerifications = liveKpis.pendingVerifications;
      openDisputes = liveKpis.openDisputes;
      openFlags = liveKpis.openFlags;
      pendingPayouts = liveKpis.pendingPayouts;
    }

    const all: KpiEntry[] = [
      {
        count: pendingVerifications,
        label: "Verifications",
        singularLabel: "Verification",
        href: ROUTES.ADMIN_INBOX,
        priorityRank: 3,
      },
      {
        count: openDisputes,
        label: "Disputes",
        singularLabel: "Dispute",
        href: `${ROUTES.ADMIN_INBOX}?type=disputes`,
        priorityRank: 1,
      },
      {
        count: openFlags,
        label: "Flags",
        singularLabel: "Flag",
        href: `${ROUTES.ADMIN_INBOX}?type=flags`,
        priorityRank: 2,
      },
      {
        count: pendingPayouts,
        label: "Payouts",
        singularLabel: "Payout",
        href: `${ROUTES.ADMIN_INBOX}?type=payouts`,
        priorityRank: 4,
      },
    ];
    return ADMIN_FLAGS_ENABLED
      ? all
      : all.filter((e) => e.label !== "Flags");
  }, [
    userRows,
    disputeOverrides,
    disputeSubmissions,
    flagOverrides,
    payoutOverrides,
    liveKpis,
  ]);

  // Pick the priority queue (lowest rank with count > 0). The dot lights
  // up only that one row, preserving the one-fresh-per-viewport rule.
  const priorityHref = useMemo(() => {
    let bestRank = Infinity;
    let bestHref: string | null = null;
    for (const e of entries) {
      if (e.count > 0 && e.priorityRank < bestRank) {
        bestRank = e.priorityRank;
        bestHref = e.href;
      }
    }
    return bestHref;
  }, [entries]);

  return (
    <nav aria-label="Queue counts" className="border-y border-line">
      <ul className="flex flex-wrap items-baseline gap-x-8 md:gap-x-12 gap-y-4 py-4 md:py-5">
        {entries.map((entry) => {
          const isPriority = entry.href === priorityHref;
          const labelText =
            entry.count === 1 ? entry.singularLabel : entry.label;
          return (
            <li key={entry.href}>
              <Link
                href={entry.href}
                aria-label={`${entry.count} ${labelText.toLowerCase()}`}
                className="group inline-flex items-baseline gap-2.5"
              >
                <span
                  className={cn(
                    "font-display text-2xl tnum leading-none transition-colors",
                    entry.count === 0 ? "text-slate-soft" : "text-ink",
                  )}
                >
                  {entry.count}
                </span>
                <Kicker
                  tone={entry.count === 0 ? "soft" : "default"}
                  className="group-hover:text-ink transition-colors"
                >
                  {labelText}
                </Kicker>
                {isPriority && (
                  <span
                    aria-hidden="true"
                    className="size-1.5 rounded-full bg-fresh breathe self-center"
                  />
                )}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
