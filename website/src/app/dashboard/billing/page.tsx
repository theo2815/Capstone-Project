"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { Slab } from "@/components/profile-shell";
import { HowPayoutsModal } from "@/components/dashboard/how-payouts-modal";
import { FilePayoutReportModal } from "@/components/dashboard/file-payout-report-modal";
import { TrackPayoutReportModal } from "@/components/dashboard/track-payout-report-modal";
import { RequestPayoutModal } from "@/components/dashboard/request-payout-modal";
import { WithdrawPayoutModal } from "@/components/dashboard/withdraw-payout-modal";
import { Kicker } from "@/components/ui/kicker";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  usePhotographerPayouts,
  usePhotographerPayoutBalance,
  usePhotographerPayoutReports,
  usePhotographerTransactions,
} from "@/hooks/use-photographer-data";
import { ROUTES } from "@/lib/constants";
import { formatLongDate, formatMonthYear } from "@/lib/format";
import { formatPayoutNumber } from "@/lib/payout-format";
import { PAGE_SIZE } from "@/lib/pagination-config";
import {
  type PayoutStatus,
  type PhotographerPayout,
  type PhotographerTransaction,
} from "@/lib/photographer-mock";
import {
  getLatestReportForCycle,
  mergeReportsWithOverrides,
  type PayoutReport,
} from "@/lib/admin-payout-reports";
import { useAdminPayoutReportStore } from "@/store/admin-payout-report-store";
import { resolveCurrentPhotographer } from "@/lib/current-photographer";
import { useAuth } from "@/hooks/use-auth";
import { cn } from "@/lib/utils";
import {
  PAYOUT_METHOD_LABEL,
  usePhotographerSettingsStore,
} from "@/store/photographer-settings-store";

const STATUS_LABEL: Record<PayoutStatus, string> = {
  paid: "PAID",
  pending: "PENDING",
  scheduled: "SCHEDULED",
  held: "HELD",
};

const STATUS_TONE: Record<PayoutStatus, string> = {
  paid: "text-slate-soft",
  pending: "text-ink",
  scheduled: "text-slate",
  held: "text-error",
};

export default function BillingPage() {
  return (
    <>
      <PayoutsSlab />
      <TransactionsSlab />
    </>
  );
}

function PayoutsSlab() {
  const [howOpen, setHowOpen] = useState(false);
  const [requestOpen, setRequestOpen] = useState(false);
  const [withdrawOpen, setWithdrawOpen] = useState(false);
  const [reportingCycle, setReportingCycle] = useState<PhotographerPayout | null>(
    null,
  );
  const [trackingCycle, setTrackingCycle] = useState<PhotographerPayout | null>(
    null,
  );
  const [loadedCount, setLoadedCount] = useState(PAGE_SIZE.PAYOUT_INITIAL);
  const livePayouts = usePhotographerPayouts();
  const balance = usePhotographerPayoutBalance();
  const payouts = livePayouts ?? [];
  const isLoading = livePayouts === null || balance === null;
  const { user } = useAuth();
  const photographer = resolveCurrentPhotographer(user);
  // Read reports from BE so admin acknowledge/resolve responses surface to
  // the photographer (the local submissions array only held reports from
  // *this* browser session). overrides still apply for instant feedback
  // after the photographer submits their own.
  const serverReports = usePhotographerPayoutReports() ?? [];
  const overrides = useAdminPayoutReportStore((s) => s.overrides);

  const reports = useMemo(
    () => mergeReportsWithOverrides(serverReports, overrides),
    [serverReports, overrides],
  );

  // Lookup of latestReport per cycle id, scoped to the current photographer.
  // null entries mean "no report yet" — row shows File a report; non-null
  // entries flip the row to Track your report.
  const reportByCycleId = useMemo(() => {
    const map = new Map<string, PayoutReport>();
    if (!photographer) return map;
    for (const p of payouts ?? []) {
      const r = getLatestReportForCycle(reports, photographer.id, p.id);
      if (r) map.set(p.id, r);
    }
    return map;
  }, [reports, photographer, payouts]);

  const trackingReport = trackingCycle
    ? (reportByCycleId.get(trackingCycle.id) ?? null)
    : null;

  if (isLoading || !payouts) {
    return (
      <Slab id="payouts" number="01" title="Payouts" caption="Weekly · GCash">
        <div>
          <Skeleton className="h-3 w-28" />
          <Skeleton className="h-12 md:h-16 w-72 md:w-96 mt-3" />
          <Skeleton className="h-4 w-56 mt-4" />
        </div>
        <div className="mt-12 md:mt-16">
          <Skeleton className="h-3 w-32 mb-5" />
          <ul className="border-y border-line divide-y divide-line">
            {[0, 1, 2].map((i) => (
              <li key={i} className="py-5 md:py-6">
                <Skeleton className="h-3 w-44" />
                <Skeleton className="h-4 w-64 mt-2" />
              </li>
            ))}
          </ul>
        </div>
      </Slab>
    );
  }

  return (
    <Slab
      id="payouts"
      number="01"
      title="Payouts"
      caption="Request when ready"
    >
      <RequestPayoutHero
        balance={balance}
        onOpenHow={() => setHowOpen(true)}
        onRequest={() => setRequestOpen(true)}
        onWithdraw={() => setWithdrawOpen(true)}
      />

      {payouts.length > 0 && (
        <div className="mt-12 md:mt-16">
          <div className="flex items-baseline justify-between gap-6 mb-5">
            <Kicker as="p" tone="soft">
              Recent payouts
            </Kicker>
            <Kicker as="p" tone="soft" tnum>
              {payouts.length} payouts
            </Kicker>
          </div>
          <ul className="border-y border-line divide-y divide-line">
            {payouts.slice(0, loadedCount).map((payout) => (
              <li key={payout.id} id={`cycle-${payout.id}`}>
                <PayoutRow
                  payout={payout}
                  hasReport={reportByCycleId.has(payout.id)}
                  onReport={() => setReportingCycle(payout)}
                  onTrack={() => setTrackingCycle(payout)}
                />
              </li>
            ))}
          </ul>
          <LoadMoreButton
            shown={Math.min(loadedCount, payouts.length)}
            total={payouts.length}
            increment={PAGE_SIZE.PAYOUT_INCREMENT}
            onLoadMore={() =>
              setLoadedCount((n) => n + PAGE_SIZE.PAYOUT_INCREMENT)
            }
          />
        </div>
      )}

      <HowPayoutsModal isOpen={howOpen} onClose={() => setHowOpen(false)} />
      <RequestPayoutModal
        isOpen={requestOpen}
        balance={balance}
        onClose={() => setRequestOpen(false)}
      />
      <WithdrawPayoutModal
        isOpen={withdrawOpen}
        openRequest={balance?.openRequest ?? null}
        onClose={() => setWithdrawOpen(false)}
      />
      <FilePayoutReportModal
        cycle={reportingCycle}
        onClose={() => setReportingCycle(null)}
      />
      <TrackPayoutReportModal
        report={trackingReport}
        cycle={trackingCycle}
        onClose={() => setTrackingCycle(null)}
        onFileFollowUp={() => {
          // Hand off to the file-report modal on the same cycle. Closing the
          // track modal first avoids two modals briefly stacking.
          const cycle = trackingCycle;
          setTrackingCycle(null);
          if (cycle) setReportingCycle(cycle);
        }}
      />
    </Slab>
  );
}

// Photographer-initiated payout hero. Three states:
//   1. Open request — shows status (pending review / approved / held) +
//      Withdraw button when held.
//   2. No open request, balance ≥ minimum — Request payout CTA active.
//   3. No open request, balance < minimum — muted "Available · ₱X" with
//      helper "Reach ₱500 to request a payout."
// A missing primary payout account shows a Set-up CTA instead of the
// Request button — the BE rejects the request anyway, so block early.
function RequestPayoutHero({
  balance,
  onOpenHow,
  onRequest,
  onWithdraw,
}: {
  balance: PayoutBalanceResponseLike | null;
  onOpenHow: () => void;
  onRequest: () => void;
  onWithdraw: () => void;
}) {
  const primary = usePhotographerSettingsStore((s) =>
    s.payouts.find((p) => p.isPrimary),
  );

  if (!balance) {
    return (
      <div>
        <Skeleton className="h-3 w-28" />
        <Skeleton className="h-12 md:h-16 w-72 md:w-96 mt-3" />
        <Skeleton className="h-4 w-56 mt-4" />
      </div>
    );
  }

  if (balance.hasOpenRequest && balance.openRequest) {
    return (
      <OpenRequestHero
        cycle={balance.openRequest}
        primary={primary ?? null}
        onWithdraw={onWithdraw}
        onOpenHow={onOpenHow}
      />
    );
  }

  const eligible = balance.unpaidBalance >= balance.minimum;

  return (
    <div>
      <Kicker as="p" tone="soft">
        Available to request
      </Kicker>
      <p
        className={cn(
          "font-display text-5xl md:text-7xl font-semibold tracking-tight tnum mt-3 leading-none",
          eligible ? "text-fresh" : "text-slate",
        )}
      >
        ₱{balance.unpaidBalance.toLocaleString()}
      </p>
      <p className="font-sans text-sm md:text-base text-ink mt-4">
        {eligible ? (
          <>Ready to request — admin reviews each request.</>
        ) : (
          <>
            Reach{" "}
            <span className="font-mono tnum">
              ₱{balance.minimum.toLocaleString()}
            </span>{" "}
            in unpaid earnings to request a payout.
          </>
        )}
      </p>

      <div className="mt-6 border-t border-line pt-5 max-w-lg">
        {primary ? (
          <Kicker as="p" className="flex items-baseline gap-2 flex-wrap">
            <span aria-hidden="true" className="text-slate-soft">
              →
            </span>
            <span className="text-ink">
              {PAYOUT_METHOD_LABEL[primary.method]}
            </span>
            <span className="text-slate-soft">·</span>
            <span className="font-mono tnum text-ink">
              {formatPayoutNumber(primary.method, primary.accountNumber)}
            </span>
            <span className="text-slate-soft">·</span>
            <span>Primary</span>
          </Kicker>
        ) : (
          <Link
            href={`${ROUTES.DASHBOARD_SETTINGS}#payout`}
            className="inline-flex items-center gap-1.5 font-sans text-sm text-fresh hover:text-fresh/80 transition-colors group"
          >
            <span className="underline decoration-fresh/40 underline-offset-4 decoration-1 group-hover:decoration-fresh">
              Set up your payout account
            </span>
            <span
              aria-hidden="true"
              className="transition-transform group-hover:translate-x-0.5"
            >
              →
            </span>
          </Link>
        )}
      </div>

      <div className="mt-6">
        <button
          type="button"
          onClick={onRequest}
          disabled={!eligible || !primary}
          className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-bone bg-fresh hover:bg-fresh-deep disabled:opacity-40 disabled:cursor-not-allowed transition-colors rounded-full px-6 py-3"
        >
          Request payout
        </button>
      </div>

      <HowItWorksLink onClick={onOpenHow} />
    </div>
  );
}

// Avoids importing the response type at the top to dodge a circular import
// risk; mirrors the actual shape returned by GET /payouts/balance.
interface PayoutBalanceResponseLike {
  unpaidBalance: number;
  minimum: number;
  hasOpenRequest: boolean;
  openRequest: PhotographerPayout | null;
}

interface PrimaryPayoutAccount {
  method: "gcash" | "maya" | "gotyme";
  accountNumber: string;
  isPrimary: boolean;
}

function OpenRequestHero({
  cycle,
  primary,
  onWithdraw,
  onOpenHow,
}: {
  cycle: PhotographerPayout;
  primary: PrimaryPayoutAccount | null;
  onWithdraw: () => void;
  onOpenHow: () => void;
}) {
  // BE returns DB wire ("pending" / "held" / "paid"). For pending, settledAt
  // is null until admin approves, then populated — that's the photographer-
  // visible "approved" stage.
  const stage: "pending_review" | "approved" | "held" =
    cycle.status === "held"
      ? "held"
      : cycle.settledAt
        ? "approved"
        : "pending_review";

  const stageLabel: Record<typeof stage, string> = {
    pending_review: "Pending review",
    approved: "Approved · payment incoming",
    held: "Held — needs attention",
  };
  const stageTone: Record<typeof stage, string> = {
    pending_review: "text-ink",
    approved: "text-fresh",
    held: "text-error",
  };

  return (
    <div>
      <Kicker as="p" tone="soft" className="flex items-center gap-2 flex-wrap">
        <span>Payout request</span>
        <span className="text-slate-soft">·</span>
        <span className={stageTone[stage]}>{stageLabel[stage]}</span>
      </Kicker>
      <p className="font-display text-5xl md:text-7xl font-semibold tracking-tight text-ink tnum mt-3 leading-none">
        ₱{cycle.amount.toLocaleString()}
      </p>
      <p className="font-sans text-sm md:text-base text-ink mt-4 tnum">
        {stage === "pending_review" && (
          <>
            Submitted{" "}
            <span className="font-mono">
              {formatLongDate(cycle.weekOf, true)}
            </span>{" "}
            · admin is reviewing
          </>
        )}
        {stage === "approved" && (
          <>
            Approved{" "}
            <span className="font-mono">
              {formatLongDate(cycle.settledAt)}
            </span>{" "}
            · admin is processing the transfer
          </>
        )}
        {stage === "held" && cycle.holdReason && (
          <>
            <span className="text-error">Reason: </span>
            <span className="text-ink">{cycle.holdReason}</span>
          </>
        )}
      </p>

      {primary && (
        <div className="mt-6 border-t border-line pt-5 max-w-lg">
          <Kicker as="p" className="flex items-baseline gap-2 flex-wrap">
            <span aria-hidden="true" className="text-slate-soft">
              →
            </span>
            <span className="text-ink">
              {PAYOUT_METHOD_LABEL[primary.method]}
            </span>
            <span className="text-slate-soft">·</span>
            <span className="font-mono tnum text-ink">
              {formatPayoutNumber(primary.method, primary.accountNumber)}
            </span>
          </Kicker>
        </div>
      )}

      {stage === "held" && (
        <div className="mt-6">
          <button
            type="button"
            onClick={onWithdraw}
            className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-error border border-error hover:bg-error hover:text-bone transition-colors rounded-full px-5 py-2"
          >
            Withdraw request
          </button>
          <p className="font-sans text-sm text-slate mt-3 max-w-md">
            Fix the issue above, withdraw this request, then submit a new one.
          </p>
        </div>
      )}

      <HowItWorksLink onClick={onOpenHow} />
    </div>
  );
}

function HowItWorksLink({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="mt-6 inline-flex items-center gap-1.5 font-sans text-sm text-slate hover:text-ink transition-colors group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone rounded-sm"
    >
      <span className="underline decoration-line underline-offset-4 decoration-1 group-hover:decoration-ink">
        How payouts work
      </span>
      <span
        aria-hidden="true"
        className="transition-transform group-hover:translate-x-0.5"
      >
        →
      </span>
    </button>
  );
}

function PayoutRow({
  payout,
  hasReport,
  onReport,
  onTrack,
}: {
  payout: PhotographerPayout;
  hasReport: boolean;
  onReport: () => void;
  onTrack: () => void;
}) {
  // Reports only make sense for cycles that have moved past the schedule
  // boundary — paid (money should have landed) and pending (funds in
  // transit, but might not arrive). Scheduled cycles are too early to
  // report on. Track-your-report stays available regardless once filed,
  // so the photographer can keep checking back even after status shifts.
  const canReport = payout.status === "paid" || payout.status === "pending";

  return (
    <div className="py-5 md:py-6 flex flex-col md:flex-row md:items-baseline md:justify-between gap-2 md:gap-6">
      <div className="flex-1 min-w-0">
        <Kicker as="p" tnum className="flex items-center gap-2 flex-wrap">
          <span>Payout of {formatLongDate(payout.weekOf, true)}</span>
          <span className="text-slate-soft">·</span>
          <span className={STATUS_TONE[payout.status]}>
            {STATUS_LABEL[payout.status]}
          </span>
        </Kicker>
        <p className="font-sans text-sm text-slate mt-2">
          {payout.status === "paid" ? (
            <>
              Settled{" "}
              <span className="font-mono tnum text-ink-soft">
                {formatLongDate(payout.settledAt)}
              </span>
              <span className="text-slate-soft"> · </span>
              <span className="font-mono">{payout.reference}</span>
            </>
          ) : payout.status === "pending" ? (
            <>Processing — funds in transit.</>
          ) : (
            <>
              Releases{" "}
              <span className="font-mono tnum text-ink-soft">
                {formatLongDate(payout.settledAt)}
              </span>
            </>
          )}
        </p>
        {hasReport ? (
          <button
            type="button"
            onClick={onTrack}
            className="mt-2 font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink hover:text-ink transition-colors"
          >
            Track your report
          </button>
        ) : canReport ? (
          <button
            type="button"
            onClick={onReport}
            className="mt-2 font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-error hover:text-error transition-colors"
          >
            File a report
          </button>
        ) : null}
      </div>
      <p
        className={cn(
          "font-mono tnum font-medium text-xl md:text-2xl shrink-0",
          payout.status === "paid" ? "text-ink-soft" : "text-ink",
        )}
      >
        ₱{payout.amount.toLocaleString()}
      </p>
    </div>
  );
}

function TransactionsSlab() {
  // monthTotals lives in the response envelope but the existing render
  // derives them client-side from the same items, so both paths agree.
  const liveTx = usePhotographerTransactions();
  const transactions = liveTx?.items ?? [];
  const isLoading = liveTx === null;
  const [loadedCount, setLoadedCount] = useState(PAGE_SIZE.TRANSACTION_INITIAL);

  // Memos must run on every render — using `?? []` so the loading branch
  // still calls them with a stable empty list. Skeleton early-return lives
  // below the hooks.
  const txList = transactions ?? [];

  // Full-month totals computed over the complete list — sticky header values
  // stay stable as load-more reveals more rows within a month.
  const fullMonthTotals = useMemo(() => {
    const map = new Map<string, number>();
    for (const tx of txList) {
      const key = monthKey(tx.paidAt);
      map.set(key, (map.get(key) ?? 0) + tx.amountKept);
    }
    return map;
  }, [txList]);

  const loadedSlice = useMemo(
    () => txList.slice(0, loadedCount),
    [txList, loadedCount],
  );

  const loadedGroups = useMemo(
    () => groupByMonth(loadedSlice, fullMonthTotals),
    [loadedSlice, fullMonthTotals],
  );

  if (isLoading || !transactions) {
    return (
      <Slab
        id="transactions"
        number="02"
        title="Transactions"
        caption="Each photo sale, post-platform-cut"
      >
        <ul className="border-y border-line divide-y divide-line">
          {[0, 1, 2, 3].map((i) => (
            <li key={i} className="py-4 md:py-5 flex items-baseline justify-between gap-4">
              <div className="flex-1 min-w-0">
                <Skeleton className="h-3 w-28" />
                <Skeleton className="h-4 w-48 mt-2" />
                <Skeleton className="h-3 w-36 mt-1" />
              </div>
              <Skeleton className="h-5 w-16 shrink-0" />
            </li>
          ))}
        </ul>
      </Slab>
    );
  }

  const total = transactions.reduce((sum, tx) => sum + tx.amountKept, 0);

  return (
    <Slab
      id="transactions"
      number="02"
      title="Transactions"
      caption="Each photo sale, post-platform-cut"
      trailing={
        transactions.length > 0
          ? `${transactions.length} · ₱${total.toLocaleString()}`
          : undefined
      }
    >
      {transactions.length === 0 ? (
        <p className="font-sans text-base text-slate max-w-md">
          Sales will land here as runners buy your photos.
        </p>
      ) : (
        <>
          <ul className="border-y border-line divide-y divide-line">
            {/* Group by month so the ledger is easier to skim. */}
            {loadedGroups.map((group) => (
              <li key={group.label} className="py-2">
                <Kicker as="p" tone="soft" tnum className="sticky top-20 bg-bone py-2">
                  {group.label} · ₱{group.total.toLocaleString()}
                </Kicker>
                <ul className="divide-y divide-line">
                  {group.items.map((tx) => (
                    <li key={tx.id}>
                      <TransactionRow tx={tx} />
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
          <LoadMoreButton
            shown={loadedSlice.length}
            total={transactions.length}
            increment={PAGE_SIZE.TRANSACTION_INCREMENT}
            onLoadMore={() =>
              setLoadedCount((n) => n + PAGE_SIZE.TRANSACTION_INCREMENT)
            }
          />
        </>
      )}
    </Slab>
  );
}

function TransactionRow({ tx }: { tx: PhotographerTransaction }) {
  // BE snapshots eventName onto the row; only genuinely hard-deleted events
  // fall through to the "Event archived" copy.
  const eventLabel = tx.eventName ?? "Event archived";

  return (
    <div className="py-4 md:py-5 flex items-baseline justify-between gap-4">
      <div className="flex-1 min-w-0">
        <Kicker as="p" tnum>
          {formatLongDate(tx.paidAt)}
        </Kicker>
        <p className="font-sans text-base text-ink mt-2 truncate">
          {eventLabel}
        </p>
        <p className="font-sans text-sm text-slate mt-1">
          <span className="font-mono">{tx.photoId.replace(/^mock-/, "")}</span>
          <span className="text-slate-soft"> · </span>
          <span>{tx.buyer}</span>
        </p>
      </div>
      <p className="font-mono tnum font-medium text-ink text-lg md:text-xl shrink-0">
        ₱{tx.amountKept.toLocaleString()}
      </p>
    </div>
  );
}

interface MonthGroup {
  label: string;
  total: number;
  items: ReadonlyArray<PhotographerTransaction>;
}

function groupByMonth(
  transactions: ReadonlyArray<PhotographerTransaction>,
  fullMonthTotals?: ReadonlyMap<string, number>,
): ReadonlyArray<MonthGroup> {
  const buckets = new Map<string, PhotographerTransaction[]>();
  for (const tx of transactions) {
    const key = monthKey(tx.paidAt);
    const list = buckets.get(key);
    if (list) list.push(tx);
    else buckets.set(key, [tx]);
  }
  // Map keys are insertion-ordered; PHOTOGRAPHER_TRANSACTIONS is newest-first
  // so the buckets iterate newest-month first. When fullMonthTotals is passed
  // (load-more case), the sticky header total stays stable across pages
  // instead of growing as more rows in the same month load.
  return Array.from(buckets.entries()).map(([key, items]) => ({
    label: formatMonthYear(`${key}-01T00:00:00Z`),
    total:
      fullMonthTotals?.get(key) ??
      items.reduce((sum, tx) => sum + tx.amountKept, 0),
    items,
  }));
}

function monthKey(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "unknown";
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  return `${yyyy}-${mm}`;
}
