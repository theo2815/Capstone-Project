"use client";

import { useMemo, useRef, useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { Slab } from "@/components/profile-shell";
import { HowPayoutsModal } from "@/components/dashboard/how-payouts-modal";
import { FilePayoutReportModal } from "@/components/dashboard/file-payout-report-modal";
import { TrackPayoutReportModal } from "@/components/dashboard/track-payout-report-modal";
import { RequestPayoutModal } from "@/components/dashboard/request-payout-modal";
import { WithdrawPayoutModal } from "@/components/dashboard/withdraw-payout-modal";
import { Kicker } from "@/components/ui/kicker";
import { BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Skeleton } from "@/components/ui/skeleton";
import { PhotographerPurchasePreview, type PurchaseSelection } from "@/components/dashboard/photographer-purchase-preview";
import {
  usePhotographerPayouts,
  usePhotographerPayoutBalance,
  usePhotographerTransactions,
} from "@/hooks/use-photographer-data";
import { ROUTES } from "@/lib/constants";
import { formatLongDate, formatSalesDate, salesMonthKey } from "@/lib/format";
import { formatPayoutNumber } from "@/lib/payout-format";
import { PAGE_SIZE } from "@/lib/pagination-config";
import {
  type PayoutStatus,
  type PhotographerPayout,
  type PhotographerTransaction,
} from "@/lib/photographer-mock";
import {
  getLatestReportForCycle,
  type PayoutReport,
} from "@/lib/admin-payout-reports";
import { usePhotographerPayoutReports, EMPTY_REPORTS } from "@/hooks/use-payout-reports";
import { resolveCurrentPhotographer } from "@/lib/current-photographer";
import { useAuth } from "@/hooks/use-auth";
import { cn, formatOrderRef, formatPrice } from "@/lib/utils";
import { PAYOUT_METHOD_LABEL } from "@/store/photographer-settings-store";

const STATUS_LABEL: Record<PayoutStatus, string> = {
  paid: "PAID",
  pending: "PENDING",
  scheduled: "SCHEDULED",
  held: "HELD",
  approved: "APPROVED",
  processing: "PROCESSING",
  failed: "FAILED",
  returned: "RETURNED",
  rejected: "REJECTED",
  cancelled: "CANCELLED",
};

const STATUS_TONE: Record<PayoutStatus, string> = {
  paid: "text-slate-soft",
  pending: "text-ink",
  scheduled: "text-slate",
  held: "text-error",
  approved: "text-ink",
  processing: "text-ink",
  failed: "text-error",
  returned: "text-error",
  rejected: "text-error",
  cancelled: "text-slate-soft",
};

// Row's secondary line, per status (paid is handled inline — it needs the
// reference). Honest about where the money actually is at each stage.
const ROW_NOTE: Record<PayoutStatus, string> = {
  paid: "",
  pending: "Awaiting admin review.",
  scheduled: "Scheduled.",
  held: "On hold — needs attention.",
  approved: "Approved — payment being sent.",
  processing: "Payment in transit.",
  failed: "Payment failed — admin will retry.",
  returned: "Payment returned — check your payout account.",
  rejected: "Request rejected.",
  cancelled: "Request cancelled.",
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
  const livePayouts = usePhotographerPayouts();
  const balance = usePhotographerPayoutBalance();
  const payouts = livePayouts.items;
  const isLoading = livePayouts.isLoading || balance === null;
  const { user } = useAuth();
  const photographer = resolveCurrentPhotographer(user);
  // Read reports from BE so admin acknowledge/resolve responses surface to
  // the photographer (the local submissions array only held reports from
  // *this* browser session). The query is scoped to the current account.
  const reports = usePhotographerPayoutReports().data ?? EMPTY_REPORTS;

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

  if (isLoading) {
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
              {livePayouts.total} payouts
            </Kicker>
          </div>
          <ul className="border-y border-line divide-y divide-line">
            {payouts.map((payout) => (
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
            shown={payouts.length}
            total={livePayouts.total}
            increment={PAGE_SIZE.PAYOUT_INCREMENT}
            onLoadMore={livePayouts.fetchNextPage}
            isLoading={livePayouts.isFetchingNextPage}
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
// `primary` comes from the BE balance response (authoritative), not the
// client settings store — the store can show a stale primary the DB lacks,
// which is what produced the demo's PAYOUT_NO_ACCOUNT on submit.
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
  if (!balance) {
    return (
      <div>
        <Skeleton className="h-3 w-28" />
        <Skeleton className="h-12 md:h-16 w-72 md:w-96 mt-3" />
        <Skeleton className="h-4 w-56 mt-4" />
      </div>
    );
  }

  const primary = balance.primaryAccount;

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
  const cooldownUntil = balance.accountCooldownUntil;
  const cooldownActive =
    !!cooldownUntil && new Date(cooldownUntil).getTime() > Date.now();

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

      {balance.clearing > 0 && (
        <div className="mt-6 max-w-lg">
          <Kicker as="p" tone="soft" className="flex items-baseline gap-2">
            <span>Clearing</span>
            <span className="font-mono tnum text-ink">
              ₱{balance.clearing.toLocaleString()}
            </span>
          </Kicker>
          <p className="font-sans text-sm text-slate mt-2">
            New sales become withdrawable after a short clearing period.
          </p>
        </div>
      )}

      {cooldownActive && (
        <div className="mt-6 rounded-xl border border-warning/40 bg-warning/5 p-4 max-w-lg">
          <Kicker as="p" className="text-warning">
            Payouts paused
          </Kicker>
          <p className="font-sans text-sm text-ink-soft mt-2">
            After a recent payout-account change, new requests are paused until{" "}
            <span className="font-mono tnum">
              {formatLongDate(cooldownUntil)}
            </span>
            . This protects your earnings if your account was compromised.
          </p>
        </div>
      )}

      <div className="mt-6 border-t border-line pt-5 max-w-lg">
        {primary?.hasQr ? (
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
              {primary ? "Upload your payout QR in Settings" : "Set up your payout account"}
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
          disabled={!eligible || !primary?.hasQr || cooldownActive}
          className={`${BTN_PRIMARY} ${BTN_SIZE.md}`}
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
  clearing: number;
  reserved: number;
  accountCooldownUntil: string | null;
  minimum: number;
  hasOpenRequest: boolean;
  openRequest: PhotographerPayout | null;
  primaryAccount: PrimaryPayoutAccount | null;
}

interface PrimaryPayoutAccount {
  method: "gcash" | "maya" | "gotyme";
  accountNumber: string;
  accountName: string;
  hasQr: boolean;
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
  // BE returns the real lifecycle status now (pending → approved → processing,
  // or held). Drive the stage off status directly — settledAt alone can't tell
  // approved from processing.
  const stage: "pending_review" | "approved" | "processing" | "held" =
    cycle.status === "held"
      ? "held"
      : cycle.status === "processing"
        ? "processing"
        : cycle.status === "approved"
          ? "approved"
          : "pending_review";

  const stageLabel: Record<typeof stage, string> = {
    pending_review: "Pending review",
    approved: "Approved · payment incoming",
    processing: "Processing · payment being sent",
    held: "Held — needs attention",
  };
  const stageTone: Record<typeof stage, string> = {
    pending_review: "text-ink",
    approved: "text-fresh",
    processing: "text-fresh",
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
              {formatLongDate(cycle.requestedAt)}
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
            · payment will be sent shortly
          </>
        )}
        {stage === "processing" && (
          <>Payment in transit · waiting on confirmation</>
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
            className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-error border border-error hover:bg-error hover:text-surface transition-colors rounded-full px-5 py-2"
          >
            Cancel request
          </button>
          <p className="font-sans text-sm text-slate mt-3 max-w-md">
            Fix the issue above, cancel this request, then submit a new one.
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
          <span>Payout of {formatLongDate(payout.requestedAt)}</span>
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
          ) : (
            ROW_NOTE[payout.status]
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
  const liveTx = usePhotographerTransactions();
  const transactions = liveTx.items;
  const isLoading = liveTx.isLoading;
  const [selection, setSelection] = useState<PurchaseSelection | null>(null);
  const returnFocus = useRef<HTMLButtonElement | null>(null);
  const selectedTransaction = transactions.find((tx) => tx.id === selection?.transactionId);
  const closePreview = () => {
    setSelection(null);
    queueMicrotask(() => returnFocus.current?.focus());
  };

  // Month-header totals come straight from the server envelope (computed over
  // ALL rows, in PHT), so sticky headers stay stable as Load-more reveals more
  // rows — no client re-derivation over the loaded page.
  const monthTotalsMap = useMemo(
    () => new Map(Object.entries(liveTx.monthTotals)),
    [liveTx.monthTotals],
  );

  const loadedGroups = useMemo(
    () => groupByMonth(transactions, monthTotalsMap),
    [transactions, monthTotalsMap],
  );

  if (isLoading) {
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

  // Grand total over ALL rows = sum of the server month totals (not just the
  // loaded page).
  const total = Object.values(liveTx.monthTotals).reduce((sum, v) => sum + v, 0);

  return (
    <Slab
      id="transactions"
      number="02"
      title="Transactions"
      caption="Each photo sale, post-platform-cut"
      trailing={
        liveTx.total > 0
          ? `${liveTx.total} · ${formatPrice(total)}`
          : undefined
      }
    >
      {liveTx.error && <div role="alert" aria-label="Transactions" className="mb-5 font-sans text-sm text-error">
        Couldn&apos;t load transactions. <button type="button" className="underline" onClick={liveTx.refetch}>Retry transactions</button>
      </div>}
      {!liveTx.error && transactions.length === 0 ? (
        <p className="font-sans text-base text-slate max-w-md">
          Sales will land here as runners buy your photos. Completed zero-charge purchases appear under each event&apos;s Purchased photos.
        </p>
      ) : transactions.length > 0 ? (
        <>
          <ul className="border-y border-line divide-y divide-line">
            {/* Group by month so the ledger is easier to skim. */}
            {loadedGroups.map((group) => (
              <li key={group.label} className="py-2">
                {/* Desktop-only sticky month label; on mobile it scrolls
                    with the content, which is what every other mobile
                    dashboard surface does. */}
                <Kicker
                  as="p"
                  tone="soft"
                  tnum
                  className="md:sticky md:top-20 bg-bone py-2"
                >
                  {group.label} · {formatPrice(group.total)}
                </Kicker>
                <ul className="divide-y divide-line">
                  {group.items.map((tx) => (
                    <li key={tx.id}>
                      <TransactionRow tx={tx} onOpen={(button) => {
                        returnFocus.current = button;
                        setSelection({ photoId: tx.photoId, orderId: tx.orderId, transactionId: tx.id });
                      }} />
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
          {liveTx.appendError && <p role="alert" aria-label="Transactions" className="mt-4 font-sans text-sm text-error">Couldn&apos;t load more transactions.</p>}
          {liveTx.appendError && <button type="button" className={`${BTN_SECONDARY} ${BTN_SIZE.sm} mt-3`} onClick={liveTx.fetchNextPage}>Retry transactions</button>}
          {liveTx.hasNextPage && !liveTx.appendError && <LoadMoreButton
            shown={transactions.length}
            total={liveTx.total}
            increment={PAGE_SIZE.TRANSACTION_INCREMENT}
            onLoadMore={liveTx.fetchNextPage}
            isLoading={liveTx.isFetchingNextPage}
          />}
        </>
      ) : null}
      <PhotographerPurchasePreview selection={selection} transaction={selectedTransaction} onClose={closePreview} />
    </Slab>
  );
}

function TransactionRow({ tx, onOpen }: { tx: PhotographerTransaction; onOpen: (button: HTMLButtonElement) => void }) {
  const [failedThumbnailUrl, setFailedThumbnailUrl] = useState<string | null>(null);
  // BE snapshots eventName onto the row; only genuinely hard-deleted events
  // fall through to the "Event archived" copy.
  const eventLabel = tx.eventName ?? "Event archived";

  const content = <span className="py-4 md:py-5 flex items-center justify-between gap-4">
      <span className="size-14 shrink-0 overflow-hidden rounded-md bg-bone-deep flex items-center justify-center text-slate-soft">
        {tx.thumbnailUrl && failedThumbnailUrl !== tx.thumbnailUrl ? <Image src={tx.thumbnailUrl} alt="" width={56} height={56} unoptimized onError={() => setFailedThumbnailUrl(tx.thumbnailUrl ?? null)} className="size-full object-cover" /> : <span role="img" aria-label="Preview unavailable" className="font-mono text-lg">▧</span>}
      </span>
      <span className="flex-1 min-w-0">
        <Kicker as="span" className="block" tnum>
          {tx.isRefund ? "Refund" : "Sale"} · {formatSalesDate(tx.recordedAt ?? tx.paidAt)}
        </Kicker>
        <span className="block font-sans text-base text-ink mt-2 truncate">
          {eventLabel}
        </span>
        <span className="block font-sans text-sm text-slate mt-1">
          <span>{tx.buyer || "Buyer unavailable"}</span>
          {tx.orderId && <><span className="text-slate-soft"> · </span><span className="font-mono tnum">{formatOrderRef(tx.orderId)}</span></>}
        </span>
      </span>
      <span className="font-mono tnum font-medium text-ink text-lg md:text-xl shrink-0">
        {tx.amountKept < 0 || tx.isRefund ? "-" : "+"}{formatPrice(Math.abs(tx.amountKept))}
      </span>
    </span>;
  return tx.orderId
    ? <button type="button" onClick={(event) => onOpen(event.currentTarget)} className="block w-full text-left hover:bg-bone-deep/50 focus-visible:bg-bone-deep/50">{content}</button>
    : <div>{content}</div>;
}

interface MonthGroup {
  label: string;
  total: number;
  items: ReadonlyArray<PhotographerTransaction>;
}

function groupByMonth(
  transactions: ReadonlyArray<PhotographerTransaction>,
  fullMonthTotals: ReadonlyMap<string, number>,
): ReadonlyArray<MonthGroup> {
  const buckets = new Map<string, PhotographerTransaction[]>();
  for (const tx of transactions) {
    const key = salesMonthKey(tx.paidAt);
    const list = buckets.get(key);
    if (list) list.push(tx);
    else buckets.set(key, [tx]);
  }
  // Map keys preserve the server's newest-first row order. Month totals come
  // from the full server ledger, so loading more rows cannot change a header.
  return Array.from(buckets.entries()).map(([key, items]) => ({
    label: key === "—" ? "—" : new Intl.DateTimeFormat("en-US", { month: "short", year: "numeric", timeZone: "Asia/Manila" }).format(new Date(`${key}-01T00:00:00Z`)).toUpperCase(),
    total: fullMonthTotals.get(key) ?? 0,
    items,
  }));
}
