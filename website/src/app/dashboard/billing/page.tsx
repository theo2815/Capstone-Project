"use client";

import { useState } from "react";
import Link from "next/link";
import { Slab } from "@/components/profile-shell";
import { HowPayoutsModal } from "@/components/dashboard/how-payouts-modal";
import { Kicker } from "@/components/ui/kicker";
import { ROUTES } from "@/lib/constants";
import { formatLongDate, formatMonthYear } from "@/lib/format";
import { formatPayoutNumber } from "@/lib/payout-format";
import {
  PHOTOGRAPHER_PAYOUTS,
  PHOTOGRAPHER_TRANSACTIONS,
  getPhotographerEventById,
  type PayoutStatus,
  type PhotographerPayout,
  type PhotographerTransaction,
} from "@/lib/photographer-mock";
import { cn } from "@/lib/utils";
import {
  PAYOUT_METHOD_LABEL,
  usePhotographerSettingsStore,
} from "@/store/photographer-settings-store";

const STATUS_LABEL: Record<PayoutStatus, string> = {
  paid: "PAID",
  pending: "PENDING",
  scheduled: "SCHEDULED",
};

const STATUS_TONE: Record<PayoutStatus, string> = {
  paid: "text-slate-soft",
  pending: "text-ink",
  scheduled: "text-slate",
};

const CYCLE_MS = 7 * 24 * 60 * 60 * 1000;

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
  const payouts = PHOTOGRAPHER_PAYOUTS;
  const next = pickNextScheduled(payouts);
  const inReviewTotal = payouts
    .filter((p) => p.status === "pending")
    .reduce((sum, p) => sum + p.amount, 0);

  return (
    <Slab
      id="payouts"
      number="01"
      title="Payouts"
      caption="Weekly · GCash"
    >
      <NextPayoutHero
        payout={next}
        inReviewTotal={inReviewTotal}
        onOpenHow={() => setHowOpen(true)}
      />

      {payouts.length > 0 && (
        <div className="mt-12 md:mt-16">
          <div className="flex items-baseline justify-between gap-6 mb-5">
            <Kicker as="p" tone="soft">
              Recent cycles
            </Kicker>
            <Kicker as="p" tone="soft" tnum>
              {payouts.length} cycles
            </Kicker>
          </div>
          <ul className="border-y border-line divide-y divide-line">
            {payouts.map((payout) => (
              <li key={payout.id}>
                <PayoutRow payout={payout} />
              </li>
            ))}
          </ul>
        </div>
      )}

      <HowPayoutsModal isOpen={howOpen} onClose={() => setHowOpen(false)} />
    </Slab>
  );
}

function NextPayoutHero({
  payout,
  inReviewTotal,
  onOpenHow,
}: {
  payout: PhotographerPayout | undefined;
  inReviewTotal: number;
  onOpenHow: () => void;
}) {
  const primary = usePhotographerSettingsStore((s) =>
    s.payouts.find((p) => p.isPrimary),
  );
  const salesInCycle = payout ? countSalesInCycle(payout.weekOf) : 0;

  if (!payout) {
    return (
      <div>
        <Kicker as="p" tone="soft">
          Next payout
        </Kicker>
        <p className="font-display text-3xl md:text-4xl font-medium tracking-tight text-ink mt-3">
          No payout scheduled
        </p>
        <p className="font-sans text-sm md:text-base text-slate mt-3 max-w-md">
          Cycles run Monday to Sunday. Your next sale opens the next cycle.
        </p>
        <HowItWorksLink onClick={onOpenHow} />
      </div>
    );
  }

  return (
    <div>
      <Kicker as="p" tone="soft">
        Next payout
      </Kicker>
      <p className="font-display text-5xl md:text-7xl font-semibold tracking-tight text-fresh tnum mt-3 leading-none">
        ₱{payout.amount.toLocaleString()}
      </p>
      <p className="font-sans text-base md:text-lg text-ink mt-4">
        Releases{" "}
        <span className="font-mono tnum">
          {formatLongDate(payout.settledAt)}
        </span>
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
            href={`${ROUTES.DASHBOARD_SETTINGS}#payouts`}
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
        <p className="font-sans text-sm text-slate mt-3">
          {salesInCycle > 0 ? (
            <>
              Includes{" "}
              <span className="font-mono tnum text-ink-soft">
                {salesInCycle}
              </span>{" "}
              {salesInCycle === 1 ? "sale" : "sales"} ·{" "}
            </>
          ) : null}
          processing 0–24h after release
        </p>
      </div>

      {inReviewTotal > 0 && (
        <p className="font-sans text-sm text-ink-soft mt-5 max-w-md">
          ₱
          <span className="font-mono tnum">
            {inReviewTotal.toLocaleString()}
          </span>{" "}
          still in review from last cycle.
        </p>
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

function PayoutRow({ payout }: { payout: PhotographerPayout }) {
  return (
    <div className="py-5 md:py-6 flex flex-col md:flex-row md:items-baseline md:justify-between gap-2 md:gap-6">
      <div className="flex-1 min-w-0">
        <Kicker as="p" tnum className="flex items-center gap-2 flex-wrap">
          <span>Cycle of {formatLongDate(payout.weekOf, true)}</span>
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
  const transactions = PHOTOGRAPHER_TRANSACTIONS;
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
        <ul className="border-y border-line divide-y divide-line">
          {/* Group by month so the ledger is easier to skim. */}
          {groupByMonth(transactions).map((group) => (
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
      )}
    </Slab>
  );
}

function TransactionRow({ tx }: { tx: PhotographerTransaction }) {
  const event = getPhotographerEventById(tx.eventId);

  return (
    <div className="py-4 md:py-5 flex items-baseline justify-between gap-4">
      <div className="flex-1 min-w-0">
        <Kicker as="p" tnum>
          {formatLongDate(tx.paidAt)}
        </Kicker>
        <p className="font-sans text-base text-ink mt-2 truncate">
          {event?.name ?? "Event archived"}
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

function pickNextScheduled(
  payouts: ReadonlyArray<PhotographerPayout>,
): PhotographerPayout | undefined {
  return [...payouts]
    .filter((p) => p.status === "scheduled")
    .sort((a, b) => a.settledAt.localeCompare(b.settledAt))[0];
}

// Sales whose paidAt falls inside the cycle window [weekOf, weekOf + 7 days).
function countSalesInCycle(weekOf: string): number {
  const start = new Date(`${weekOf}T00:00:00.000Z`).getTime();
  if (Number.isNaN(start)) return 0;
  const end = start + CYCLE_MS;
  return PHOTOGRAPHER_TRANSACTIONS.filter((tx) => {
    const t = new Date(tx.paidAt).getTime();
    return !Number.isNaN(t) && t >= start && t < end;
  }).length;
}

interface MonthGroup {
  label: string;
  total: number;
  items: ReadonlyArray<PhotographerTransaction>;
}

function groupByMonth(
  transactions: ReadonlyArray<PhotographerTransaction>,
): ReadonlyArray<MonthGroup> {
  const buckets = new Map<string, PhotographerTransaction[]>();
  for (const tx of transactions) {
    const key = monthKey(tx.paidAt);
    const list = buckets.get(key);
    if (list) list.push(tx);
    else buckets.set(key, [tx]);
  }
  // Map keys are insertion-ordered; PHOTOGRAPHER_TRANSACTIONS is newest-first
  // so the buckets iterate newest-month first.
  return Array.from(buckets.entries()).map(([key, items]) => ({
    label: formatMonthYear(`${key}-01T00:00:00Z`),
    total: items.reduce((sum, tx) => sum + tx.amountKept, 0),
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
