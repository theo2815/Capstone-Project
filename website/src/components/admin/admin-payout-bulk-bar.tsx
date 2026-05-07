"use client";

import { formatPrice } from "@/lib/utils";

interface AdminPayoutBulkBarProps {
  count: number;
  totalAmount: number;
  approveDisabled: boolean;
  approveDisabledReason: string | null;
  onApprove: () => void;
  onHold: () => void;
  onClear: () => void;
}

// Sticky bottom bar for /admin/payouts multi-select. Slides up from
// `bottom-0` when count >= 1. ESC handling lives on the page (clears
// selection when no modal is open). One-fresh-per-viewport: when the bar
// is visible, the Approve button is the page's only fresh accent.
export function AdminPayoutBulkBar({
  count,
  totalAmount,
  approveDisabled,
  approveDisabledReason,
  onApprove,
  onHold,
  onClear,
}: AdminPayoutBulkBarProps) {
  if (count === 0) return null;

  return (
    <div
      role="region"
      aria-label="Bulk action bar"
      className="fixed inset-x-4 bottom-4 lg:left-auto lg:right-8 lg:bottom-6 lg:max-w-xl z-40 rounded-2xl border border-ink bg-bone shadow-2xl px-4 md:px-6 py-4 animate-[fade-up_0.2s_ease-out_both]"
    >
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="min-w-0">
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
            <span className="tnum">{count}</span> selected
            <span className="text-slate-soft"> · </span>
            <span className="text-ink tnum">{formatPrice(totalAmount)}</span>{" "}
            total
          </p>
          {approveDisabled && approveDisabledReason && (
            <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-amber-700 mt-1.5">
              {approveDisabledReason}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <button
            type="button"
            onClick={onApprove}
            disabled={approveDisabled}
            className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] bg-fresh text-bone hover:bg-fresh-deep transition-colors rounded-full px-4 py-2 disabled:opacity-40 disabled:hover:bg-fresh"
          >
            Approve {count}
          </button>
          <button
            type="button"
            onClick={onHold}
            className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink border border-line hover:bg-ink hover:text-bone hover:border-ink transition-colors rounded-full px-4 py-2"
          >
            Hold {count}…
          </button>
          <button
            type="button"
            onClick={onClear}
            className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors px-3 py-2"
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}
