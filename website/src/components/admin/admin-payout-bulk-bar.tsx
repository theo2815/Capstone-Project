"use client";

import { formatPrice } from "@/lib/utils";
import { Kicker } from "@/components/ui/kicker";
import {
  BTN_DANGER,
  BTN_GHOST,
  BTN_PRIMARY,
  BTN_SIZE,
} from "@/components/ui/button-styles";

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
          <Kicker as="p">
            <span className="tnum">{count}</span> selected
            <span className="text-slate-soft"> · </span>
            <span className="text-ink tnum">{formatPrice(totalAmount)}</span>{" "}
            total
          </Kicker>
          {approveDisabled && approveDisabledReason && (
            <p className="font-mono uppercase tracking-[0.18em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-amber-700 mt-1.5">
              {approveDisabledReason}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <button
            type="button"
            onClick={onApprove}
            disabled={approveDisabled}
            className={`${BTN_PRIMARY} ${BTN_SIZE.sm} disabled:hover:bg-fresh`}
          >
            Approve {count}
          </button>
          <button
            type="button"
            onClick={onHold}
            className={`${BTN_DANGER} ${BTN_SIZE.sm}`}
          >
            Hold {count}…
          </button>
          <button
            type="button"
            onClick={onClear}
            className={`${BTN_GHOST} ${BTN_SIZE.sm} px-3`}
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}
