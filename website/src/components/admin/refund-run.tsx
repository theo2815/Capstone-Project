"use client";

import { useState } from "react";
import { BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { AdminRefundPanel } from "@/components/admin/admin-refund-panel";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { useAdminDisputePage, EMPTY_DISPUTES } from "@/hooks/use-admin-disputes";
import { DISPUTE_ATTENTION_LABEL } from "@/lib/admin-disputes";
import { formatSalesDate } from "@/lib/format";
import { cn, formatPrice } from "@/lib/utils";

const SERVER_PAGE_SIZE = 50;
// Mono chips stay on one line at 375 px (ui-pitfalls 2026-05-06).
const CHIP = "block max-w-full truncate tnum";

// Every approved QR Ph refund that still needs paying, oldest due first (the server sorts).
// Each item reuses the refund panel unchanged; Open goes through the queue's setRowId so the view stays in the URL.
export function RefundRun({ onOpen }: { onOpen: (id: string) => void }) {
  const [offset, setOffset] = useState(0);
  const query = useAdminDisputePage({ view: "refund_run", offset });
  const items = query.data?.items ?? EMPTY_DISPUTES;
  const total = query.data?.total ?? 0;
  const limit = query.data?.limit || SERVER_PAGE_SIZE;
  // Refunds left this page (paid, or the list shrank): step back to the last page that has rows.
  if (query.isSuccess && !query.isPlaceholderData && items.length === 0 && offset > 0) {
    setOffset(Math.floor(Math.max(total - 1, 0) / limit) * limit);
  }

  if (query.isPending) return <p role="status" className="py-6">Loading refunds…</p>;
  if (query.isError && !query.data) {
    return (
      <div role="alert" className="space-y-4 py-6">
        <p>Couldn&apos;t load refunds.</p>
        <button type="button" className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`} onClick={() => void query.refetch()}>
          Try again
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <p className="font-sans text-sm text-slate">
        Every approved QR Ph refund that still needs paying. Work through them in one PayMongo session: copy the
        payment ID, make the refund, paste its link, then mark each one refunded once it is claimed.
      </p>

      {items.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">No approved refunds are waiting to be paid.</p>
      ) : (
        <ul className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {items.map((d) => {
            const attention = d.attention ? DISPUTE_ATTENTION_LABEL[d.attention] : null;
            return (
              <li key={d.id}>
                <article data-row-id={d.id} className="space-y-4 rounded-2xl border border-line bg-bone p-5 md:p-6">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <h3 className="font-display text-lg font-medium text-ink truncate">@{d.runnerHandle}</h3>
                      <p className="font-sans text-sm text-slate truncate">{d.eventName ?? "Event archived"}</p>
                      <p className="font-mono tnum text-sm text-ink mt-1">
                        {formatPrice(d.refundAmount ?? d.itemChargeAmount ?? d.orderSnapshot.total)}
                      </p>
                    </div>
                    <button
                      type="button"
                      aria-label={`Open dispute ${d.id}`}
                      onClick={() => onOpen(d.id)}
                      className={cn(BTN_GHOST, BTN_SIZE.sm, "shrink-0")}
                    >
                      Open
                    </button>
                  </div>
                  {(attention || d.dueAt || d.linkExpiringSoon) && (
                    <div className="flex flex-wrap gap-2">
                      {attention && <AdminStatusPill label={attention} tone="ink" className={CHIP} />}
                      {d.dueAt && (
                        <AdminStatusPill
                          label={d.overdue ? `Overdue · due ${formatSalesDate(d.dueAt)}` : `Due ${formatSalesDate(d.dueAt)}`}
                          tone={d.overdue ? "error" : "muted"}
                          className={CHIP}
                        />
                      )}
                      {d.linkExpiringSoon && d.refundLinkExpiresAt && (
                        <AdminStatusPill label={`Link expires ${formatSalesDate(d.refundLinkExpiresAt)}`} tone="amber" className={CHIP} />
                      )}
                    </div>
                  )}
                  <AdminRefundPanel dispute={d} />
                </article>
              </li>
            );
          })}
        </ul>
      )}

      {total > limit && (
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" className={cn(BTN_GHOST, BTN_SIZE.sm)} disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>← Previous</button>
          <span className="font-mono tnum text-sm text-slate">{offset + 1}–{offset + items.length} of {total}</span>
          <button type="button" className={cn(BTN_GHOST, BTN_SIZE.sm)} disabled={offset + items.length >= total} onClick={() => setOffset(offset + items.length)}>Next →</button>
        </div>
      )}
    </div>
  );
}
