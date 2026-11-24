"use client";

import { LoadError } from "@/components/ui/load-error";
import { Skeleton } from "@/components/ui/skeleton";
import { BTN_GHOST, BTN_SIZE } from "@/components/ui/button-styles";
import { AdminRefundPanel } from "@/components/admin/admin-refund-panel";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { useAdminDisputePage } from "@/hooks/use-admin-disputes";
import { DISPUTE_ATTENTION_LABEL } from "@/lib/admin-disputes";
import { formatSalesDate } from "@/lib/format";
import { cn, formatPrice } from "@/lib/utils";

// The server page size, for the button's count.
const PAGE_INCREMENT = 50;
// Mono chips stay on one line at 375 px (ui-pitfalls 2026-05-06).
const CHIP = "block max-w-full truncate tnum";

// Every approved QR Ph refund that still needs paying, oldest due first (the server sorts).
// Each item reuses the refund panel unchanged; Open goes through the queue's setRowId so the view stays in the URL.
export function RefundRun({ onOpen }: { onOpen: (id: string) => void }) {
  const list = useAdminDisputePage({ view: "refund_run" });
  const items = list.items;

  if (list.isLoading) return <div aria-busy="true" className="space-y-4">{Array.from({ length: 3 }, (_, i) => <Skeleton key={i} className="h-28 w-full rounded-2xl" />)}</div>;
  if (list.error && items.length === 0) return <LoadError what="refunds" onRetry={list.refetch} className="text-sm text-error" />;

  return (
    <div className="space-y-6">
      <p className="font-sans text-sm text-slate">
        Every approved QR Ph refund that still needs paying. Work through them in one PayMongo session.
      </p>

      {items.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">No approved refunds are waiting to be paid.</p>
      ) : (
        <ul className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {items.map((d) => {
            const attention = d.attention ? DISPUTE_ATTENTION_LABEL[d.attention] : null;
            const amount = d.refundAmount ?? d.itemChargeAmount;
            return (
              <li key={d.id}>
                <article data-row-id={d.id} className="space-y-4 rounded-2xl border border-line bg-bone p-5 md:p-6">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <h3 className="font-display text-lg font-medium text-ink truncate">@{d.runnerHandle}</h3>
                      <p className="font-mono text-[13px] text-slate tnum">{d.id}</p>
                      <p className="font-sans text-sm text-slate truncate">{d.eventName ?? "Event archived"}</p>
                      <p className="font-mono tnum text-sm text-ink mt-1">
                        {amount != null ? formatPrice(amount) : `Order total ${formatPrice(d.orderSnapshot.total)}`}
                      </p>
                    </div>
                    <button
                      type="button"
                      aria-label={`Open dispute ${d.id}`}
                      onClick={() => onOpen(d.id)}
                      className={cn(BTN_GHOST, BTN_SIZE.sm, "shrink-0")}
                    >
                      Open dispute
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

      {/* Like the old pager: only once the list is longer than one server page. */}
      {list.total > PAGE_INCREMENT && (
        <LoadMoreButton
          shown={items.length}
          total={list.total}
          increment={PAGE_INCREMENT}
          onLoadMore={list.fetchNextPage}
          isLoading={list.isFetchingNextPage}
          label="Load more"
        />
      )}
    </div>
  );
}
