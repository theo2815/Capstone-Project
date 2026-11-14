"use client";

import Link from "next/link";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { useMarkRelevanceAlertSeen, useUnseenRelevanceAlerts } from "@/hooks/use-admin-relevance";
import { useToast } from "@/hooks/use-toast";
import { ALERTS_EMPTY, ALERT_SEEN_FAILED, WOULD_HAVE_LABEL, alertKindLabel, type RelevanceAlert } from "@/lib/admin-relevance";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";

const metaCls = "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft tnum";
const linkCls = "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink hover:text-fresh transition-colors";

// RV-1.11b: the inbox body for the "Relevance alerts" chip. Unseen alerts only; each links to its case on the page.
export function RelevanceAlertsInbox() {
  const list = useUnseenRelevanceAlerts();

  return (
    <div className="pt-6 md:pt-8 space-y-6">
      <Link href={ROUTES.ADMIN_RELEVANCE} className={linkCls}>Open Photo Relevance →</Link>
      {list.isLoading ? (
        <p role="status" className="font-sans text-sm text-slate-soft">Loading alerts…</p>
      ) : list.error && list.items.length === 0 ? (
        <div role="alert" className="space-y-3">
          <p className="font-sans text-sm text-ink-soft">Could not load the alerts.</p>
          <button type="button" onClick={list.refetch} className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`}>Retry</button>
        </div>
      ) : list.items.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">{ALERTS_EMPTY}</p>
      ) : (
        <>
          <ul className="space-y-3">
            {list.items.map((a) => <AlertRow key={a.alertId} alert={a} />)}
          </ul>
          <LoadMoreButton
            shown={list.items.length}
            total={list.total}
            increment={50}
            onLoadMore={list.fetchNextPage}
            isLoading={list.isFetchingNextPage}
            label="Load more"
          />
        </>
      )}
    </div>
  );
}

function AlertRow({ alert }: { alert: RelevanceAlert }) {
  const { markSeen, isPending } = useMarkRelevanceAlertSeen();
  const { showToast } = useToast();
  const pending = isPending(alert.alertId);
  const kind = alertKindLabel(alert.kind);

  async function handleSeen() {
    try {
      await markSeen(alert.alertId);
    } catch {
      showToast({ kind: "error", message: ALERT_SEEN_FAILED });
    }
  }

  return (
    <li aria-label={`${kind} · ${alert.eventName}`} className="rounded-2xl border border-line bg-surface p-4 md:p-5 space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-display text-base md:text-lg font-bold text-ink">{kind}</span>
        {alert.wouldHave && <AdminStatusPill label={WOULD_HAVE_LABEL} tone="amber" />}
      </div>
      <p className="font-sans text-sm text-ink">{alert.eventName}</p>
      <p className={metaCls}>
        {alert.photographerHandle ? `@${alert.photographerHandle}` : alert.photographerName}
        {alert.photos !== null && ` · ${alert.photos.toLocaleString()} photos`}
        {" · "}
        {formatLongDate(alert.createdAt)}
      </p>
      <div className="flex flex-wrap items-center gap-4">
        {alert.caseId && (
          <Link href={`${ROUTES.ADMIN_RELEVANCE}?tab=all&row=${encodeURIComponent(alert.caseId)}`} className={linkCls}>
            Open case →
          </Link>
        )}
        <button type="button" disabled={pending} onClick={handleSeen} className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`}>
          Mark seen
        </button>
      </div>
    </li>
  );
}
