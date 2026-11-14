"use client";

import { useMemo } from "react";
import { AdminDetailDrawer } from "@/components/admin/admin-detail-drawer";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { useRelevanceCase, useRelevanceCases } from "@/hooks/use-admin-relevance";
import { useUrlState } from "@/hooks/use-url-state";
import {
  RELEVANCE_TABS,
  isWouldHave,
  parseRelevanceTab,
  verdictLabel,
  type RelevanceCaseRow,
  type RelevanceTab,
} from "@/lib/admin-relevance";
import { formatLongDate } from "@/lib/format";
import { RelevanceTabBar } from "./relevance-tab-bar";
import { RelevanceCaseBody, RelevanceCaseSubtitle } from "./relevance-case-drawer";

const PAGE_INCREMENT = 50;
const kicker = "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px]";

// RV-1.11a: read-only. The tab (`?tab=`) and the open case (`?row=`) live in the URL, so a link opens the same view.
export function RelevanceCases() {
  const [tab, setTab] = useUrlState<RelevanceTab>("tab", "hidden", { parse: parseRelevanceTab });
  const [rowId, setRowId] = useUrlState<string>("row", "");
  const list = useRelevanceCases(tab);
  const byId = useMemo(() => new Map(list.items.map((c) => [c.caseId, c])), [list.items]);
  const detail = useRelevanceCase(rowId);
  // A ?row= link outside the loaded page still opens: the drawer reads the case by id.
  const openRow: RelevanceCaseRow | null = rowId ? (byId.get(rowId) ?? detail.data?.case ?? null) : null;

  return (
    <div className="pt-6 md:pt-8 space-y-6">
      <RelevanceTabBar
        tab={tab}
        onChange={(next) => {
          // A new view starts with no case open.
          setRowId("");
          setTab(next);
        }}
      />

      {list.isLoading ? (
        <p role="status" className="font-sans text-sm text-slate-soft">Loading cases…</p>
      ) : list.error && list.items.length === 0 ? (
        <div role="alert" className="space-y-3">
          <p className="font-sans text-sm text-ink-soft">Could not load the cases.</p>
          <button type="button" onClick={list.refetch} className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`}>Retry</button>
        </div>
      ) : list.items.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">{RELEVANCE_TABS.find((t) => t.value === tab)?.empty}</p>
      ) : (
        <>
          <ul className="space-y-3">
            {list.items.map((c) => (
              <li key={c.caseId}>
                <CaseRow row={c} onOpen={() => setRowId(c.caseId)} />
              </li>
            ))}
          </ul>
          <LoadMoreButton
            shown={list.items.length}
            total={list.total}
            increment={PAGE_INCREMENT}
            onLoadMore={list.fetchNextPage}
            isLoading={list.isFetchingNextPage}
            label="Load more"
          />
        </>
      )}

      {rowId && (
        <AdminDetailDrawer
          open={true}
          onClose={() => setRowId("")}
          kicker={`Case · ${rowId}`}
          title={openRow?.eventName ?? "Case"}
          rightHeader={openRow && <AdminStatusPill label={verdictLabel(openRow.verdict)} tone={isWouldHave(openRow) ? "amber" : "slate"} />}
          subtitle={openRow && <RelevanceCaseSubtitle row={openRow} />}
        >
          {detail.isPending ? (
            <p role="status" className="font-sans text-sm text-slate-soft">Loading case…</p>
          ) : detail.isError ? (
            <div role="alert" className="space-y-3">
              <p className="font-sans text-sm text-ink-soft">Could not load the case.</p>
              <button type="button" onClick={() => void detail.refetch()} className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`}>Retry</button>
            </div>
          ) : (
            <RelevanceCaseBody detail={detail.data} />
          )}
        </AdminDetailDrawer>
      )}
    </div>
  );
}

function CaseRow({ row, onOpen }: { row: RelevanceCaseRow; onOpen: () => void }) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className="w-full text-left rounded-2xl border border-line bg-surface p-4 md:p-5 transition-colors hover:bg-bone-deep/50"
    >
      <span className="flex flex-wrap items-center gap-2">
        <AdminStatusPill label={verdictLabel(row.verdict)} tone="slate" />
        {isWouldHave(row) && <AdminStatusPill label="Would have hidden" tone="amber" />}
      </span>
      <span className="mt-3 block font-display text-base md:text-lg font-bold text-ink">{row.eventName}</span>
      <span className={`${kicker} mt-1 block text-slate-soft tnum`}>
        {row.photographerHandle ? `@${row.photographerHandle}` : row.photographerName}
        {" · "}
        {row.batchPhotos} photos
        {row.junkPercent !== null && ` · ${row.junkPercent}% junk`}
        {" · "}
        {formatLongDate(row.decidedAt)}
      </span>
    </button>
  );
}
