"use client";

import { useMemo, useState } from "react";
import { KICKER_SIZE_CLASS } from "@/components/ui/kicker";
import { AdminDetailDrawer } from "@/components/admin/admin-detail-drawer";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { AdminStatusTabs } from "@/components/admin/admin-status-tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { LoadError } from "@/components/ui/load-error";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { useAdminKpis } from "@/hooks/use-admin-data";
import { useOpenRelevanceAppealsCount, useRelevanceCase, useRelevanceCases, useRelevanceHiddenPhotos } from "@/hooks/use-admin-relevance";
import { useUrlState } from "@/hooks/use-url-state";
import {
  CONSEQUENCES_TEXT,
  RELEVANCE_TABS,
  WAITING_FOR_YOU,
  caseActions,
  caseStateLabel,
  isReleasable,
  isWouldHave,
  type RelevanceCaseAction,
  parseRelevanceTab,
  verdictLabel,
  type RelevanceCaseRow,
  type RelevanceTab,
} from "@/lib/admin-relevance";
import { formatLongDate, sentenceCaseDate } from "@/lib/format";
import { RelevanceCaseBody, RelevanceCaseSubtitle } from "./relevance-case-drawer";
import { RelevanceCaseActions, RelevanceHiddenPhotos } from "./relevance-case-actions";
import { RelevanceAppeals } from "./relevance-appeals";

const PAGE_INCREMENT = 50;
const kicker = `font-mono uppercase ${KICKER_SIZE_CLASS.sm}`;

// The cases and their actions. The tab (`?tab=`) and the open case (`?row=`) live in the URL, so a link opens the same view.
// defaultTab: both the Inbox and the full page open on "Can't tell", the cases a person must decide (FT-20).
export function RelevanceCases({ defaultTab = "cant_tell" }: { defaultTab?: RelevanceTab }) {
  const [tab, setTab] = useUrlState<RelevanceTab>("tab", defaultTab, { parse: parseRelevanceTab });
  const [rowId, setRowId] = useUrlState<string>("row", "");
  // RV-4.3: "Photographers with repeat cases" lives on the All tab only (`?repeat=1`).
  const [repeatParam, setRepeatParam] = useUrlState<string>("repeat", "");
  const repeat = tab === "all" && repeatParam === "1";
  const appeals = tab === "appeals";
  const openAppeals = useOpenRelevanceAppealsCount();
  const kpis = useAdminKpis();
  // AX-8 tab totals; an older backend leaves them out and the tabs show no number (appeals keep their own count).
  const counts: Partial<Record<RelevanceTab, number>> = {
    hidden: kpis?.relevanceHidden,
    cant_tell: kpis?.relevanceCantTell,
    appeals: kpis?.relevanceAppealsOpen ?? (openAppeals || undefined),
  };
  const list = useRelevanceCases(tab, repeat);
  const byId = useMemo(() => new Map(list.items.map((c) => [c.caseId, c])), [list.items]);
  // RV-3.5: on the Appeals tab `?row=` names an appeal, which RelevanceAppeals reads itself.
  const caseId = appeals ? "" : rowId;
  const detail = useRelevanceCase(caseId);
  // RV-2.6: the drawer's actions. The holds decide whether a record-only case can still be restored.
  const held = useRelevanceHiddenPhotos(caseId);
  const [confirming, setConfirming] = useState<RelevanceCaseAction | null>(null);
  // shortcut: reads only the loaded pages of holds, newest first; fine while batches hold a few hundred photos.
  const kinds = detail.data ? caseActions(detail.data, held.items.some(isReleasable)) : [];
  // A ?row= link outside the loaded page still opens: the drawer reads the case by id.
  const openRow: RelevanceCaseRow | null = rowId ? (byId.get(rowId) ?? detail.data?.case ?? null) : null;

  return (
    <div className="pt-6 md:pt-8 space-y-6">
      <AdminStatusTabs
        label="Photo Relevance views"
        tabs={RELEVANCE_TABS.map((t) => ({ key: t.value, label: t.label, count: counts[t.value] }))}
        value={tab}
        onChange={(next) => {
          // A new view starts with no case open and no filter.
          setRowId("");
          setRepeatParam("");
          setTab(next);
        }}
      />

      {tab === "all" && (
        <button
          type="button"
          aria-pressed={repeat}
          onClick={() => {
            setRowId("");
            setRepeatParam(repeat ? "" : "1");
          }}
          className={`${kicker} min-h-[44px] rounded-full border px-4 py-2 transition-colors ${repeat ? "border-ink bg-ink/5 text-ink" : "border-line text-slate-soft hover:text-ink"}`}
        >
          {CONSEQUENCES_TEXT.repeat}
        </button>
      )}

      {appeals ? (
        <RelevanceAppeals appealId={rowId} onOpen={setRowId} />
      ) : list.isLoading ? (
        <div role="status" aria-busy="true" aria-label="Loading cases" className="space-y-3">
          {[0, 1, 2].map((i) => <Skeleton key={i} className="h-24 rounded-2xl" />)}
        </div>
      ) : list.error && list.items.length === 0 ? (
        <LoadError what="the cases" onRetry={list.refetch} className="text-sm text-error" />
      ) : list.items.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">{repeat ? CONSEQUENCES_TEXT.repeatEmpty : RELEVANCE_TABS.find((t) => t.value === tab)?.empty}</p>
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

      {caseId && (
        <AdminDetailDrawer
          open={true}
          onClose={() => {
            setConfirming(null);
            setRowId("");
          }}
          // A re-read can drop the open action (its confirm unmounts); the drawer's Escape then works again.
          escDisabled={confirming !== null && kinds.includes(confirming)}
          actions={
            detail.data && kinds.length > 0 && (
              <RelevanceCaseActions detail={detail.data} kinds={kinds} confirming={confirming} onConfirming={setConfirming} />
            )
          }
          kicker={`Case · ${rowId}`}
          title={openRow?.eventName ?? "Case"}
          rightHeader={openRow && <AdminStatusPill label={verdictLabel(openRow.verdict)} tone={isWouldHave(openRow) ? "amber" : "slate"} />}
          subtitle={openRow && <RelevanceCaseSubtitle row={openRow} />}
        >
          {detail.isPending ? (
            <div aria-busy="true" className="space-y-3">{Array.from({ length: 2 }, (_, i) => <Skeleton key={i} className="h-14 w-full rounded-xl" />)}</div>
          ) : detail.isError ? (
            <LoadError what="the case" onRetry={() => void detail.refetch()} className="text-sm text-error" />
          ) : (
            <div className="space-y-10">
              <RelevanceCaseBody detail={detail.data} />
              <RelevanceHiddenPhotos detail={detail.data} />
            </div>
          )}
        </AdminDetailDrawer>
      )}
    </div>
  );
}

function CaseRow({ row, onOpen }: { row: RelevanceCaseRow; onOpen: () => void }) {
  const state = caseStateLabel(row);
  return (
    <button
      type="button"
      onClick={onOpen}
      className="w-full text-left rounded-2xl border border-line bg-surface p-4 md:p-5 transition-colors hover:bg-bone-deep/50"
    >
      <span className="flex flex-wrap items-center gap-2">
        <AdminStatusPill label={verdictLabel(row.verdict)} tone="slate" />
        {isWouldHave(row) && <AdminStatusPill label="Would have hidden" tone="amber" />}
        {state && <AdminStatusPill label={state} tone={state === WAITING_FOR_YOU ? "amber" : state === "Hidden now" ? "ink" : "muted"} />}
      </span>
      <span className="mt-3 block font-display text-base md:text-lg font-bold text-ink">{row.eventName}</span>
      <span className="mt-1 block font-sans text-sm text-slate-soft tnum">
        {row.photographerHandle ? `@${row.photographerHandle}` : row.photographerName}
        {" · "}
        {row.batchPhotos} photos
        {row.junkPercent !== null && ` · ${row.junkPercent}% junk`}
        {" · "}
        {sentenceCaseDate(formatLongDate(row.decidedAt))}
      </span>
    </button>
  );
}
