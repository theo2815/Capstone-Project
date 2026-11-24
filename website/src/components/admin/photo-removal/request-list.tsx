"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo } from "react";
import { useQueueKeyboardNav } from "@/hooks/use-admin-keyboard";
import { useUrlState, useUrlStateBatch } from "@/hooks/use-url-state";
import { useAdminRemovalRequests } from "@/hooks/use-photo-removal";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { AdminFieldLabel } from "@/components/admin/admin-form-fields";
import { AdminStatusTabs } from "@/components/admin/admin-status-tabs";
import { BTN_GHOST, BTN_SIZE } from "@/components/ui/button-styles";
import { DropdownSelect } from "@/components/ui/dropdown";
import { LoadError } from "@/components/ui/load-error";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Skeleton } from "@/components/ui/skeleton";
import { useAdminQueueCounts } from "@/lib/admin-queues";
import { ROUTES } from "@/lib/constants";
import { formatLongDate, sentenceCaseDate } from "@/lib/format";
import { REMOVAL_STATUS_LABEL } from "@/lib/admin-removal-labels";
import { cn } from "@/lib/utils";
import type { PhotoRemovalRequestDto, RemovalRequestSource, RemovalRequestStatus } from "@/types/photo-removal";

const PAGE = 20;
// List rows arrive without items; count photos that passed verification from server phase counts.
const verifiedPhotos = (c: PhotoRemovalRequestDto["phaseCounts"]) => c.verified + c.removalAuthorized + c.removed + c.retained + c.retainedRestricted;
const SOURCE_LABEL: Record<RemovalRequestSource, string> = { RUNNER: "Runner", SUPPORT: "Support" };
const SOURCE_OPTIONS: ReadonlyArray<{ value: RemovalRequestSource | ""; label: string }> = [
  { value: "", label: "Runner and support" }, { value: "RUNNER", label: "Runner requests" }, { value: "SUPPORT", label: "Support requests" },
];
type Tab = "waiting" | "all" | RemovalRequestStatus;

// needsActionOnly: the Inbox shows only the cases waiting on an admin, without the status tabs.
export function RemovalRequestList({ needsActionOnly = false }: { needsActionOnly?: boolean }) {
  // Filters live in the URL so Back from a case returns to the same view (loaded pages stay cached).
  const [statusRaw] = useUrlState<string>("status", "");
  const [sourceRaw] = useUrlState<string>("source", "");
  const [needsRaw] = useUrlState<string>("needs", "");
  const writeUrl = useUrlStateBatch();
  const status = (Object.hasOwn(REMOVAL_STATUS_LABEL, statusRaw) ? statusRaw : "") as RemovalRequestStatus | "";
  const source = (Object.hasOwn(SOURCE_LABEL, sourceRaw) ? sourceRaw : "") as RemovalRequestSource | "";
  // No params (or needs=1) is Waiting; All needs needs=0, because the default view is not the unfiltered one.
  const tab: Tab = needsActionOnly ? "waiting" : status || (needsRaw === "0" ? "all" : "waiting");
  const waitingCount = useAdminQueueCounts().byKey.removals;
  // A filter change starts a new list; the old offset param from Prev/Next is dropped.
  const setFilter = (patch: Record<string, string | null>) => writeUrl({ ...patch, offset: null });
  const pickTab = (next: Tab) =>
    setFilter(next === "waiting" ? { status: null, needs: "1" } : next === "all" ? { status: null, needs: "0" } : { status: next, needs: null });
  const byStatus = tab !== "waiting" && tab !== "all";
  const filtered = byStatus || source !== "";
  const list = useAdminRemovalRequests({
    ...(byStatus && { status: tab }), ...(source && { source }), ...(tab === "waiting" && { needsAction: true }),
  }, PAGE);
  const rows = list.items;
  const router = useRouter();
  const requestHref = (id: string) => `${ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS}/${encodeURIComponent(id)}`;
  const rowIds = useMemo(() => rows.map((r) => r.id), [rows]);
  // Enter opens the request's own page, as a click on its row does.
  const { focusedId } = useQueueKeyboardNav({ rowIds, disabled: false, onActivate: (id) => router.push(requestHref(id)) });
  return (
    // The status tabs bring their own top margin; the Inbox has none, so the list keeps mt-8 there.
    <div className={cn("space-y-4", needsActionOnly && "mt-8")}>
      {!needsActionOnly && (
        <>
          <AdminStatusTabs<Tab>
            label="Status"
            tabs={[
              { key: "waiting", label: "Waiting", count: waitingCount ?? undefined },
              ...(Object.entries(REMOVAL_STATUS_LABEL) as [RemovalRequestStatus, string][]).map(([key, label]) => ({ key, label })),
              { key: "all", label: "All" },
            ]}
            value={tab}
            onChange={pickTab}
          />
          <p className="font-sans text-sm text-slate">Closing: a removal was authorized and the files are still being deleted.</p>
        </>
      )}
      <div className="space-y-1 sm:max-w-xs">
        <AdminFieldLabel htmlFor="removal-filter-source">Source</AdminFieldLabel>
        <DropdownSelect id="removal-filter-source" value={source} options={SOURCE_OPTIONS} onChange={(next) => setFilter({ source: next || null })} />
      </div>
      {list.isLoading && (
        <div role="status" aria-busy="true" aria-label="Loading removal requests" className="space-y-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-16 w-full" />)}</div>
      )}
      {!!list.error && <LoadError what="removal requests" onRetry={() => void list.refetch()} />}
      {!list.isLoading && !list.error && rows.length === 0 && (filtered ? (
        <div className="flex flex-wrap items-center gap-3">
          <p className="text-slate">No requests match these filters.</p>
          <button type="button" className={cn(BTN_GHOST, BTN_SIZE.sm)} onClick={() => setFilter({ status: null, source: null, needs: null })}>Clear filters</button>
        </div>
      ) : <p className="text-slate">{tab === "waiting" ? "No requests waiting. You're caught up." : "No removal requests yet."}</p>)}
      <ul className="divide-y divide-line border-y border-line">
        {rows.map((row) => {
          const runner = row.source === "RUNNER";
          return (
            <li key={row.id} data-row-id={row.id}>
              <Link href={requestHref(row.id)} className={cn("flex flex-wrap items-center gap-x-4 gap-y-1 py-4 px-2 hover:bg-bone-deep/50", focusedId === row.id && "ring-2 ring-inset ring-ink")}>
                {/* Below md the name takes the whole first line, so the count, date and pills wrap under it instead of cutting it. */}
                <span className="min-w-0 grow basis-full md:basis-0">
                  <span className="block font-display font-bold text-ink truncate">{row.event.name}</span>
                  {/* A runner's case has only a system reference; the event and date already identify it. */}
                  <span className="block text-sm text-slate truncate">{runner ? "Filed by the runner" : row.supportReference}</span>
                </span>
                <span className="font-mono tnum text-sm text-slate">{verifiedPhotos(row.phaseCounts)} photos · {sentenceCaseDate(formatLongDate(row.createdAt))}</span>
                <AdminStatusPill label={SOURCE_LABEL[row.source ?? "SUPPORT"]} tone={runner ? "ink" : "muted"} />
                <AdminStatusPill label={REMOVAL_STATUS_LABEL[row.status]} tone={row.status === "DRAFT" ? "amber" : row.status === "OPEN" ? "ink" : "slate"} />
              </Link>
            </li>
          );
        })}
      </ul>
      <LoadMoreButton shown={rows.length} total={list.total} increment={PAGE} onLoadMore={list.fetchNextPage} isLoading={list.isFetchingNextPage} />
    </div>
  );
}
