"use client";

import Link from "next/link";
import { useUrlState, useUrlStateBatch } from "@/hooks/use-url-state";
import { useAdminRemovalRequests } from "@/hooks/use-photo-removal";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { AdminFieldLabel } from "@/components/admin/admin-form-fields";
import { BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { ROUTES } from "@/lib/constants";
import { formatLongDate, sentenceCaseDate } from "@/lib/format";
import { REMOVAL_STATUS_LABEL } from "@/lib/admin-removal-labels";
import { cn } from "@/lib/utils";
import { REMOVAL_INPUT } from "./request-intake";
import type { PhotoRemovalRequestDto, RemovalRequestSource, RemovalRequestStatus } from "@/types/photo-removal";

const PAGE = 20;
// List rows arrive without items; count photos that passed verification from server phase counts.
const verifiedPhotos = (c: PhotoRemovalRequestDto["phaseCounts"]) => c.verified + c.removalAuthorized + c.removed + c.retained + c.retainedRestricted;
const SOURCE_LABEL: Record<RemovalRequestSource, string> = { RUNNER: "Runner", SUPPORT: "Support" };

// needsActionOnly: the Inbox shows only the cases waiting on an admin, without the checkbox.
export function RemovalRequestList({ needsActionOnly = false }: { needsActionOnly?: boolean }) {
  // Filters live in the URL so Back from a case returns to the same view (loaded pages stay cached).
  const [statusRaw] = useUrlState<string>("status", "");
  const [sourceRaw] = useUrlState<string>("source", "");
  const [needsRaw] = useUrlState<string>("needs", "");
  const writeUrl = useUrlStateBatch();
  const status = (Object.hasOwn(REMOVAL_STATUS_LABEL, statusRaw) ? statusRaw : "") as RemovalRequestStatus | "";
  const source = (Object.hasOwn(SOURCE_LABEL, sourceRaw) ? sourceRaw : "") as RemovalRequestSource | "";
  const needsAction = needsActionOnly || needsRaw === "1";
  // A filter change starts a new list; the old offset param from Prev/Next is dropped.
  const setFilter = (patch: Record<string, string | null>) => writeUrl({ ...patch, offset: null });
  const filtered = status !== "" || source !== "" || needsRaw === "1";
  const list = useAdminRemovalRequests({ ...(status && { status }), ...(source && { source }), ...(needsAction && { needsAction }) }, PAGE);
  const rows = list.items;
  return (
    <div className="mt-8 space-y-4">
      <div className="grid gap-4 sm:grid-cols-3 sm:items-end">
        <div className="space-y-1">
          <AdminFieldLabel htmlFor="removal-filter-status">Status</AdminFieldLabel>
          <select id="removal-filter-status" value={status} className={REMOVAL_INPUT}
            onChange={(e) => setFilter({ status: e.target.value || null })}>
            <option value="">Any status</option>
            {Object.entries(REMOVAL_STATUS_LABEL).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </div>
        <div className="space-y-1">
          <AdminFieldLabel htmlFor="removal-filter-source">Source</AdminFieldLabel>
          <select id="removal-filter-source" value={source} className={REMOVAL_INPUT}
            onChange={(e) => setFilter({ source: e.target.value || null })}>
            <option value="">Runner and support</option>
            <option value="RUNNER">Runner requests</option>
            <option value="SUPPORT">Support requests</option>
          </select>
        </div>
        {!needsActionOnly && (
          <label className="flex min-h-[44px] items-center gap-2 text-sm text-ink cursor-pointer">
            <input type="checkbox" checked={needsAction} onChange={(e) => setFilter({ needs: e.target.checked ? "1" : null })} />
            Needs action
          </label>
        )}
      </div>
      {list.isLoading && <p role="status" className="text-slate">Loading cases…</p>}
      {!!list.error && (
        <div role="alert" className="space-y-4">
          <p>Couldn&apos;t load removal cases.</p>
          <button type="button" className={cn(BTN_SECONDARY, BTN_SIZE.sm)} onClick={() => void list.refetch()}>Try again</button>
        </div>
      )}
      {!list.isLoading && !list.error && rows.length === 0 && (filtered ? (
        <div className="flex flex-wrap items-center gap-3">
          <p className="text-slate">No cases match these filters.</p>
          <button type="button" className={cn(BTN_GHOST, BTN_SIZE.sm)} onClick={() => setFilter({ status: null, source: null, needs: null })}>Clear filters</button>
        </div>
      ) : <p className="text-slate">{needsActionOnly ? "No requests waiting. You're caught up." : "No removal cases yet."}</p>)}
      <ul className="divide-y divide-line border-y border-line">
        {rows.map((row) => {
          const runner = row.source === "RUNNER";
          return (
            <li key={row.id}>
              <Link href={`${ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS}/${encodeURIComponent(row.id)}`} className="flex flex-wrap items-center gap-x-4 gap-y-1 py-4 px-2 hover:bg-bone-deep/50">
                <span className="min-w-0 flex-1">
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
