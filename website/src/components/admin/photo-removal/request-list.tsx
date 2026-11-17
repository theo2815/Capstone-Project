"use client";

import Link from "next/link";
import { useState } from "react";
import { useAdminRemovalRequests } from "@/hooks/use-photo-removal";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { AdminFieldLabel } from "@/components/admin/admin-form-fields";
import { BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import { cn } from "@/lib/utils";
import { REMOVAL_INPUT } from "./request-intake";
import type { PhotoRemovalRequestDto, RemovalRequestFilters, RemovalRequestSource, RemovalRequestStatus } from "@/types/photo-removal";

const PAGE = 20;
// List rows arrive without items; count photos that passed verification from server phase counts.
const verifiedPhotos = (c: PhotoRemovalRequestDto["phaseCounts"]) => c.verified + c.removalAuthorized + c.removed + c.retained + c.retainedRestricted;
export const REMOVAL_STATUS_LABEL: Record<RemovalRequestStatus, string> = {
  DRAFT: "Draft", OPEN: "Open", RESOLVING: "Closing", RESOLVED: "Resolved", CANCELLED: "Cancelled",
};
const SOURCE_LABEL: Record<RemovalRequestSource, string> = { RUNNER: "Runner", SUPPORT: "Support" };

export function RemovalRequestList() {
  const [offset, setOffset] = useState(0);
  const [status, setStatus] = useState<RemovalRequestStatus | "">("");
  const [source, setSource] = useState<RemovalRequestSource | "">("");
  const [needsAction, setNeedsAction] = useState(false);
  const filters: RemovalRequestFilters = { offset, limit: PAGE, ...(status && { status }), ...(source && { source }), ...(needsAction && { needsAction }) };
  const list = useAdminRemovalRequests(filters);
  const rows = list.data?.items ?? [];
  const total = list.data?.total ?? 0;
  return (
    <div className="mt-8 space-y-4">
      <div className="grid gap-4 sm:grid-cols-3 sm:items-end">
        <div className="space-y-1">
          <AdminFieldLabel htmlFor="removal-filter-status">Status</AdminFieldLabel>
          <select id="removal-filter-status" value={status} className={REMOVAL_INPUT}
            onChange={(e) => { setStatus(e.target.value as RemovalRequestStatus | ""); setOffset(0); }}>
            <option value="">Any status</option>
            {Object.entries(REMOVAL_STATUS_LABEL).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </div>
        <div className="space-y-1">
          <AdminFieldLabel htmlFor="removal-filter-source">Source</AdminFieldLabel>
          <select id="removal-filter-source" value={source} className={REMOVAL_INPUT}
            onChange={(e) => { setSource(e.target.value as RemovalRequestSource | ""); setOffset(0); }}>
            <option value="">Runner and support</option>
            <option value="RUNNER">Runner requests</option>
            <option value="SUPPORT">Support requests</option>
          </select>
        </div>
        <label className="flex min-h-[44px] items-center gap-2 text-sm text-ink cursor-pointer">
          <input type="checkbox" checked={needsAction} onChange={(e) => { setNeedsAction(e.target.checked); setOffset(0); }} />
          Needs action
        </label>
      </div>
      {list.isPending && <p role="status" className="text-slate">Loading cases…</p>}
      {list.isError && (
        <div role="alert" className="space-y-4">
          <p>Couldn&apos;t load removal cases.</p>
          <button type="button" className={cn(BTN_SECONDARY, BTN_SIZE.sm)} onClick={() => void list.refetch()}>Try again</button>
        </div>
      )}
      {list.isSuccess && rows.length === 0 && <p className="text-slate">No removal cases yet.</p>}
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
                <span className="font-mono tnum text-sm text-slate">{verifiedPhotos(row.phaseCounts)} photos · {formatLongDate(row.createdAt)}</span>
                <AdminStatusPill label={SOURCE_LABEL[row.source ?? "SUPPORT"]} tone={runner ? "ink" : "muted"} />
                <AdminStatusPill label={REMOVAL_STATUS_LABEL[row.status]} tone={row.status === "DRAFT" ? "amber" : row.status === "OPEN" ? "ink" : "slate"} />
              </Link>
            </li>
          );
        })}
      </ul>
      {total > PAGE && (
        <div className="flex items-center gap-2">
          <button type="button" className={cn(BTN_GHOST, BTN_SIZE.sm)} disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>← Previous</button>
          <span className="font-mono tnum text-sm text-slate">{offset + 1}–{Math.min(offset + PAGE, total)} of {total}</span>
          <button type="button" className={cn(BTN_GHOST, BTN_SIZE.sm)} disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>Next →</button>
        </div>
      )}
    </div>
  );
}
