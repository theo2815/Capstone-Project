"use client";

import Link from "next/link";
import { useState } from "react";
import { useAdminRemovalRequests } from "@/hooks/use-photo-removal";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import { cn } from "@/lib/utils";

const PAGE = 20;

export function RemovalRequestList() {
  const [offset, setOffset] = useState(0);
  const list = useAdminRemovalRequests({ offset, limit: PAGE });
  const rows = list.data?.items ?? [];
  const total = list.data?.total ?? 0;
  if (list.isPending) return <p role="status" className="mt-8 text-slate">Loading cases…</p>;
  if (list.isError) return (
    <div role="alert" className="mt-8 space-y-4">
      <p>Couldn&apos;t load removal cases.</p>
      <button type="button" className={cn(BTN_SECONDARY, BTN_SIZE.sm)} onClick={() => void list.refetch()}>Try again</button>
    </div>
  );
  return (
    <div className="mt-8 space-y-4">
      {rows.length === 0 && <p className="text-slate">No removal cases yet.</p>}
      <ul className="divide-y divide-line border-y border-line">
        {rows.map((row) => (
          <li key={row.id}>
            <Link href={`${ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS}/${encodeURIComponent(row.id)}`} className="flex flex-wrap items-center gap-x-4 gap-y-1 py-4 px-2 hover:bg-bone-deep/50">
              <span className="min-w-0 flex-1">
                <span className="block font-display font-bold text-ink truncate">{row.event.name}</span>
                <span className="block text-sm text-slate truncate">{row.supportReference}</span>
              </span>
              <span className="font-mono tnum text-sm text-slate">{row.items.length} photos · {formatLongDate(row.createdAt)}</span>
              <AdminStatusPill label={row.status} tone={row.status === "DRAFT" ? "amber" : row.status === "OPEN" ? "ink" : "slate"} />
            </Link>
          </li>
        ))}
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
