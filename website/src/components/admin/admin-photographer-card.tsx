import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { computeCompleteness, photographerHref, photographerStatus } from "@/lib/admin-photographer-view";
import { formatMemberSince } from "@/lib/format";
import type { AdminUserRow } from "@/lib/admin-user-registry";
import { cn } from "@/lib/utils";

interface AdminPhotographerCardProps {
  row: AdminUserRow;
  /** Phase 4: J/K keyboard focus indicator. Adds an ink ring + offset. */
  focused?: boolean;
}

// Directory row for /admin/photographers. Whole card is a Link to the
// detail page. Status pill, completeness fraction, member-since on the
// right; identity on the left. No nested interactive elements — admin
// actions live on the detail page only.
export function AdminPhotographerCard({
  row,
  focused = false,
}: AdminPhotographerCardProps) {
  const completeness = computeCompleteness(row.settingsSnapshot);
  const status = photographerStatus(row);
  const place = `${row.region ?? "Region not set"} · ${row.city}`;

  return (
    <Link href={photographerHref(row)} className="block">
      <article
        data-row-id={row.userId}
        className={cn(
          "rounded-2xl border border-line bg-bone p-5 md:p-6 transition-all duration-300",
          "hover:border-ink hover:-translate-y-[2px]",
          focused && "ring-2 ring-ink ring-offset-2 ring-offset-bone",
        )}
      >
        <div className="flex items-start gap-4 flex-wrap md:flex-nowrap">
          <AvatarDisc name={row.brandName ?? row.name} size="sm" tone="ink" avatarOverride={null} />
          <div className="min-w-0 flex-1">
            <div className="flex items-baseline gap-3 flex-wrap">
              <h3 title={row.brandName ?? row.name} className="font-display text-lg md:text-xl font-medium text-ink truncate">
                {row.brandName ?? row.name}
              </h3>
              {row.handle && (
                <Kicker as="p">
                  @{row.handle}
                </Kicker>
              )}
            </div>
            <Kicker as="p" tone="soft" className="mt-2 truncate">
              <span title={place}>{place}</span>
              <span className="text-slate-soft"> · </span>
              <span className="tnum">since {formatMemberSince(row.createdAt)}</span>
            </Kicker>
          </div>
          <div className="flex flex-col items-end gap-2 shrink-0 ml-auto md:ml-0">
            <AdminStatusPill label={status.label} tone={status.tone} />
            <Kicker as="p" tone="soft" className="tnum">
              {completeness.filled}/{completeness.total} fields
            </Kicker>
          </div>
        </div>
      </article>
    </Link>
  );
}
