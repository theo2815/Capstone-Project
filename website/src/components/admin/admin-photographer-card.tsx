import Link from "next/link";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { ROUTES } from "@/lib/constants";
import { computeCompleteness } from "@/lib/admin-photographer-view";
import { formatMemberSince } from "@/lib/format";
import type { AdminUserRow } from "@/lib/admin-user-registry";
import { cn } from "@/lib/utils";

interface AdminPhotographerCardProps {
  row: AdminUserRow;
  /** Phase 4: J/K keyboard focus indicator. Adds an ink ring + offset. */
  focused?: boolean;
}

const STATUS_LABEL: Record<AdminUserRow["verificationStatus"], string> = {
  approved: "Approved",
  pending: "Pending",
  incomplete: "Incomplete",
};

// Directory row for /admin/photographers. Whole card is a Link to the
// detail page. Status pill, completeness fraction, member-since on the
// right; identity on the left. No nested interactive elements — admin
// actions live on the detail page only.
export function AdminPhotographerCard({
  row,
  focused = false,
}: AdminPhotographerCardProps) {
  const completeness = computeCompleteness(row.settingsSnapshot);
  const isSuspended = row.suspendedAt !== null;
  const handleHref = row.handle
    ? `${ROUTES.ADMIN_PHOTOGRAPHERS}/${row.handle}`
    : ROUTES.ADMIN_PHOTOGRAPHERS;
  const handleDisabled = !row.handle;

  const inner = (
    <article
      data-row-id={row.userId}
      className={cn(
        "rounded-2xl border border-line bg-bone p-5 md:p-6 transition-all duration-300",
        !handleDisabled && "hover:border-ink hover:-translate-y-[2px]",
        focused && "ring-2 ring-ink ring-offset-2 ring-offset-bone",
      )}
    >
      <div className="flex items-start gap-4 flex-wrap md:flex-nowrap">
        <AvatarDisc name={row.brandName ?? row.name} size="sm" tone="ink" avatarOverride={null} />
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-3 flex-wrap">
            <h3 className="font-display text-lg md:text-xl font-medium text-ink truncate">
              {row.brandName ?? row.name}
            </h3>
            {row.handle && (
              <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
                @{row.handle}
              </p>
            )}
          </div>
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft mt-2 truncate">
            {row.region ?? "Region not set"}
            <span className="text-slate-soft"> · </span>
            {row.city}
            <span className="text-slate-soft"> · </span>
            <span className="tnum">since {formatMemberSince(row.createdAt)}</span>
          </p>
        </div>
        <div className="flex flex-col items-end gap-2 shrink-0 ml-auto md:ml-0">
          {/* Same tone language as the other queues: amber = waiting on an admin. */}
          <AdminStatusPill
            label={isSuspended ? "Suspended" : STATUS_LABEL[row.verificationStatus]}
            tone={
              isSuspended
                ? "ink"
                : row.verificationStatus === "approved"
                  ? "fresh"
                  : row.verificationStatus === "pending"
                    ? "amber"
                    : "slate"
            }
          />
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft tnum">
            {completeness.filled}/{completeness.total} fields
          </p>
        </div>
      </div>
    </article>
  );

  if (handleDisabled) {
    return inner;
  }
  return (
    <Link href={handleHref} className="block">
      {inner}
    </Link>
  );
}
