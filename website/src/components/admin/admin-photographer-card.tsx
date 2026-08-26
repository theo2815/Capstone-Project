import Link from "next/link";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { ROUTES } from "@/lib/constants";
import { formatMemberSince } from "@/lib/format";
import type {
  AdminUserRow,
  PhotographerSettingsSnapshot,
} from "@/lib/admin-user-registry";
import { cn } from "@/lib/utils";

interface AdminPhotographerCardProps {
  row: AdminUserRow;
  /** Phase 4: J/K keyboard focus indicator. Adds an ink ring + offset. */
  focused?: boolean;
}

const COMPLETENESS_KEYS: ReadonlyArray<keyof PhotographerSettingsSnapshot> = [
  "hasCover",
  "hasBrandName",
  "hasWatermark",
  "hasHandle",
  "hasRegion",
];

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
          <StatusPill
            label={isSuspended ? "Suspended" : STATUS_LABEL[row.verificationStatus]}
            tone={
              isSuspended
                ? "ink"
                : row.verificationStatus === "approved"
                  ? "fresh"
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

function StatusPill({
  label,
  tone,
}: {
  label: string;
  tone: "fresh" | "slate" | "ink";
}) {
  const toneClass =
    tone === "fresh"
      ? "border-fresh/30 text-fresh"
      : tone === "ink"
        ? "border-ink text-ink"
        : "border-line text-slate";
  return (
    <span
      className={cn(
        "inline-flex items-center font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] rounded-full border px-3 py-0.5",
        toneClass,
      )}
    >
      {label}
    </span>
  );
}

function computeCompleteness(
  snapshot: PhotographerSettingsSnapshot | null,
): { filled: number; total: number } {
  if (!snapshot) return { filled: 0, total: COMPLETENESS_KEYS.length + 2 };
  const fieldCount = COMPLETENESS_KEYS.reduce(
    (acc, key) => acc + (snapshot[key] ? 1 : 0),
    0,
  );
  const socials = snapshot.socialCount > 0 ? 1 : 0;
  const payouts = snapshot.payoutCount > 0 ? 1 : 0;
  return {
    filled: fieldCount + socials + payouts,
    total: COMPLETENESS_KEYS.length + 2,
  };
}
