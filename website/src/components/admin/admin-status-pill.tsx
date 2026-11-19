import { cn } from "@/lib/utils";
import { KICKER_SIZE_CLASS } from "@/components/ui/kicker";

export type AdminStatusPillTone = "fresh" | "ink" | "slate" | "muted" | "amber" | "error";

interface AdminStatusPillProps {
  label: string;
  tone: AdminStatusPillTone;
  className?: string;
}

// Generic status pill for the admin moderation surfaces (disputes, flags,
// payouts). Mono caps + thin border + restrained tone palette. Each domain
// computes its own status → tone helper; this component just renders.
export function AdminStatusPill({
  label,
  tone,
  className,
}: AdminStatusPillProps) {
  return (
    <span
      className={cn(
        `inline-flex items-center font-mono uppercase ${KICKER_SIZE_CLASS.sm} rounded-full border px-3 py-0.5`,
        TONE_CLASS[tone],
        className,
      )}
    >
      {label}
    </span>
  );
}

const TONE_CLASS: Record<AdminStatusPillTone, string> = {
  fresh: "border-fresh/30 text-fresh",
  ink: "border-ink text-ink",
  slate: "border-line text-slate",
  muted: "border-line text-slate-soft",
  amber: "border-warning-text/30 text-warning-text",
  error: "border-error/30 text-error",
};
