import { cn } from "@/lib/utils";

export type AdminStatusPillTone = "fresh" | "ink" | "slate" | "muted" | "amber";

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
        "inline-flex items-center font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] rounded-full border px-3 py-0.5",
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
  amber: "border-amber-600/30 text-amber-700",
};
