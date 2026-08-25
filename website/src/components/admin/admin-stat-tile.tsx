import Link from "next/link";
import { cn } from "@/lib/utils";

interface StatTileProps {
  number: string;
  kicker: string;
  value: number | string;
  caption?: string;
  href?: string;
  variant?: "default" | "muted";
}

// Reusable KPI tile for the admin Overview. Mono kicker + display value +
// optional caption + optional `href` (turns the card into a Link with chevron).
// Variant `muted` softens borders + text for "coming next" / Phase 2b stub
// tiles so they read as informational rather than active.
export function AdminStatTile({
  number,
  kicker,
  value,
  caption,
  href,
  variant = "default",
}: StatTileProps) {
  const isMuted = variant === "muted";
  const isInteractive = !!href && !isMuted;
  const baseClasses = cn(
    "block rounded-2xl border p-6 md:p-7 transition-all duration-300",
    isMuted
      ? "border-dashed border-line bg-bone"
      : "border-line bg-bone",
    isInteractive && "hover:border-ink hover:-translate-y-[2px] group",
  );

  const inner = (
    <>
      <p
        className={cn(
          "font-mono uppercase tracking-[0.18em] text-[13px] min-[400px]:text-[14px] md:text-[12px]",
          isMuted ? "text-slate-soft" : "text-slate-soft",
        )}
      >
        <span className="tnum">{number}</span> · {kicker}
      </p>
      <p
        className={cn(
          "font-display text-3xl md:text-4xl font-medium tracking-tight mt-4 tnum",
          isMuted ? "text-slate" : "text-ink",
        )}
      >
        {typeof value === "number" ? value.toLocaleString() : value}
      </p>
      {caption && (
        <p
          className={cn(
            "font-sans text-sm mt-1",
            isMuted ? "text-slate-soft" : "text-ink-soft",
          )}
        >
          {caption}
        </p>
      )}
      {isInteractive && (
        <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink group-hover:text-fresh transition-colors mt-6">
          Open →
        </p>
      )}
    </>
  );

  if (isInteractive && href) {
    return (
      <Link href={href} className={baseClasses}>
        {inner}
      </Link>
    );
  }
  return <div className={baseClasses}>{inner}</div>;
}
