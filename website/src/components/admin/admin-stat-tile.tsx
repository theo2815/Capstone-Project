import Link from "next/link";
import { cn } from "@/lib/utils";

interface StatTileProps {
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
  kicker,
  value,
  caption,
  href,
  variant = "default",
}: StatTileProps) {
  const isMuted = variant === "muted";
  const isInteractive = !!href && !isMuted;
  const baseClasses = cn(
    "relative block rounded-2xl border p-6 md:p-7 transition-all duration-300",
    isMuted
      ? "border-dashed border-line bg-bone"
      : "border-line bg-bone",
    isInteractive && "hover:border-ink hover:-translate-y-[2px] group",
  );

  const inner = (
    <>
      <p
        className={cn(
          // Two lines reserved so every tile's value starts at the same height.
          "font-mono uppercase tracking-[0.18em] text-[14px] min-[400px]:text-[15px] md:text-[13px] min-h-[2.8em] pr-6",
          isMuted ? "text-slate-soft" : "text-slate-soft",
        )}
      >
        {kicker}
      </p>
      <p
        className={cn(
          // Wraps only a value too long for its tile (big money on Sales); short counts never do.
          "font-display text-3xl md:text-4xl font-medium tracking-tight mt-4 tnum [overflow-wrap:anywhere]",
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
      {/* The whole tile is the link; a corner arrow marks it instead of a repeated "Open →". */}
      {isInteractive && (
        <span
          aria-hidden="true"
          className="absolute top-6 right-6 md:top-7 md:right-7 font-mono text-[15px] text-slate-soft group-hover:text-fresh transition-colors"
        >
          →
        </span>
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
