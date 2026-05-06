import type { ReactNode } from "react";

interface SlabProps {
  id?: string;
  number: string;
  title: string;
  caption?: string;
  trailing?: ReactNode;
  children: ReactNode;
}

// Section vocabulary shared across /profile, /account, /orders, /dashboard.
// Mono kicker ("01 · TITLE · caption") with optional trailing right-aligned
// metadata. Uses the same border/scroll-mt rhythm everywhere so anchor jumps
// clear the sticky SiteHeader.
export function Slab({
  id,
  number,
  title,
  caption,
  trailing,
  children,
}: SlabProps) {
  return (
    <section
      id={id}
      className="border-t border-line py-12 md:py-16 scroll-mt-24 first:border-0 first:pt-10 md:first:pt-20"
    >
      <div className="flex items-baseline justify-between gap-6 mb-8 md:mb-10">
        <div className="flex items-baseline gap-4 min-w-0">
          <span className="font-mono text-[10px] tracking-[0.15em] text-slate-soft tnum">
            {number}
          </span>
          <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-ink shrink-0">
            {title}
          </p>
          {caption && (
            <p className="hidden md:block font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft truncate">
              {caption}
            </p>
          )}
        </div>
        {trailing && (
          <div className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft tnum shrink-0">
            {trailing}
          </div>
        )}
      </div>
      {children}
    </section>
  );
}
