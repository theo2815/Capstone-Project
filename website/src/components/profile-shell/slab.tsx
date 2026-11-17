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
          <span className="font-mono text-[14px] min-[400px]:text-[15px] md:text-[13px] tracking-[0.15em] text-slate-soft tnum">
            {number}
          </span>
          {/* A real h2 so screen readers get page h1 → section h2 → card h3. On
              phones the title may wrap rather than run into the trailing count. */}
          <h2 className="font-mono font-normal uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink min-w-0 md:shrink-0">
            {title}
          </h2>
          {caption && (
            <p className="hidden md:block font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft min-w-0">
              {caption}
            </p>
          )}
        </div>
        {trailing && (
          <div className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft tnum shrink-0">
            {trailing}
          </div>
        )}
      </div>
      {children}
    </section>
  );
}
