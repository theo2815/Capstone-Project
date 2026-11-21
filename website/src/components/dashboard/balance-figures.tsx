import { cn, formatPrice } from "@/lib/utils";

// The payout balance as three side-by-side figures (Billing hero + Overview
// glance). A lone giant "₱0.00" was the loudest thing on the page while it
// said nothing; three figures say where the money is, and a zero stays quiet.
// Fresh marks Available only when it can be requested (one fresh per viewport).
export function BalanceFigures({
  available,
  clearing,
  reserved,
  ready,
}: {
  available: number;
  clearing: number;
  reserved: number;
  /** Available meets the minimum and nothing blocks a request. */
  ready: boolean;
}) {
  const figures = [
    // The sentence under the figures explains Available; the other two need a word each.
    { label: "Available to request", value: available, caption: null, lead: true },
    { label: "Clearing", value: clearing, caption: "Withdrawable soon", lead: false },
    { label: "In a request", value: reserved, caption: "Being paid out", lead: false },
  ];
  return (
    <dl className="grid grid-cols-2 xl:grid-cols-3 gap-x-4 xl:gap-x-6 gap-y-6">
      {figures.map((f) => (
        <div
          key={f.label}
          className={cn(
            // Three across only from xl: beside the rail, the column is too narrow below 1280px for six-figure amounts.
            "min-w-0 xl:border-l xl:border-line xl:pl-6 xl:first:border-0 xl:first:pl-0",
            f.lead && "col-span-2 xl:col-span-1",
          )}
        >
          <dt className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
            {f.label}
          </dt>
          <dd
            className={cn(
              "font-display font-semibold tracking-tight tnum mt-3 leading-none",
              f.lead ? "text-4xl xl:text-5xl" : "text-2xl xl:text-4xl",
              f.value === 0 ? "text-slate-soft" : f.lead && ready ? "text-fresh" : "text-ink",
            )}
          >
            {formatPrice(f.value)}
          </dd>
          {f.caption && <dd className="font-sans text-sm text-slate mt-3">{f.caption}</dd>}
        </div>
      ))}
    </dl>
  );
}
