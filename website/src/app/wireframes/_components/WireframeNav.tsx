import Link from "next/link";

const items = [
  { slug: "two-doors", label: "01 / Two Doors" },
  { slug: "race-trace", label: "02 / Race Trace" },
  { slug: "channel-switch", label: "03 / Channel Switch" },
  { slug: "dual-sidebar", label: "04 / Dual Sidebar" },
];

export function WireframeNav({ active }: { active: string }) {
  return (
    <div className="sticky top-0 z-40 border-b border-dashed border-neutral-300 bg-[#FAFAF7]/95 backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-6 py-3 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.18em] text-neutral-600">
        <Link href="/wireframes" className="hover:text-neutral-900">
          ← Wireframes hub
        </Link>
        <nav className="flex gap-2">
          {items.map((it) => (
            <Link
              key={it.slug}
              href={`/wireframes/${it.slug}`}
              className={`border px-3 py-1 transition-colors ${
                active === it.slug
                  ? "border-neutral-900 bg-neutral-900 text-white"
                  : "border-dashed border-neutral-400 hover:border-neutral-900"
              }`}
            >
              {it.label}
            </Link>
          ))}
        </nav>
      </div>
    </div>
  );
}

export function WireframeAnnotation({ children }: { children: React.ReactNode }) {
  return (
    <p className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
      {children}
    </p>
  );
}

export function WireframeStamp({ label }: { label: string }) {
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 border border-dashed border-neutral-400 bg-white/80 px-3 py-1 font-[family-name:var(--font-mono)] text-[10px] uppercase tracking-[0.22em] text-neutral-500 backdrop-blur">
      {label}
    </div>
  );
}
