import Link from "next/link";

export const metadata = {
  title: "Wireframes — Quick Pitik",
};

const wireframes = [
  {
    slug: "two-doors",
    number: "01",
    name: "The Two Doors",
    aesthetic: "Brutalist Split",
    summary:
      "Hard 50/50 split-screen on landing. Two stencil-numbered doors. Click expands one side into a numbered marathon-corral lane.",
    accent: "#D63B1F",
    palette: ["#0E0E0E", "#FAFAF7", "#D63B1F"],
  },
  {
    slug: "race-trace",
    number: "02",
    name: "Race Trace",
    aesthetic: "Track-and-Trace Logistics",
    summary:
      "Two parallel rails running the full page. Nodes light up sequentially. Photographer rail forks by event volume — solves the 3-app problem.",
    accent: "#0F766E",
    palette: ["#0B1220", "#F4F4F0", "#0F766E"],
  },
  {
    slug: "channel-switch",
    number: "03",
    name: "Channel Switch",
    aesthetic: "Refined Broadcast Hybrid",
    summary:
      "TV-style channel selector at top. Pick CH 01 (RUNNER FEED) or CH 02 (PHOTOGRAPHER FEED). Each channel plays a chaptered explainer reel.",
    accent: "#FF4A1C",
    palette: ["#0E0E0E", "#161616", "#FF4A1C"],
  },
  {
    slug: "dual-sidebar",
    number: "04",
    name: "Dual Sidebar",
    aesthetic: "Field Notebook · Variant B",
    summary:
      "Two equal panels open like a hand-bound notebook spread. Click one — its page expands to 80%, the other folds to a 20% closed-edge rail. Marginalia, ruled lines, coffee stains.",
    accent: "#B23A2C",
    palette: ["#EFE5D0", "#9C6B3F", "#B23A2C"],
  },
];

export default function WireframesHub() {
  return (
    <main className="min-h-screen bg-[#FAFAF7] text-neutral-900">
      <div className="border-b border-dashed border-neutral-300">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.18em] text-neutral-500">
          <span>Quick Pitik · Wireframes</span>
          <span>2026-04-30 · Exploratory</span>
        </div>
      </div>

      <header className="mx-auto max-w-6xl px-6 pt-16 pb-12">
        <p className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
          [Pick one to prototype next]
        </p>
        <h1 className="mt-4 font-[family-name:var(--font-display)] text-5xl font-black uppercase leading-[0.92] tracking-tight md:text-7xl">
          Three landing-page directions.
          <br />
          <span className="text-neutral-400">One job: zero confusion.</span>
        </h1>
        <p className="mt-6 max-w-2xl font-[family-name:var(--font-editorial)] text-lg italic text-neutral-700">
          Each wireframe solves the same problem differently — separate runners
          from photographers on first paint, then guide each one through their
          flow without a single &ldquo;wait, am I in the right place?&rdquo;
          moment.
        </p>
      </header>

      <section className="mx-auto max-w-6xl px-6 pb-24">
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {wireframes.map((w) => (
            <Link
              key={w.slug}
              href={`/wireframes/${w.slug}`}
              className="group relative flex flex-col border border-dashed border-neutral-400 bg-white p-6 transition-colors hover:border-neutral-900"
            >
              <div className="flex items-baseline justify-between">
                <span className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
                  WF / {w.number}
                </span>
                <div className="flex gap-1">
                  {w.palette.map((c) => (
                    <span
                      key={c}
                      className="h-3 w-3 border border-neutral-300"
                      style={{ background: c }}
                    />
                  ))}
                </div>
              </div>

              <h2 className="mt-6 font-[family-name:var(--font-display)] text-3xl font-black uppercase leading-[0.95] tracking-tight">
                {w.name}
              </h2>
              <p className="mt-1 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                {w.aesthetic}
              </p>

              <p className="mt-6 font-[family-name:var(--font-editorial)] text-[15px] leading-relaxed text-neutral-700">
                {w.summary}
              </p>

              <div
                className="mt-8 flex items-center justify-between border-t border-dashed border-neutral-300 pt-4 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em]"
                style={{ color: w.accent }}
              >
                <span>Open wireframe</span>
                <span className="transition-transform group-hover:translate-x-1">
                  →
                </span>
              </div>
            </Link>
          ))}
        </div>

        <div className="mt-16 border-t border-dashed border-neutral-300 pt-6 font-[family-name:var(--font-mono)] text-[11px] uppercase leading-relaxed tracking-[0.18em] text-neutral-500">
          <p>
            [Note] These are wireframes, not committed designs. Aesthetic hints
            only. Final visual layer comes after one is picked.
          </p>
          <p className="mt-2">
            [Note] Each wireframe covers: persona selection → photographer flow
            (with volume fork) → runner flow → marketplace tie-in.
          </p>
        </div>
      </section>
    </main>
  );
}
