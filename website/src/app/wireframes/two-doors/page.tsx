import {
  WireframeAnnotation,
  WireframeNav,
  WireframeStamp,
} from "../_components/WireframeNav";

export const metadata = { title: "WF / 01 — Two Doors" };

const photographerSteps = [
  { n: "01", title: "Pick your event", note: "Or create a new gallery" },
  { n: "02", title: "Choose your volume", note: "< 500 / 500–10k / 10k+" },
  { n: "03", title: "Connect or upload", note: "Mobile / web / desktop" },
  { n: "04", title: "Your gallery is live", note: "Auto-sorted, blur-culled" },
];

const runnerSteps = [
  { n: "01", title: "Find your race", note: "Filter by date, city" },
  { n: "02", title: "Selfie or bib?", note: "Pick one — both work" },
  { n: "03", title: "Match in seconds", note: "AI-powered search" },
  { n: "04", title: "Buy and download", note: "GCash / card / Maya" },
];

export default function TwoDoorsWireframe() {
  return (
    <main className="min-h-screen bg-[#FAFAF7] text-neutral-900">
      <WireframeNav active="two-doors" />
      <WireframeStamp label="WF · 01 · Two Doors · Brutalist" />

      <section className="border-b border-dashed border-neutral-400">
        <div className="mx-auto max-w-6xl px-6 py-6">
          <WireframeAnnotation>
            ▸ Section A — Hero / persona selection (above the fold)
          </WireframeAnnotation>
        </div>

        <div className="mx-auto grid max-w-6xl grid-cols-2 gap-0 border-y border-neutral-900">
          <Door
            number="01"
            label="I'm a runner"
            sub="Find my photos"
            stat="2.8s avg search"
            accent="#0E0E0E"
          />
          <Door
            number="02"
            label="I'm a photographer"
            sub="Sell my photos"
            stat="10s sort, no blur"
            accent="#D63B1F"
          />
        </div>

        <div className="mx-auto max-w-6xl px-6 py-6">
          <WireframeAnnotation>
            [Interaction] Hover one door → other dims to 35%. Click → full-bleed
            slide takes over the screen with a numbered marathon-corral lane.
            Stencil number stays visible top-left as anchor.
          </WireframeAnnotation>
        </div>
      </section>

      <section className="border-b border-dashed border-neutral-400 bg-neutral-900 text-neutral-100">
        <div className="mx-auto max-w-6xl px-6 py-12">
          <div className="flex items-baseline justify-between border-b border-dashed border-neutral-700 pb-4">
            <span className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-400">
              ▸ Section B — Door 02 expanded · Photographer lane
            </span>
            <span
              className="font-[family-name:var(--font-display)] text-[80px] font-black leading-none tracking-tight"
              style={{ color: "#D63B1F" }}
            >
              02
            </span>
          </div>

          <h2 className="mt-8 max-w-3xl font-[family-name:var(--font-display)] text-5xl font-black uppercase leading-[0.92] tracking-tight">
            Sell every keeper.
            <br />
            <span className="text-neutral-500">Cull every blur.</span>
          </h2>

          <div className="mt-12 border border-dashed border-neutral-700 bg-neutral-800 p-6">
            <span className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-400">
              [The 3-app problem, solved with one question]
            </span>
            <p className="mt-3 font-[family-name:var(--font-display)] text-2xl font-bold uppercase tracking-tight">
              How big is the event you&apos;re shooting?
            </p>

            <div className="mt-6 grid gap-3 md:grid-cols-3">
              {[
                {
                  label: "Under 500",
                  app: "Upload via web",
                  desc: "Drag-drop in your browser. No install.",
                },
                {
                  label: "500 — 10k",
                  app: "Mobile app",
                  desc: "Tether your camera. Auto-uploads as you shoot.",
                },
                {
                  label: "10k+",
                  app: "Desktop app",
                  desc: "Auto-batches into 500s for stable upload.",
                },
              ].map((opt) => (
                <div
                  key={opt.label}
                  className="border border-dashed border-neutral-600 p-4 transition-colors hover:border-[#D63B1F]"
                >
                  <span className="font-[family-name:var(--font-mono)] text-[10px] uppercase tracking-[0.22em] text-neutral-400">
                    Photos
                  </span>
                  <p className="mt-1 font-[family-name:var(--font-display)] text-3xl font-black tracking-tight">
                    {opt.label}
                  </p>
                  <p
                    className="mt-3 font-[family-name:var(--font-mono)] text-[12px] uppercase tracking-[0.18em]"
                    style={{ color: "#FF6B3D" }}
                  >
                    → {opt.app}
                  </p>
                  <p className="mt-2 font-[family-name:var(--font-editorial)] text-sm italic text-neutral-300">
                    {opt.desc}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-10">
            <span className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-400">
              [The lane — numbered corral down the page]
            </span>
            <ol className="mt-4 grid gap-0 md:grid-cols-4">
              {photographerSteps.map((s, i) => (
                <li
                  key={s.n}
                  className={`relative border border-dashed border-neutral-700 p-5 ${
                    i === 0 ? "" : "md:border-l-0"
                  }`}
                >
                  <span
                    className="font-[family-name:var(--font-display)] text-7xl font-black leading-none tracking-tight"
                    style={{ color: "#D63B1F" }}
                  >
                    {s.n}
                  </span>
                  <p className="mt-3 font-[family-name:var(--font-display)] text-lg font-bold uppercase tracking-tight">
                    {s.title}
                  </p>
                  <p className="mt-1 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.18em] text-neutral-400">
                    {s.note}
                  </p>
                </li>
              ))}
            </ol>
          </div>
        </div>
      </section>

      <section className="border-b border-dashed border-neutral-400 bg-[#FAFAF7]">
        <div className="mx-auto max-w-6xl px-6 py-12">
          <div className="flex items-baseline justify-between border-b border-dashed border-neutral-300 pb-4">
            <span className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
              ▸ Section C — Door 01 expanded · Runner lane
            </span>
            <span className="font-[family-name:var(--font-display)] text-[80px] font-black leading-none tracking-tight text-neutral-900">
              01
            </span>
          </div>

          <h2 className="mt-8 max-w-3xl font-[family-name:var(--font-display)] text-5xl font-black uppercase leading-[0.92] tracking-tight">
            Find your finish.
            <br />
            <span className="text-neutral-400">Faster than a split.</span>
          </h2>

          <div className="mt-10">
            <span className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
              [The lane — numbered corral down the page]
            </span>
            <ol className="mt-4 grid gap-0 md:grid-cols-4">
              {runnerSteps.map((s, i) => (
                <li
                  key={s.n}
                  className={`relative border border-dashed border-neutral-400 bg-white p-5 ${
                    i === 0 ? "" : "md:border-l-0"
                  }`}
                >
                  <span className="font-[family-name:var(--font-display)] text-7xl font-black leading-none tracking-tight text-neutral-900">
                    {s.n}
                  </span>
                  <p className="mt-3 font-[family-name:var(--font-display)] text-lg font-bold uppercase tracking-tight">
                    {s.title}
                  </p>
                  <p className="mt-1 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                    {s.note}
                  </p>
                </li>
              ))}
            </ol>
          </div>

          <div className="mt-8 border border-dashed border-neutral-400 bg-white p-5">
            <span className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
              [Privacy reassurance — sits next to selfie input, not in tooltip]
            </span>
            <p className="mt-2 font-[family-name:var(--font-editorial)] text-base italic text-neutral-700">
              &ldquo;Selfie used once for matching. Never stored. Never shared.&rdquo;
            </p>
          </div>
        </div>
      </section>

      <section>
        <div className="mx-auto max-w-6xl px-6 py-12">
          <WireframeAnnotation>
            ▸ Section D — Persistent persona indicator (top nav, all subsequent
            pages)
          </WireframeAnnotation>
          <div className="mt-3 flex items-center gap-4 border border-dashed border-neutral-400 bg-white p-3">
            <span className="border border-neutral-900 bg-neutral-900 px-3 py-1 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-white">
              👟 Runner
            </span>
            <span className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.18em] text-neutral-500">
              Switch to photographer →
            </span>
          </div>
          <p className="mt-3 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
            [Why] Removes the &ldquo;am I in the right place?&rdquo; anxiety
            that costs you signups.
          </p>
        </div>
      </section>
    </main>
  );
}

function Door({
  number,
  label,
  sub,
  stat,
  accent,
}: {
  number: string;
  label: string;
  sub: string;
  stat: string;
  accent: string;
}) {
  const dark = accent === "#0E0E0E";
  return (
    <div
      className={`group relative flex aspect-[4/5] flex-col justify-between overflow-hidden border-r border-neutral-900 p-8 transition-colors last:border-r-0 ${
        dark
          ? "bg-neutral-900 text-neutral-100 hover:bg-black"
          : "bg-[#FAFAF7] hover:bg-white"
      }`}
    >
      <div className="flex items-baseline justify-between">
        <span
          className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em]"
          style={{ color: dark ? "#999" : "#666" }}
        >
          DOOR {number}
        </span>
        <span
          className="h-2 w-2 rounded-full"
          style={{ background: accent }}
        />
      </div>

      <div
        className="absolute left-6 top-1/2 -translate-y-1/2 font-[family-name:var(--font-display)] font-black leading-[0.78] tracking-tight"
        style={{
          fontSize: "clamp(120px, 22vw, 280px)",
          color: dark ? "#161616" : "#EAE7DF",
        }}
      >
        {number}
      </div>

      <div className="relative z-10">
        <p
          className="font-[family-name:var(--font-display)] text-3xl font-black uppercase leading-[0.95] tracking-tight md:text-5xl"
          style={{ color: accent }}
        >
          {label}
        </p>
        <p className="mt-2 font-[family-name:var(--font-editorial)] text-lg italic">
          {sub}
        </p>
        <p className="mt-6 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
          {stat}
        </p>
      </div>
    </div>
  );
}
