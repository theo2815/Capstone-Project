import {
  WireframeAnnotation,
  WireframeNav,
  WireframeStamp,
} from "../_components/WireframeNav";

export const metadata = { title: "WF / 02 — Race Trace" };

const ACCENT = "#0F766E";

const runnerNodes = [
  { n: 1, label: "Pick event", state: "done", time: "5s" },
  { n: 2, label: "Selfie or bib", state: "done", time: "10s" },
  { n: 3, label: "AI match", state: "active", time: "2.8s" },
  { n: 4, label: "Buy & download", state: "pending", time: "30s" },
];

const photographerNodesPostFork = [
  { n: 2, label: "Connect / upload", state: "active", time: "—" },
  { n: 3, label: "AI sort + cull", state: "pending", time: "10s" },
  { n: 4, label: "Gallery live", state: "pending", time: "0s" },
];

export default function RaceTraceWireframe() {
  return (
    <main className="min-h-screen bg-[#F4F4F0] text-neutral-900">
      <WireframeNav active="race-trace" />
      <WireframeStamp label="WF · 02 · Race Trace · Logistics" />

      <section className="border-b border-dashed border-neutral-400">
        <div className="mx-auto max-w-6xl px-6 py-6">
          <WireframeAnnotation>
            ▸ Section A — Hero. Two parallel rails kick off here and run the
            full page. Both rails meet at the marketplace at the bottom.
          </WireframeAnnotation>
        </div>

        <div className="mx-auto max-w-6xl px-6 pb-10">
          <h1 className="font-[family-name:var(--font-display)] text-5xl font-black uppercase leading-[0.92] tracking-tight md:text-7xl">
            Track every photo
            <br />
            <span style={{ color: ACCENT }}>from shutter to runner.</span>
          </h1>
          <p className="mt-4 max-w-xl font-[family-name:var(--font-editorial)] text-lg italic text-neutral-700">
            Two rails. One marketplace. You can see exactly where you are.
          </p>
        </div>

        <div className="mx-auto max-w-6xl px-6 pb-12">
          <div className="grid gap-6 md:grid-cols-2">
            <RailHeader
              tag="RAIL · A"
              title="I'm a runner"
              sub="Find me in the gallery"
              stat="Avg search 2.8s"
              cta="Start tracking →"
            />
            <RailHeader
              tag="RAIL · B"
              title="I'm a photographer"
              sub="Get my shoot online"
              stat="10s sort, no blur"
              cta="Start uploading →"
            />
          </div>
        </div>
      </section>

      <section className="border-b border-dashed border-neutral-400 bg-white">
        <div className="mx-auto max-w-6xl px-6 py-12">
          <WireframeAnnotation>
            ▸ Section B — The rails. Status pills update as user progresses.
            Photographer rail forks at node 1 by event volume.
          </WireframeAnnotation>

          <div className="mt-8 grid gap-12 md:grid-cols-2">
            <RailColumn
              tag="RAIL · A · RUNNER"
              nodes={runnerNodes}
              forked={false}
            />
            <PhotographerRail />
          </div>

          <div className="mt-12 border border-dashed border-neutral-400 bg-[#F4F4F0] p-6">
            <WireframeAnnotation>
              ▸ Section C — Confluence. Both rails terminate here.
            </WireframeAnnotation>
            <div className="mt-4 flex flex-col items-center justify-center gap-4 py-8 md:flex-row">
              <div className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
                Rail A ↘
              </div>
              <div
                className="border-2 border-dashed px-12 py-8 text-center"
                style={{ borderColor: ACCENT }}
              >
                <p className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
                  Node X · Confluence
                </p>
                <p
                  className="mt-2 font-[family-name:var(--font-display)] text-4xl font-black uppercase tracking-tight"
                  style={{ color: ACCENT }}
                >
                  Marketplace
                </p>
                <p className="mt-2 font-[family-name:var(--font-editorial)] text-sm italic text-neutral-600">
                  A photo a photographer uploaded becomes a photo a runner
                  buys.
                </p>
              </div>
              <div className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
                ↙ Rail B
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="border-b border-dashed border-neutral-400 bg-[#F4F4F0]">
        <div className="mx-auto max-w-6xl px-6 py-12">
          <WireframeAnnotation>
            ▸ Section D — Persistent &ldquo;trace&rdquo; widget (sticky bottom-right
            during user journey)
          </WireframeAnnotation>
          <div className="mt-4 flex justify-end">
            <div className="border border-dashed border-neutral-400 bg-white p-4 shadow-sm">
              <div className="flex items-center gap-2 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em]">
                <span
                  className="h-2 w-2 rounded-full animate-pulse"
                  style={{ background: ACCENT }}
                />
                <span style={{ color: ACCENT }}>In transit</span>
              </div>
              <p className="mt-2 font-[family-name:var(--font-display)] text-lg font-bold uppercase tracking-tight">
                Node 3 · AI match
              </p>
              <div className="mt-3 flex gap-1">
                {[1, 2, 3, 4].map((n) => (
                  <span
                    key={n}
                    className="h-1.5 w-12 border border-neutral-300"
                    style={{
                      background: n <= 3 ? ACCENT : "transparent",
                    }}
                  />
                ))}
              </div>
              <p className="mt-2 font-[family-name:var(--font-mono)] text-[10px] uppercase tracking-[0.22em] text-neutral-500">
                ETA · 2.8s
              </p>
            </div>
          </div>
          <p className="mt-3 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
            [Why] Users always see where they are and how much is left. Like
            tracking a parcel — anxiety drops, completion rises.
          </p>
        </div>
      </section>
    </main>
  );
}

function RailHeader({
  tag,
  title,
  sub,
  stat,
  cta,
}: {
  tag: string;
  title: string;
  sub: string;
  stat: string;
  cta: string;
}) {
  return (
    <div className="border border-dashed border-neutral-400 bg-white p-6">
      <div className="flex items-baseline justify-between">
        <span
          className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em]"
          style={{ color: ACCENT }}
        >
          {tag}
        </span>
        <span className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
          {stat}
        </span>
      </div>
      <p className="mt-6 font-[family-name:var(--font-display)] text-3xl font-black uppercase tracking-tight">
        {title}
      </p>
      <p className="mt-1 font-[family-name:var(--font-editorial)] text-base italic text-neutral-600">
        {sub}
      </p>
      <button
        className="mt-6 border-2 px-4 py-2 font-[family-name:var(--font-mono)] text-[12px] uppercase tracking-[0.22em] transition-colors"
        style={{ borderColor: ACCENT, color: ACCENT }}
      >
        {cta}
      </button>
    </div>
  );
}

function RailColumn({
  tag,
  nodes,
}: {
  tag: string;
  nodes: { n: number; label: string; state: string; time: string }[];
  forked?: boolean;
}) {
  return (
    <div>
      <span
        className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em]"
        style={{ color: ACCENT }}
      >
        {tag}
      </span>
      <ol className="relative mt-4 space-y-0 border-l-2 border-dashed border-neutral-300 pl-6">
        {nodes.map((node) => (
          <RailNode key={node.n} {...node} />
        ))}
      </ol>
    </div>
  );
}

function RailNode({
  n,
  label,
  state,
  time,
}: {
  n: number;
  label: string;
  state: string;
  time: string;
}) {
  const stateStyle =
    state === "done"
      ? { bg: ACCENT, fg: "white", label: "Done" }
      : state === "active"
      ? { bg: "#fff", fg: ACCENT, label: "Active" }
      : { bg: "#fff", fg: "#999", label: "Pending" };

  return (
    <li className="relative pb-8 last:pb-0">
      <span
        className="absolute -left-[34px] top-0 flex h-6 w-6 items-center justify-center border-2 font-[family-name:var(--font-mono)] text-[10px] font-bold"
        style={{
          background: stateStyle.bg,
          color: stateStyle.fg,
          borderColor: ACCENT,
        }}
      >
        {n}
      </span>
      <div className="border border-dashed border-neutral-300 bg-white p-4">
        <div className="flex items-baseline justify-between gap-2">
          <p className="font-[family-name:var(--font-display)] text-xl font-bold uppercase tracking-tight">
            Node {n} · {label}
          </p>
          <span className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
            {time}
          </span>
        </div>
        <div className="mt-2 flex gap-2">
          <span
            className="border px-2 py-0.5 font-[family-name:var(--font-mono)] text-[10px] uppercase tracking-[0.22em]"
            style={{
              borderColor: ACCENT,
              color: stateStyle.fg === "white" ? "#fff" : ACCENT,
              background: state === "done" ? ACCENT : "transparent",
            }}
          >
            {stateStyle.label}
          </span>
        </div>
      </div>
    </li>
  );
}

function PhotographerRail() {
  return (
    <div>
      <span
        className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em]"
        style={{ color: ACCENT }}
      >
        RAIL · B · PHOTOGRAPHER
      </span>

      <ol className="relative mt-4 border-l-2 border-dashed border-neutral-300 pl-6">
        <li className="relative pb-8">
          <span
            className="absolute -left-[34px] top-0 flex h-6 w-6 items-center justify-center border-2 font-[family-name:var(--font-mono)] text-[10px] font-bold"
            style={{
              background: ACCENT,
              color: "white",
              borderColor: ACCENT,
            }}
          >
            1
          </span>
          <div className="border border-dashed border-neutral-300 bg-white p-4">
            <p className="font-[family-name:var(--font-display)] text-xl font-bold uppercase tracking-tight">
              Node 1 · Pick event + volume
            </p>
            <p className="mt-1 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.18em] text-neutral-500">
              [Fork point — volume question routes to the right tool]
            </p>

            <div className="mt-4 grid gap-2 md:grid-cols-3">
              {[
                { label: "< 500 photos", route: "Web upload" },
                { label: "500 – 10k", route: "Mobile app + tether" },
                { label: "10k+", route: "Desktop batch tool" },
              ].map((opt) => (
                <div
                  key={opt.label}
                  className="border border-dashed border-neutral-400 p-3"
                >
                  <p className="font-[family-name:var(--font-mono)] text-[10px] uppercase tracking-[0.22em] text-neutral-500">
                    Volume
                  </p>
                  <p className="mt-1 font-[family-name:var(--font-display)] text-base font-bold uppercase tracking-tight">
                    {opt.label}
                  </p>
                  <p
                    className="mt-2 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.18em]"
                    style={{ color: ACCENT }}
                  >
                    → {opt.route}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </li>

        {photographerNodesPostFork.map((node) => (
          <RailNode key={node.n} {...node} />
        ))}
      </ol>
    </div>
  );
}
