import { WireframePage } from "../_components/WireframePage";
import { BrowserFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Field, Placeholder, Tag } from "../_components/primitives";

export default function M32() {
  const events = [
    { id: "EV-118", name: "Cebu City Marathon 2026", date: "12 Apr 2026", loc: "Cebu City", count: 3204 },
    { id: "EV-119", name: "Mactan Sunrise Run", date: "20 Apr 2026", loc: "Mactan", count: 1180 },
    { id: "EV-120", name: "Talisay 10K Charity Run", date: "27 Apr 2026", loc: "Talisay", count: 940 },
    { id: "EV-121", name: "Bohol Coastal Run", date: "04 May 2026", loc: "Tagbilaran", count: 612 },
    { id: "EV-122", name: "Lapu-Lapu Heritage 5K", date: "11 May 2026", loc: "Lapu-Lapu", count: 480 },
    { id: "EV-123", name: "Argao Sugar Trail Run", date: "18 May 2026", loc: "Argao", count: 220 },
  ];
  return (
    <WireframePage
      module="M3"
      ucId="UC-M3-3.2"
      title="Browse Events"
      tracesTo="(supports GO3)"
      mustShow={[
        "Event card grid (hero image, name, date, location, participant count)",
        "Date / location filters",
        "Name search",
        "Pagination",
        "Empty / error states",
        "Mobile-responsive layout",
      ]}
    >
      <BrowserFrame url="quickpitik.ph/events">
        <div className="grid grid-cols-12">
          {/* nav */}
          <div className="col-span-12 flex items-center justify-between border-b border-neutral-200 px-6 py-3">
            <div className="flex items-center gap-4">
              <span className="font-display text-lg font-semibold">QuickPitik</span>
              <span className="font-mono text-[10px] text-neutral-500">
                events · gallery · about
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Tag tone="info">guest</Tag>
              <Btn small>Log in</Btn>
              <Btn small primary>Sign up</Btn>
            </div>
          </div>

          {/* filters */}
          <aside className="col-span-3 border-r border-neutral-200 p-5">
            <Caption>Filters</Caption>
            <div className="mt-3 space-y-3">
              <Field label="Search by name" placeholder="e.g. Cebu" />
              <Field label="Date from" placeholder="2026-04-01" />
              <Field label="Date to" placeholder="2026-05-31" />
              <div>
                <Caption>Location</Caption>
                <ul className="mt-1 space-y-1 text-[11px] text-neutral-800">
                  {["Cebu City", "Mactan", "Talisay", "Tagbilaran", "Argao"].map((l) => (
                    <li key={l} className="flex items-center gap-2">
                      <span className="inline-block h-3 w-3 border border-neutral-700 bg-white" />
                      {l}
                    </li>
                  ))}
                </ul>
              </div>
              <Btn small>Apply</Btn>
            </div>

            <Box className="mt-5 !bg-amber-50 !border-amber-700">
              <Caption>A1 · empty state</Caption>
              <div className="mt-1 text-[11px] text-neutral-800">
                No active events match your filters.
              </div>
            </Box>

            <Box className="mt-2 !bg-rose-50 !border-rose-700">
              <Caption>E1 · backend unreachable</Caption>
              <div className="mt-1 flex items-center justify-between">
                <span className="text-[11px] text-neutral-800">Try again</span>
                <Btn small>Retry</Btn>
              </div>
            </Box>
          </aside>

          {/* events grid */}
          <main className="col-span-9 p-5">
            <div className="flex items-center justify-between border-b border-neutral-200 pb-3">
              <div>
                <Caption>Active events · 6 shown</Caption>
                <h3 className="mt-1 font-display text-xl font-semibold">
                  Find your race
                </h3>
              </div>
              <div className="font-mono text-[10px] text-neutral-500">
                page 1 of 4 · sort: date asc
              </div>
            </div>

            <div className="mt-4 grid grid-cols-3 gap-3">
              {events.map((e) => (
                <Box key={e.id} className="!p-0 hover:shadow-md transition">
                  <Placeholder label="HERO IMAGE" height="h-28" />
                  <div className="p-3">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[10px] text-neutral-500">
                        {e.id}
                      </span>
                      <Tag tone="ok">active</Tag>
                    </div>
                    <div className="mt-1 text-sm font-semibold leading-tight">
                      {e.name}
                    </div>
                    <div className="mt-0.5 font-mono text-[10px] text-neutral-600">
                      {e.date} · {e.loc}
                    </div>
                    <div className="mt-2 flex items-center justify-between">
                      <span className="font-mono text-[10px] text-neutral-700">
                        {e.count.toLocaleString()} photos
                      </span>
                      <Btn small primary>View →</Btn>
                    </div>
                  </div>
                </Box>
              ))}
            </div>

            {/* pagination */}
            <div className="mt-5 flex items-center justify-center gap-1 font-mono text-[11px]">
              {["‹", "1", "2", "3", "4", "›"].map((p, i) => (
                <span
                  key={i}
                  className={[
                    "h-7 w-7 inline-flex items-center justify-center border",
                    p === "1"
                      ? "border-neutral-900 bg-neutral-900 text-white"
                      : "border-neutral-300 bg-white",
                  ].join(" ")}
                >
                  {p}
                </span>
              ))}
            </div>

            <div className="mt-5 grid grid-cols-3 gap-2">
              <Annot n={1}>Card grid · hero · name · date · location · participant count.</Annot>
              <Annot n={2}>Filters · date range + location + name search.</Annot>
              <Annot n={3}>Pagination · 6 cards / page · client-side or backend cursor.</Annot>
              <Annot n={4}>Empty / error states surfaced in side rail.</Annot>
              <Annot n={5}>Click card → event landing → M3.3 / M3.4 search.</Annot>
              <Annot n={6}>Mobile-responsive · 3-col → 2-col → 1-col stack.</Annot>
            </div>
          </main>
        </div>
      </BrowserFrame>
    </WireframePage>
  );
}
