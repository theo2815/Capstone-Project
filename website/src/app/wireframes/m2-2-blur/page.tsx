import { WireframePage } from "../_components/WireframePage";
import { DesktopFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Placeholder, Tag } from "../_components/primitives";

export default function M22() {
  const grid = [
    { score: 0.18, state: "culled" },
    { score: 0.62, state: "clean" },
    { score: 0.91, state: "clean" },
    { score: 0.21, state: "culled" },
    { score: 0.84, state: "clean" },
    { score: 0.07, state: "culled" },
    { score: 0.55, state: "clean" },
    { score: 0.43, state: "clean" },
  ];
  return (
    <WireframePage
      module="M2"
      ucId="UC-M2-2.2"
      title="Run Blur Detection"
      tracesTo="SO2.1 · GO2"
      mustShow={[
        "‘Detect blur’ CTA",
        "In-progress state with job ID and live percentage",
        "Classification results grid (clean / culled tabs)",
        "Per-photo blur score badge",
        "Threshold slider for re-classification",
        "Manual-override toggle",
        "Failed-scoring panel",
      ]}
    >
      <DesktopFrame appLabel="BatchMyPhotos · Detect Blur">
        <div className="grid grid-cols-12">
          {/* sidebar */}
          <aside className="col-span-3 border-r border-neutral-300 bg-neutral-50 p-3">
            <Caption>Pipeline</Caption>
            <ol className="mt-1 space-y-0.5 font-mono text-[10px] text-neutral-700">
              <li>1 · Sync ✓</li>
              <li className="text-neutral-900">▸ 2 · Detect blur</li>
              <li>3 · Sort to batches</li>
              <li>4 · Upload to QuickPitik</li>
            </ol>

            <div className="mt-4 border-t border-neutral-200 pt-3">
              <Caption>ai-api · scopes</Caption>
              <div className="mt-1 flex flex-wrap gap-1">
                <Tag tone="info">blur:read</Tag>
                <Tag tone="info">jobs:read</Tag>
              </div>
            </div>

            <div className="mt-4 border-t border-neutral-200 pt-3">
              <Caption>Threshold (A1)</Caption>
              <div className="mt-2">
                <input
                  type="range"
                  min={0}
                  max={100}
                  defaultValue={50}
                  className="w-full accent-neutral-900"
                  readOnly
                />
                <div className="flex justify-between font-mono text-[10px] text-neutral-500">
                  <span>0.00</span>
                  <span className="text-neutral-900">0.50</span>
                  <span>1.00</span>
                </div>
              </div>
              <div className="mt-2 text-[11px] text-neutral-700">
                <span className="font-mono">is_blurry = score &lt; 0.50</span>
              </div>
            </div>
          </aside>

          {/* main */}
          <main className="col-span-9 p-5">
            <div className="flex items-center justify-between border-b border-neutral-300 pb-3">
              <div>
                <Caption>Step 2 of 4 · Detect blur</Caption>
                <h3 className="mt-1 font-display text-lg font-semibold">
                  14,973 photos · pending blur check
                </h3>
              </div>
              <div className="flex gap-2">
                <Btn>Re-classify</Btn>
                <Btn primary>Detect blur</Btn>
              </div>
            </div>

            {/* in-progress */}
            <Box className="mt-4 !bg-neutral-900 !text-white !border-neutral-900">
              <div className="flex items-center justify-between">
                <Caption>
                  <span className="text-neutral-400">
                    Job · ai-api /v1/blur/batch
                  </span>
                </Caption>
                <Tag tone="info">job_8f3a · running</Tag>
              </div>
              <div className="mt-2 h-2 w-full overflow-hidden bg-neutral-700">
                <div className="h-full w-[63%] bg-emerald-400" />
              </div>
              <div className="mt-2 flex justify-between font-mono text-[10px] text-neutral-300">
                <span>9,433 / 14,973 scored · 63 %</span>
                <span>poll every 2 s · ETA 1 m 20 s</span>
              </div>
            </Box>

            {/* result tabs */}
            <div className="mt-4 flex items-center gap-2">
              <Tag tone="ok">Clean · 9,141</Tag>
              <Tag tone="err">Culled · 5,832</Tag>
              <Tag tone="warn">Failed scoring · 12</Tag>
              <span className="ml-auto text-[11px] text-neutral-600">
                cull rate · 38.9 %
              </span>
            </div>

            <div className="mt-3 grid grid-cols-4 gap-2">
              {grid.map((g, i) => (
                <div
                  key={i}
                  className={[
                    "relative aspect-[4/3] border bg-neutral-100",
                    g.state === "culled"
                      ? "border-rose-700"
                      : "border-neutral-300",
                  ].join(" ")}
                >
                  <Placeholder label={`IMG_24${i.toString().padStart(2, "0")}`} height="h-full" />
                  <div className="absolute left-1 top-1">
                    <Tag tone={g.state === "culled" ? "err" : "ok"}>
                      {g.state}
                    </Tag>
                  </div>
                  <div className="absolute right-1 top-1">
                    <Tag>score {g.score.toFixed(2)}</Tag>
                  </div>
                  <div className="absolute right-1 bottom-1">
                    <span className="inline-flex items-center gap-1 border border-neutral-900 bg-white px-1 py-0.5 font-mono text-[9px]">
                      <span className="h-2 w-2 rounded-full border border-neutral-900 bg-white" />
                      override
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {/* failed scoring panel */}
            <Box className="mt-4 !bg-amber-50 !border-amber-700">
              <div className="flex items-center justify-between">
                <Caption>Failed scoring (E2)</Caption>
                <Tag tone="warn">12 photos · pending_blur_check</Tag>
              </div>
              <div className="mt-1 text-[11px] text-neutral-800">
                ai-api returned per-photo errors · these remain unscored and are
                surfaced for retry. No DB state change for E1 (ai-api unreachable).
              </div>
            </Box>

            <div className="mt-4 grid grid-cols-2 gap-2">
              <Annot n={1}>Detect-blur CTA → POST /v1/blur/batch with desktop&apos;s blur:read + jobs:read scopes.</Annot>
              <Annot n={2}>Live job ticker — job_id + percentage, polled every 2 s.</Annot>
              <Annot n={3}>Per-photo score badge + culled/clean tabs (MSS step 7-8).</Annot>
              <Annot n={4}>Threshold slider (A1) — re-applies comparison locally without re-calling ai-api.</Annot>
              <Annot n={5}>Manual override (A2) — toggles is_blurry; honoured by M2.3.</Annot>
              <Annot n={6}>Failed-scoring panel (E2) — surfaces per-photo failures.</Annot>
            </div>
          </main>
        </div>
      </DesktopFrame>
    </WireframePage>
  );
}
