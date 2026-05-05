import { WireframePage } from "../_components/WireframePage";
import { DesktopFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Tag } from "../_components/primitives";

export default function M23() {
  const batches = [
    { name: "Batch-001", range: "06:01 – 06:42", count: 500 },
    { name: "Batch-002", range: "06:42 – 07:18", count: 500 },
    { name: "Batch-003", range: "07:18 – 07:56", count: 500 },
    { name: "Batch-004", range: "07:56 – 08:31", count: 500 },
    { name: "Batch-005", range: "08:31 – 09:09", count: 500 },
    { name: "Batch-019", range: "13:34 – 14:12", count: 141 },
  ];

  return (
    <WireframePage
      module="M2"
      ucId="UC-M2-2.3"
      title="Auto-Sort into Batch Folders"
      tracesTo="SO2.2 · GO2"
      mustShow={[
        "Batch-size input with default",
        "‘Sort to batches’ CTA",
        "In-progress indicator with elapsed time",
        "Folder tree preview after sort",
        "Per-batch photo count",
        "Re-sort affordance after overrides",
      ]}
    >
      <DesktopFrame appLabel="BatchMyPhotos · Sort to Batches">
        <div className="grid grid-cols-12">
          {/* sidebar */}
          <aside className="col-span-3 border-r border-neutral-300 bg-neutral-50 p-3">
            <Caption>Pipeline</Caption>
            <ol className="mt-1 space-y-0.5 font-mono text-[10px] text-neutral-700">
              <li>1 · Sync ✓</li>
              <li>2 · Detect blur ✓</li>
              <li className="text-neutral-900">▸ 3 · Sort to batches</li>
              <li>4 · Upload to QuickPitik</li>
            </ol>

            <div className="mt-4 border-t border-neutral-200 pt-3">
              <Caption>Sort settings</Caption>
              <label className="mt-2 block">
                <span className="font-mono text-[10px] uppercase tracking-wider text-neutral-500">
                  Batch size (A2)
                </span>
                <span className="mt-1 flex items-stretch border border-neutral-700">
                  <span className="w-full px-2 py-1.5 font-mono text-sm text-neutral-900">
                    500
                  </span>
                  <span className="border-l border-neutral-700 px-2 py-1.5 font-mono text-sm">−</span>
                  <span className="border-l border-neutral-700 px-2 py-1.5 font-mono text-sm">+</span>
                </span>
              </label>
              <div className="mt-2 text-[10px] text-neutral-500">
                default · 500 / batch
              </div>

              <div className="mt-3">
                <Caption>Strategy</Caption>
                <div className="mt-1 flex flex-col gap-1 text-[11px] text-neutral-800">
                  <label className="flex items-center gap-2">
                    <span className="h-3 w-3 rounded-full border border-neutral-900 bg-neutral-900" />
                    capture-timestamp asc
                  </label>
                  <label className="flex items-center gap-2 text-neutral-500">
                    <span className="h-3 w-3 rounded-full border border-neutral-400" />
                    photographer-id asc
                  </label>
                </div>
              </div>

              <div className="mt-3 border-t border-neutral-200 pt-3">
                <Caption>Filesystem</Caption>
                <div className="mt-1 flex flex-wrap gap-1">
                  <Tag tone="info">hardlink</Tag>
                  <Tag>fallback · move</Tag>
                </div>
              </div>
            </div>
          </aside>

          {/* main */}
          <main className="col-span-9 p-5">
            <div className="flex items-center justify-between border-b border-neutral-300 pb-3">
              <div>
                <Caption>Step 3 of 4 · Auto-sort</Caption>
                <h3 className="mt-1 font-display text-lg font-semibold">
                  9,141 clean photos → 19 batches
                </h3>
              </div>
              <div className="flex gap-2">
                <Btn>Re-sort (A1)</Btn>
                <Btn primary>Sort to batches</Btn>
              </div>
            </div>

            {/* in-progress */}
            <Box className="mt-4">
              <div className="flex items-center justify-between">
                <Caption>Sorting in progress</Caption>
                <Tag tone="info">hardlinking</Tag>
              </div>
              <div className="mt-2 h-2 w-full overflow-hidden bg-neutral-200">
                <div className="h-full w-[40%] bg-neutral-900" />
              </div>
              <div className="mt-1 flex justify-between font-mono text-[10px] text-neutral-600">
                <span>3,656 / 9,141 placed</span>
                <span>elapsed 14 s · ETA 22 s</span>
              </div>
            </Box>

            {/* folder tree + batch list */}
            <div className="mt-4 grid grid-cols-12 gap-4">
              <Box className="col-span-5 !p-0">
                <div className="border-b border-neutral-200 px-3 py-2">
                  <Caption>Output folder · preview</Caption>
                </div>
                <div className="font-mono text-[11px] text-neutral-800">
                  <div className="border-b border-neutral-100 px-3 py-1.5">
                    📁 D:\QuickPitik\Cebu-2026\
                  </div>
                  {batches.slice(0, 5).map((b) => (
                    <div
                      key={b.name}
                      className="flex items-center justify-between border-b border-neutral-100 px-3 py-1.5 pl-7"
                    >
                      <span>📁 {b.name}/</span>
                      <span className="text-neutral-500">
                        {b.count.toLocaleString()} photos
                      </span>
                    </div>
                  ))}
                  <div className="border-b border-neutral-100 px-3 py-1.5 pl-7 text-neutral-400">
                    … 13 more batches
                  </div>
                  <div className="flex items-center justify-between px-3 py-1.5 pl-7">
                    <span>📁 Batch-019/</span>
                    <span className="text-neutral-500">141 photos</span>
                  </div>
                </div>
              </Box>

              <Box className="col-span-7 !p-0">
                <div className="border-b border-neutral-200 px-3 py-2">
                  <div className="flex items-center justify-between">
                    <Caption>Batch summary</Caption>
                    <span className="font-mono text-[10px] text-neutral-500">
                      19 batches · 9,141 photos
                    </span>
                  </div>
                </div>
                <table className="w-full font-mono text-[11px]">
                  <thead className="bg-neutral-50">
                    <tr>
                      <th className="px-3 py-1.5 text-left text-neutral-500">Batch</th>
                      <th className="px-3 py-1.5 text-left text-neutral-500">Capture range</th>
                      <th className="px-3 py-1.5 text-right text-neutral-500">Photos</th>
                      <th className="px-3 py-1.5 text-left text-neutral-500">State</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-100">
                    {batches.map((b, i) => (
                      <tr key={b.name}>
                        <td className="px-3 py-1.5">{b.name}</td>
                        <td className="px-3 py-1.5">{b.range}</td>
                        <td className="px-3 py-1.5 text-right">{b.count}</td>
                        <td className="px-3 py-1.5">
                          <Tag tone={i < 3 ? "ok" : i < 5 ? "info" : "neutral"}>
                            {i < 3 ? "placed" : i < 5 ? "linking" : "queued"}
                          </Tag>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Box>
            </div>

            {/* exception */}
            <Box className="mt-4 !bg-rose-50 !border-rose-700">
              <div className="flex items-center justify-between">
                <Caption>E1 / E2 · Roll-back state</Caption>
                <Tag tone="err">aborted</Tag>
              </div>
              <div className="mt-1 text-[11px] text-neutral-800">
                If a move fails for space reasons or the output dir is not
                writable, partial moves are reverted; no half-sorted folders
                left on disk.
              </div>
            </Box>

            <div className="mt-4 grid grid-cols-2 gap-2">
              <Annot n={1}>Batch-size input + default 500 (A2 custom-size).</Annot>
              <Annot n={2}>Sort CTA + in-progress bar with elapsed / ETA.</Annot>
              <Annot n={3}>Folder tree preview — Batch-001 .. Batch-N inside event output dir.</Annot>
              <Annot n={4}>Per-batch photo count + capture range (deterministic timestamp partition).</Annot>
              <Annot n={5}>Re-sort (A1) — surfaced after manual overrides from M2.2.</Annot>
              <Annot n={6}>Atomic roll-back on E1/E2 — never leaves a partial sort.</Annot>
            </div>
          </main>
        </div>
      </DesktopFrame>
    </WireframePage>
  );
}
