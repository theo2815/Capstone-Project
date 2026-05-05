import { WireframePage } from "../_components/WireframePage";
import { DesktopFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Tag } from "../_components/primitives";

export default function M21() {
  const events = [
    { id: "EV-118", name: "Cebu City Marathon 2026", date: "12 Apr 2026", state: "active", tone: "ok" as const },
    { id: "EV-119", name: "Mactan Sunrise Run", date: "20 Apr 2026", state: "active", tone: "ok" as const },
    { id: "EV-117", name: "Sinulog 5K", date: "05 Apr 2026", state: "completed", tone: "neutral" as const },
  ];
  return (
    <WireframePage
      module="M2"
      ucId="UC-M2-2.1"
      title="Sync Event Library"
      tracesTo="SO2.3 · GO2"
      mustShow={[
        "Event picker",
        "‘Sync local folder’ CTA",
        "Folder-walk progress with file count",
        "Skipped-files report panel",
        "Total / pending / skipped tally on completion",
        "Re-sync affordance",
      ]}
    >
      <DesktopFrame appLabel="BatchMyPhotos · Sync Library">
        <div className="grid grid-cols-12 gap-0">
          {/* sidebar */}
          <aside className="col-span-3 border-r border-neutral-300 bg-neutral-50 p-3">
            <Caption>Events</Caption>
            <ul className="mt-2 space-y-1">
              {events.map((e, i) => (
                <li
                  key={e.id}
                  className={[
                    "rounded border px-2 py-2 cursor-pointer",
                    i === 0
                      ? "border-neutral-900 bg-white"
                      : "border-transparent hover:bg-white",
                  ].join(" ")}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[10px] text-neutral-500">
                      {e.id}
                    </span>
                    <Tag tone={e.tone}>{e.state}</Tag>
                  </div>
                  <div className="mt-0.5 text-[12px] font-medium text-neutral-900">
                    {e.name}
                  </div>
                  <div className="text-[10px] text-neutral-500">{e.date}</div>
                </li>
              ))}
            </ul>
            <div className="mt-3 border-t border-neutral-200 pt-3">
              <Caption>Pipeline</Caption>
              <ol className="mt-1 space-y-0.5 font-mono text-[10px] text-neutral-700">
                <li className="text-neutral-900">▸ 1 · Sync</li>
                <li>2 · Detect blur</li>
                <li>3 · Sort to batches</li>
                <li>4 · Upload to QuickPitik</li>
              </ol>
            </div>
          </aside>

          {/* main */}
          <main className="col-span-9 p-5">
            <div className="flex items-center justify-between border-b border-neutral-300 pb-3">
              <div>
                <Caption>Step 1 of 4 · Sync</Caption>
                <h3 className="mt-1 font-display text-lg font-semibold">
                  Cebu City Marathon 2026
                </h3>
              </div>
              <div className="flex gap-2">
                <Btn>Re-sync</Btn>
                <Btn primary>Sync local folder</Btn>
              </div>
            </div>

            <div className="grid grid-cols-12 gap-4 pt-4">
              {/* folder picker */}
              <div className="col-span-7 space-y-3">
                <Box>
                  <Caption>Source folder</Caption>
                  <div className="mt-1 font-mono text-[12px] text-neutral-800 truncate">
                    D:\Photos\2026-04-12 Cebu Marathon\
                  </div>
                  <div className="mt-2 flex gap-2 text-[11px]">
                    <Tag>JPEG</Tag>
                    <Tag>CR3 (Canon)</Tag>
                    <Tag>NEF (Nikon)</Tag>
                  </div>
                </Box>

                {/* walk progress */}
                <Box className="!bg-neutral-50">
                  <div className="flex items-center justify-between">
                    <Caption>Walking folder tree…</Caption>
                    <Tag tone="info">EXIF ONLY · NO BYTES</Tag>
                  </div>
                  <div className="mt-2 h-2 w-full overflow-hidden bg-neutral-200">
                    <div className="h-full w-[72%] bg-neutral-900" />
                  </div>
                  <div className="mt-1 flex justify-between font-mono text-[10px] text-neutral-600">
                    <span>10,824 / 15,000 files indexed</span>
                    <span>43 s elapsed · ETA 17 s</span>
                  </div>
                </Box>

                {/* tally on completion */}
                <Box>
                  <Caption>Completion tally (preview)</Caption>
                  <div className="mt-2 grid grid-cols-3 gap-2 text-center">
                    <div>
                      <div className="font-mono text-2xl tabular-nums">14,973</div>
                      <div className="text-[10px] text-neutral-500">indexed</div>
                    </div>
                    <div>
                      <div className="font-mono text-2xl tabular-nums">14,973</div>
                      <div className="text-[10px] text-neutral-500">pending blur check</div>
                    </div>
                    <div>
                      <div className="font-mono text-2xl tabular-nums text-amber-700">27</div>
                      <div className="text-[10px] text-neutral-500">skipped</div>
                    </div>
                  </div>
                </Box>
              </div>

              {/* skipped files report */}
              <div className="col-span-5">
                <Box className="!p-0 !bg-amber-50 !border-amber-700">
                  <div className="flex items-center justify-between border-b border-amber-700 px-3 py-2">
                    <Caption>Skipped files (E1)</Caption>
                    <Tag tone="warn">27</Tag>
                  </div>
                  <ul className="divide-y divide-amber-200 font-mono text-[10px]">
                    {[
                      "IMG_0148.jpg · permission denied",
                      "IMG_0291.cr3 · header corrupt",
                      "IMG_0612.nef · zero bytes",
                      "IMG_1004.jpg · checksum mismatch",
                      "IMG_1221.cr3 · partial file",
                    ].map((s, i) => (
                      <li key={i} className="px-3 py-1.5 text-neutral-800">
                        {s}
                      </li>
                    ))}
                    <li className="px-3 py-1.5 text-neutral-500">… 22 more</li>
                  </ul>
                </Box>

                <div className="mt-4 space-y-2">
                  <Annot n={1}>Event picker on left — only Admin-active events appear; status badge per event.</Annot>
                  <Annot n={2}>Sync CTA + folder picker — supported types are JPEG / CR3 / NEF (configurable).</Annot>
                  <Annot n={3}>Walk progress reads EXIF only — no image bytes loaded (special requirement).</Annot>
                  <Annot n={4}>Tally panel on completion — indexed / pending / skipped, all written to local SQLite.</Annot>
                  <Annot n={5}>Skipped-files report (E1) — file + reason; indexing continues regardless.</Annot>
                </div>
              </div>
            </div>
          </main>
        </div>
      </DesktopFrame>
    </WireframePage>
  );
}
