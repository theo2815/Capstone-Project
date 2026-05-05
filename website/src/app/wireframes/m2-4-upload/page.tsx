import { WireframePage } from "../_components/WireframePage";
import { DesktopFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Tag } from "../_components/primitives";

export default function M24() {
  const rows = [
    { name: "Batch-001", count: 500, state: "uploaded", tone: "ok" as const, pct: 100 },
    { name: "Batch-002", count: 500, state: "uploaded", tone: "ok" as const, pct: 100 },
    { name: "Batch-003", count: 500, state: "uploading", tone: "info" as const, pct: 62 },
    { name: "Batch-004", count: 500, state: "queued", tone: "neutral" as const, pct: 0 },
    { name: "Batch-005", count: 500, state: "queued", tone: "neutral" as const, pct: 0 },
    { name: "Batch-006", count: 500, state: "failed · 4 photos", tone: "err" as const, pct: 99 },
  ];

  return (
    <WireframePage
      module="M2"
      ucId="UC-M2-2.4"
      title="Upload Sorted Batch to Backend"
      tracesTo="SO2.3 · GO2"
      mustShow={[
        "Batch selector with checkboxes",
        "Concurrency / throttle setting",
        "Aggregate progress bar",
        "Per-photo state (queued / uploading / finalised / failed)",
        "Failed-upload retry panel",
        "Resume affordance for partial uploads",
      ]}
    >
      <DesktopFrame appLabel="BatchMyPhotos · Upload to QuickPitik">
        <div className="grid grid-cols-12">
          {/* sidebar */}
          <aside className="col-span-3 border-r border-neutral-300 bg-neutral-50 p-3">
            <Caption>Pipeline</Caption>
            <ol className="mt-1 space-y-0.5 font-mono text-[10px] text-neutral-700">
              <li>1 · Sync ✓</li>
              <li>2 · Detect blur ✓</li>
              <li>3 · Sort to batches ✓</li>
              <li className="text-neutral-900">▸ 4 · Upload to QuickPitik</li>
            </ol>

            <div className="mt-4 border-t border-neutral-200 pt-3">
              <Caption>Concurrency</Caption>
              <input
                type="range"
                min={1}
                max={8}
                defaultValue={4}
                className="mt-2 w-full accent-neutral-900"
                readOnly
              />
              <div className="flex justify-between font-mono text-[10px] text-neutral-500">
                <span>1</span>
                <span className="text-neutral-900">4 streams</span>
                <span>8</span>
              </div>
            </div>

            <div className="mt-4 border-t border-neutral-200 pt-3">
              <Caption>Backend</Caption>
              <ul className="mt-1 space-y-0.5 font-mono text-[10px] text-neutral-700">
                <li>POST /upload-init-batch</li>
                <li>PUT signed S3 URL</li>
                <li>POST /finalize · per photo</li>
              </ul>
              <div className="mt-2 flex flex-wrap gap-1">
                <Tag tone="info">HTTPS · TLS 1.3</Tag>
                <Tag tone="info">JWT</Tag>
              </div>
            </div>
          </aside>

          {/* main */}
          <main className="col-span-9 p-5">
            <div className="flex items-center justify-between border-b border-neutral-300 pb-3">
              <div>
                <Caption>Step 4 of 4 · Upload</Caption>
                <h3 className="mt-1 font-display text-lg font-semibold">
                  Upload sorted batches → quickpitik.ph
                </h3>
              </div>
              <div className="flex gap-2">
                <Btn>Resume (A1)</Btn>
                <Btn primary>Upload to QuickPitik</Btn>
              </div>
            </div>

            {/* aggregate */}
            <Box className="mt-4 !bg-neutral-900 !text-white !border-neutral-900">
              <div className="flex items-center justify-between">
                <Caption><span className="text-neutral-400">Aggregate · 19 batches</span></Caption>
                <Tag tone="ok">12 / 19 batches done</Tag>
              </div>
              <div className="mt-2 h-2 w-full overflow-hidden bg-neutral-700">
                <div className="h-full w-[63%] bg-emerald-400" />
              </div>
              <div className="mt-2 flex justify-between font-mono text-[10px] text-neutral-300">
                <span>5,776 / 9,141 photos · 12.3 MB/s</span>
                <span>ETA 7 m 32 s · 4 streams</span>
              </div>
            </Box>

            {/* batch table */}
            <Box className="mt-4 !p-0">
              <div className="border-b border-neutral-200 px-3 py-2 flex items-center justify-between">
                <Caption>Batches</Caption>
                <span className="font-mono text-[10px] text-neutral-500">
                  ☑ select · 19 of 19 selected
                </span>
              </div>
              <table className="w-full font-mono text-[11px]">
                <thead className="bg-neutral-50">
                  <tr>
                    <th className="w-8 px-3 py-1.5"></th>
                    <th className="px-3 py-1.5 text-left text-neutral-500">Batch</th>
                    <th className="px-3 py-1.5 text-right text-neutral-500">Photos</th>
                    <th className="px-3 py-1.5 text-left text-neutral-500">Progress</th>
                    <th className="px-3 py-1.5 text-left text-neutral-500">State</th>
                    <th className="px-3 py-1.5"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-100">
                  {rows.map((r) => (
                    <tr key={r.name}>
                      <td className="px-3 py-1.5">
                        <span className="inline-block h-3 w-3 border border-neutral-700 bg-neutral-900" />
                      </td>
                      <td className="px-3 py-1.5">{r.name}</td>
                      <td className="px-3 py-1.5 text-right">{r.count}</td>
                      <td className="px-3 py-1.5">
                        <div className="h-1.5 w-32 overflow-hidden bg-neutral-200">
                          <div
                            className={[
                              "h-full",
                              r.tone === "err"
                                ? "bg-rose-700"
                                : r.tone === "ok"
                                ? "bg-emerald-600"
                                : r.tone === "info"
                                ? "bg-neutral-900"
                                : "bg-neutral-300",
                            ].join(" ")}
                            style={{ width: `${r.pct}%` }}
                          />
                        </div>
                      </td>
                      <td className="px-3 py-1.5">
                        <Tag tone={r.tone}>{r.state}</Tag>
                      </td>
                      <td className="px-3 py-1.5 text-right">
                        {r.tone === "err" && <Btn small>Retry</Btn>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Box>

            {/* failed panel + exceptions */}
            <div className="mt-4 grid grid-cols-2 gap-3">
              <Box className="!bg-rose-50 !border-rose-700">
                <div className="flex items-center justify-between">
                  <Caption>Failed uploads (E2)</Caption>
                  <Tag tone="err">4 photos</Tag>
                </div>
                <ul className="mt-1 font-mono text-[10px] text-neutral-800 space-y-0.5">
                  <li>IMG_3029 · S3 timeout · back-off 8 s</li>
                  <li>IMG_3041 · 5xx · retry 3 / 5</li>
                  <li>IMG_3088 · network reset</li>
                  <li>IMG_3090 · partial · resumable</li>
                </ul>
                <div className="mt-2 flex gap-2">
                  <Btn small primary>Retry all</Btn>
                  <Btn small>Skip</Btn>
                </div>
              </Box>
              <Box className="!bg-amber-50 !border-amber-700">
                <Caption>E3 · backend rejects manifest</Caption>
                <div className="mt-1 text-[11px] text-neutral-800">
                  4xx on /upload-init-batch (event not configured · scope
                  mismatch). Batch aborted with precise error displayed.
                </div>
              </Box>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-2">
              <Annot n={1}>Batch selector — checkboxes default to all unuploaded.</Annot>
              <Annot n={2}>Concurrency slider — default 4 streams (special requirement).</Annot>
              <Annot n={3}>Aggregate bar + per-batch progress + state badges.</Annot>
              <Annot n={4}>Failed-upload panel (E2) — per-photo retry; honours backend back-off.</Annot>
              <Annot n={5}>Resume (A1) — picks up from first non-uploaded photo without re-uploading completed ones.</Annot>
            </div>
          </main>
        </div>
      </DesktopFrame>
    </WireframePage>
  );
}
