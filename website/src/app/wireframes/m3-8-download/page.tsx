import { WireframePage } from "../_components/WireframePage";
import { BrowserFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Placeholder, Tag } from "../_components/primitives";

export default function M38() {
  const items = [
    { id: "PH-91204", state: "ready", tone: "ok" as const, size: "9.4 MB" },
    { id: "PH-91207", state: "downloaded", tone: "info" as const, size: "9.1 MB" },
    { id: "PH-91219", state: "preparing (E2)", tone: "warn" as const, size: "—" },
    { id: "PH-91224", state: "ready", tone: "ok" as const, size: "8.7 MB" },
  ];
  return (
    <WireframePage
      module="M3"
      ucId="UC-M3-3.8"
      title="Download Purchased Photos"
      tracesTo="SO3.3 · GO3"
      mustShow={[
        "Post-payment success → downloads transition",
        "Per-photo download button",
        "‘Download all (ZIP)’ CTA",
        "In-progress / completed state per item",
        "‘Preparing’ state for E2",
        "‘My orders’ entry point for re-download",
      ]}
    >
      <BrowserFrame url="quickpitik.ph/orders/ord_91A42/downloads">
        <div className="grid grid-cols-12">
          <div className="col-span-12 flex items-center justify-between border-b border-neutral-200 px-6 py-3">
            <div>
              <span className="font-mono text-[10px] text-neutral-500">
                Order ord_91A42 · paid · 4 photos
              </span>
              <div className="text-sm font-semibold">Downloads</div>
            </div>
            <div className="flex items-center gap-2">
              <Tag tone="info">signed URL TTL · 24 h</Tag>
              <Btn small>My orders</Btn>
            </div>
          </div>

          {/* success transition + zip */}
          <main className="col-span-8 p-6">
            <Box className="!bg-emerald-50 !border-emerald-700">
              <div className="flex items-center justify-between">
                <Caption>Payment received → downloads ready</Caption>
                <Tag tone="ok">SO3.3 · ≤ 5 min from payment</Tag>
              </div>
              <div className="mt-1 text-[11px] text-neutral-800">
                We&apos;re serving clean (un-watermarked) variants from
                quickpitik-prod-clean. Watermark is no longer applied.
              </div>
              <div className="mt-2 flex gap-2">
                <Btn primary>Download all · ZIP</Btn>
                <Btn>Email me the links</Btn>
              </div>
            </Box>

            <div className="mt-4">
              <Caption>Per-photo</Caption>
              <Box className="!p-0 !mt-2">
                <table className="w-full font-mono text-[11px]">
                  <thead className="bg-neutral-50">
                    <tr>
                      <th className="px-3 py-1.5 text-left text-neutral-500">Photo</th>
                      <th className="px-3 py-1.5 text-right text-neutral-500">Size</th>
                      <th className="px-3 py-1.5 text-left text-neutral-500">State</th>
                      <th className="px-3 py-1.5"></th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-100">
                    {items.map((it) => (
                      <tr key={it.id}>
                        <td className="px-3 py-2 flex items-center gap-2">
                          <div className="h-8 w-12 border border-neutral-300 bg-neutral-100" />
                          {it.id}
                        </td>
                        <td className="px-3 py-2 text-right">{it.size}</td>
                        <td className="px-3 py-2">
                          <Tag tone={it.tone}>{it.state}</Tag>
                        </td>
                        <td className="px-3 py-2 text-right">
                          {it.state === "ready" && <Btn small primary>Download</Btn>}
                          {it.state === "downloaded" && <Btn small>Re-download</Btn>}
                          {it.state.startsWith("preparing") && (
                            <span className="font-mono text-[10px] text-neutral-500">
                              polling…
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Box>
            </div>

            {/* download in progress */}
            <Box className="mt-4">
              <div className="flex items-center justify-between">
                <Caption>ZIP bundle · streaming</Caption>
                <Tag tone="info">36.2 MB / 36.4 MB</Tag>
              </div>
              <div className="mt-2 h-2 w-full overflow-hidden bg-neutral-200">
                <div className="h-full w-[99%] bg-neutral-900" />
              </div>
            </Box>

            {/* exceptions */}
            <div className="mt-4 grid grid-cols-2 gap-3">
              <Box className="!bg-amber-50 !border-amber-700">
                <Caption>E1 · signed URL expired</Caption>
                <div className="mt-1 text-[11px] text-neutral-800">
                  Re-fetching transparently — no user action needed.
                </div>
              </Box>
              <Box className="!bg-amber-50 !border-amber-700">
                <Caption>E2 · clean variant missing</Caption>
                <div className="mt-1 text-[11px] text-neutral-800">
                  Per-photo &quot;preparing&quot; state polled until ready.
                </div>
              </Box>
            </div>
          </main>

          {/* sidebar — receipt + my orders */}
          <aside className="col-span-4 border-l border-neutral-200 bg-neutral-50 p-6">
            <Caption>Receipt</Caption>
            <ul className="mt-2 space-y-1 font-mono text-[11px] text-neutral-800">
              <li>order · ord_91A42</li>
              <li>paid · 12 Apr 2026 · 09:48 PHT</li>
              <li>method · Visa ••• 4218</li>
              <li>total · ₱ 846.00 PHP</li>
            </ul>

            <Box className="mt-4">
              <Caption>A1 · Re-download later (My orders)</Caption>
              <div className="mt-1 text-[11px] text-neutral-800">
                Order remains downloadable for 30 days · backend re-issues
                fresh signed URLs on revisit.
              </div>
              <Btn small className="!mt-2">Open my orders</Btn>
            </Box>

            <div className="mt-4 relative aspect-[4/3] border border-neutral-300">
              <Placeholder label="HI-RES PREVIEW (no watermark)" height="h-full" />
              <div className="absolute right-1 top-1">
                <Tag tone="ok">CLEAN</Tag>
              </div>
            </div>

            <div className="mt-4 space-y-2">
              <Annot n={1}>Success transition surfaces ZIP CTA + per-photo links.</Annot>
              <Annot n={2}>Watermark NOT applied to clean variant for paid customers.</Annot>
              <Annot n={3}>Signed URL · 24 h TTL · re-fetched on E1.</Annot>
              <Annot n={4}>My orders entry · 30-day retention default.</Annot>
            </div>
          </aside>
        </div>
      </BrowserFrame>
    </WireframePage>
  );
}
