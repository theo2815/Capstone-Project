import { WireframePage } from "../_components/WireframePage";
import { PhoneFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Tag } from "../_components/primitives";

export default function M13() {
  const items = [
    { id: "IMG_2120", state: "uploaded", tone: "ok" as const, kb: "8.2 MB" },
    { id: "IMG_2121", state: "uploaded", tone: "ok" as const, kb: "8.4 MB" },
    { id: "IMG_2122", state: "uploading 64%", tone: "info" as const, kb: "9.1 MB" },
    { id: "IMG_2123", state: "queued", tone: "neutral" as const, kb: "8.0 MB" },
    { id: "IMG_2124", state: "queued · slow link", tone: "warn" as const, kb: "7.8 MB" },
    { id: "IMG_2125", state: "failed · 401", tone: "err" as const, kb: "8.6 MB" },
  ];

  return (
    <WireframePage
      module="M1"
      ucId="UC-M1-1.3"
      title="Auto-Upload to Cloud"
      tracesTo="SO1.1, SO1.2 · GO1"
      mustShow={[
        "Queue list with per-photo state (queued / uploading / uploaded / failed)",
        "Aggregate progress bar and ETA",
        "Sync-rate ticker (e.g., ‘23 of 145 uploaded’)",
        "Slow-link badge",
        "Failed-uploads filter",
        "Manual retry CTA",
      ]}
    >
      <div className="grid grid-cols-12 gap-8">
        <div className="col-span-12 lg:col-span-7 flex justify-center">
          <PhoneFrame statusLabel="QUICKPITIK · UPLOAD">
            <div className="flex flex-col gap-3 p-4">
              {/* aggregate progress */}
              <Box className="!bg-neutral-900 !text-white !border-neutral-900">
                <div className="flex items-center justify-between">
                  <Caption><span className="text-neutral-400">Sync · live</span></Caption>
                  <Tag tone="ok">UPLOADING</Tag>
                </div>
                <div className="mt-2 font-mono text-2xl tabular-nums">
                  23 / 145 <span className="text-base text-neutral-400">uploaded</span>
                </div>
                <div className="mt-2 h-2 w-full overflow-hidden bg-neutral-700">
                  <div className="h-full w-[16%] bg-emerald-400" />
                </div>
                <div className="mt-2 flex justify-between font-mono text-[10px] text-neutral-300">
                  <span>2.4 MB/s · LTE</span>
                  <span>ETA · 4 m 12 s</span>
                </div>
              </Box>

              {/* filter row */}
              <div className="flex items-center gap-2">
                <Tag>All</Tag>
                <Tag>Queued</Tag>
                <Tag tone="info">Uploading</Tag>
                <Tag tone="err">Failed · 1</Tag>
              </div>

              {/* queue list */}
              <Box className="!p-0 !border-neutral-300">
                {items.map((it, i) => (
                  <div
                    key={it.id}
                    className={[
                      "flex items-center gap-3 px-3 py-2",
                      i !== items.length - 1 ? "border-b border-neutral-200" : "",
                    ].join(" ")}
                  >
                    <div className="h-8 w-10 border border-neutral-300 bg-neutral-100" />
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-[11px] text-neutral-800">
                          {it.id}
                        </span>
                        <span className="font-mono text-[10px] text-neutral-500">
                          {it.kb}
                        </span>
                      </div>
                      <div className="mt-0.5 flex items-center gap-1.5">
                        <Tag tone={it.tone}>{it.state}</Tag>
                        {it.state.startsWith("uploading") && (
                          <div className="h-1 flex-1 overflow-hidden bg-neutral-200">
                            <div className="h-full w-[64%] bg-neutral-900" />
                          </div>
                        )}
                      </div>
                    </div>
                    {it.state.startsWith("failed") && (
                      <Btn small>Retry</Btn>
                    )}
                  </div>
                ))}
              </Box>

              {/* exception row */}
              <Box className="!border-amber-700 !bg-amber-50">
                <div className="flex items-center justify-between">
                  <Caption>Auth refresh in progress (E1)</Caption>
                  <Tag tone="warn">JWT 401</Tag>
                </div>
                <div className="mt-1 text-[11px] text-neutral-800">
                  Pausing uploads · refreshing token · auto-resume on success.
                </div>
              </Box>
            </div>
          </PhoneFrame>
        </div>

        <div className="col-span-12 lg:col-span-5">
          <div className="mb-3"><Caption>Annotations</Caption></div>
          <div className="space-y-3">
            <Annot n={1}>Aggregate progress bar + sync-rate ticker — &quot;23 of 145 uploaded&quot;, ETA derived from rolling throughput.</Annot>
            <Annot n={2}>Per-photo state — queued / uploading / uploaded / failed, with inline progress bar for active transfer.</Annot>
            <Annot n={3}>Slow-link badge (A1) — does not raise an error unless the upload times out.</Annot>
            <Annot n={4}>Failed filter chip — surfaces E1/E2/E3 items; tapping enters the failed-uploads view with manual Retry.</Annot>
            <Annot n={5}>Auth-refresh banner (E1) — JWT 401 → pause queue → refresh → resume.</Annot>
          </div>
        </div>
      </div>
    </WireframePage>
  );
}
