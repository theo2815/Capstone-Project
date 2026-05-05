import { WireframePage } from "../_components/WireframePage";
import { PhoneFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Placeholder, Tag } from "../_components/primitives";

export default function M12() {
  const frames = [
    { state: "uploaded", time: "9:41:02", tone: "ok" as const, label: "✓" },
    { state: "uploaded", time: "9:41:03", tone: "ok" as const, label: "✓" },
    { state: "uploading", time: "9:41:04", tone: "info" as const, label: "↑" },
    { state: "queued", time: "9:41:05", tone: "neutral" as const, label: "·" },
    { state: "queued", time: "9:41:06", tone: "neutral" as const, label: "·" },
    { state: "failed", time: "9:41:06", tone: "err" as const, label: "!" },
  ];
  return (
    <WireframePage
      module="M1"
      ucId="UC-M1-1.2"
      title="Capture & Queue Photo for Upload"
      tracesTo="SO1.1 · GO1"
      mustShow={[
        "Live capture counter",
        "Queue-depth counter",
        "Per-frame thumbnail strip with uploading / queued / failed badges",
        "Storage-low warning state",
        "Per-photo retry / view-error affordance",
      ]}
    >
      <div className="grid grid-cols-12 gap-8">
        <div className="col-span-12 lg:col-span-7 flex justify-center">
          <PhoneFrame statusLabel="QUICKPITIK · CAPTURE">
            <div className="flex flex-col gap-3 p-4">
              {/* counters */}
              <div className="grid grid-cols-2 gap-2">
                <Box className="!bg-neutral-900 !text-white !border-neutral-900">
                  <Caption><span className="text-neutral-400">Captured</span></Caption>
                  <div className="mt-1 font-mono text-3xl tabular-nums">2,148</div>
                </Box>
                <Box>
                  <Caption>Queue depth</Caption>
                  <div className="mt-1 font-mono text-3xl tabular-nums">37</div>
                  <div className="mt-0.5 text-[10px] text-neutral-500">
                    pending upload
                  </div>
                </Box>
              </div>

              {/* live shutter pulse */}
              <Box dashed className="!bg-neutral-50">
                <div className="flex items-center justify-between">
                  <Caption>Tether · live</Caption>
                  <Tag tone="ok">SHUTTER ARMED</Tag>
                </div>
                <div className="mt-2 flex items-center gap-2">
                  <span className="inline-block h-2 w-2 rounded-full bg-rose-600 animate-pulse" />
                  <div className="font-mono text-[11px] text-neutral-700">
                    Burst @ 8 fps · last frame 0.12 s ago
                  </div>
                </div>
              </Box>

              {/* thumbnail strip */}
              <div>
                <div className="mb-1 flex items-center justify-between">
                  <Caption>Recent captures</Caption>
                  <span className="font-mono text-[10px] text-neutral-500">
                    newest →
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-1.5">
                  {frames.map((f, i) => (
                    <div
                      key={i}
                      className="relative aspect-[3/2] border border-neutral-300 bg-neutral-100"
                    >
                      <div className="absolute inset-0 flex items-center justify-center font-mono text-xs text-neutral-400">
                        IMG_{2148 - (frames.length - 1 - i)}
                      </div>
                      <div className="absolute left-1 top-1">
                        <Tag tone={f.tone}>{f.state}</Tag>
                      </div>
                      <div className="absolute right-1 bottom-1 font-mono text-[9px] text-neutral-600">
                        {f.time}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* failed item action row */}
              <Box className="!border-rose-700 !bg-rose-50">
                <div className="flex items-center justify-between">
                  <div className="text-[11px] text-neutral-800">
                    IMG_2143 · failed (E2 camera disconnect)
                  </div>
                  <div className="flex gap-1">
                    <Btn small>Retry</Btn>
                    <Btn small>Error</Btn>
                  </div>
                </div>
              </Box>

              {/* storage warning */}
              <Box className="!border-amber-700 !bg-amber-50">
                <div className="flex items-center justify-between">
                  <Caption>Storage low (E1)</Caption>
                  <Tag tone="warn">412 MB free</Tag>
                </div>
                <div className="mt-2 h-1.5 w-full overflow-hidden bg-amber-200">
                  <div className="h-full w-[88%] bg-amber-700" />
                </div>
                <div className="mt-2 flex gap-2">
                  <Btn small primary>Free up space</Btn>
                  <Btn small>Dismiss</Btn>
                </div>
              </Box>
            </div>
          </PhoneFrame>
        </div>

        <div className="col-span-12 lg:col-span-5">
          <div className="mb-3"><Caption>Annotations</Caption></div>
          <div className="space-y-3">
            <Annot n={1}>Captured counter — increments on every shutter signal received over PTP/SDK (MSS step 1).</Annot>
            <Annot n={2}>Queue-depth counter — number of records appended to the local upload queue (MSS step 6).</Annot>
            <Annot n={3}>Thumbnail strip — each tile shows state badge (uploaded / uploading / queued / failed) per UC-M1-1.3.</Annot>
            <Annot n={4}>Per-photo retry — surfaces E2 (camera disconnect mid-capture) and offers re-fetch from SD card.</Annot>
            <Annot n={5}>Storage-low warning (E1) — fires when free space drops below the 500 MB default threshold.</Annot>
          </div>
          <div className="mt-6">
            <Placeholder
              label="Live preview · last frame"
              height="h-32"
            />
          </div>
        </div>
      </div>
    </WireframePage>
  );
}
