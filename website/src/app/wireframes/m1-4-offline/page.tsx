import { WireframePage } from "../_components/WireframePage";
import { PhoneFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Tag } from "../_components/primitives";

export default function M14() {
  return (
    <WireframePage
      module="M1"
      ucId="UC-M1-1.4"
      title="Local-Cache During Signal Loss"
      tracesTo="SO1.2 · GO1"
      mustShow={[
        "Offline banner with cached-count and storage-remaining",
        "Per-photo ‘cached’ badge",
        "‘Retry now’ CTA",
        "Storage-low blocking dialog (E1)",
        "Auto-resume transition to upload state on recovery",
      ]}
    >
      <div className="grid grid-cols-12 gap-8">
        <div className="col-span-12 lg:col-span-7 flex justify-center">
          <PhoneFrame statusLabel="QUICKPITIK · OFFLINE">
            <div className="flex flex-col gap-3 p-4">
              {/* offline banner */}
              <Box className="!bg-amber-50 !border-amber-700">
                <div className="flex items-center justify-between">
                  <Caption>Offline — caching locally</Caption>
                  <Tag tone="warn">NO INTERNET</Tag>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-neutral-800">
                  <div>
                    <div className="text-neutral-500">Cached</div>
                    <div className="font-mono text-lg tabular-nums">316</div>
                  </div>
                  <div>
                    <div className="text-neutral-500">Storage left</div>
                    <div className="font-mono text-lg tabular-nums">7.2 GB</div>
                  </div>
                </div>
                <div className="mt-2 flex gap-2">
                  <Btn small primary>Retry now</Btn>
                  <Btn small>Force Wi-Fi probe</Btn>
                </div>
              </Box>

              {/* photos still entering queue */}
              <Box>
                <Caption>Captures continue (M1.2)</Caption>
                <div className="mt-2 grid grid-cols-4 gap-1.5">
                  {Array.from({ length: 8 }).map((_, i) => (
                    <div
                      key={i}
                      className="relative aspect-square border border-neutral-300 bg-neutral-100"
                    >
                      <div className="absolute left-0.5 top-0.5">
                        <Tag>cached</Tag>
                      </div>
                    </div>
                  ))}
                </div>
              </Box>

              {/* storage exhaustion blocking dialog */}
              <Box className="!bg-rose-50 !border-rose-700">
                <div className="flex items-center justify-between">
                  <Caption>Free space (E1 · blocking)</Caption>
                  <Tag tone="err">200 MB left</Tag>
                </div>
                <div className="mt-1 text-[11px] text-neutral-800">
                  Cache cannot grow further. Delete already-uploaded photos to
                  continue capturing.
                </div>
                <ul className="mt-2 space-y-1 text-[11px] text-neutral-700">
                  <li>· 2,144 photos already uploaded · 14.6 GB</li>
                  <li>· 0 photos pending re-fetch</li>
                </ul>
                <div className="mt-2 flex gap-2">
                  <Btn small primary>Delete uploaded</Btn>
                  <Btn small>Cancel</Btn>
                </div>
              </Box>

              {/* recovery transition */}
              <Box dashed className="!bg-emerald-50 !border-emerald-700">
                <div className="flex items-center justify-between">
                  <Caption>Network restored — resuming</Caption>
                  <Tag tone="ok">PROBE OK</Tag>
                </div>
                <div className="mt-2 h-1.5 w-full overflow-hidden bg-emerald-200">
                  <div className="h-full w-[6%] bg-emerald-700" />
                </div>
                <div className="mt-1 font-mono text-[10px] text-neutral-600">
                  Auto-resuming UC-M1-1.3 from queue head · 0 / 316
                </div>
              </Box>

              {/* persistent offline reminder */}
              <Box className="!bg-neutral-100">
                <Caption>E2 · Persistent offline · 32 min</Caption>
                <div className="mt-1 text-[11px] text-neutral-700">
                  Reminder notification scheduled — cache remains intact across
                  app restarts.
                </div>
              </Box>
            </div>
          </PhoneFrame>
        </div>

        <div className="col-span-12 lg:col-span-5">
          <div className="mb-3"><Caption>Annotations</Caption></div>
          <div className="space-y-3">
            <Annot n={1}>Offline banner — surfaced within 10 s of loss; cached-count + storage-remaining always visible (special requirement).</Annot>
            <Annot n={2}>‘Cached’ badge — every photo enters the queue and is persisted locally (MSS step 4).</Annot>
            <Annot n={3}>‘Retry now’ — A1 force-resume; short-circuits the auto-poll.</Annot>
            <Annot n={4}>E1 free-space dialog — blocks new captures, lists already-uploaded photos that may be safely deleted.</Annot>
            <Annot n={5}>Recovery transition — health probe → resume UC-M1-1.3; offline indicator becomes uploading indicator.</Annot>
          </div>
        </div>
      </div>
    </WireframePage>
  );
}
