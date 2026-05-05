import { WireframePage } from "../_components/WireframePage";
import { PhoneFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Placeholder, Tag } from "../_components/primitives";

export default function M15() {
  return (
    <WireframePage
      module="M1"
      ucId="UC-M1-1.5"
      title="Receive Photo-Found Notification (Runner)"
      tracesTo="SO1.3 · GO1"
      mustShow={[
        "Lock-screen notification copy",
        "In-app toast variant",
        "Deep-link landing on filtered gallery with new-match count and watermarked previews",
        "In-app inbox fallback for OS-disabled notifications",
      ]}
    >
      <div className="grid grid-cols-12 gap-8">
        {/* lock-screen + deep-link gallery */}
        <div className="col-span-12 lg:col-span-7 flex justify-center">
          <PhoneFrame statusLabel="ANDROID · LOCK SCREEN">
            <div className="relative h-[640px] bg-gradient-to-b from-neutral-200 to-neutral-300 p-4">
              {/* clock */}
              <div className="absolute left-0 right-0 top-10 text-center">
                <div className="font-mono text-5xl text-neutral-800">9:41</div>
                <div className="mt-1 font-mono text-[11px] uppercase tracking-wider text-neutral-600">
                  Sat · 12 Apr 2026
                </div>
              </div>

              {/* notification card */}
              <div className="absolute left-3 right-3 top-44">
                <Box className="!bg-white/90 backdrop-blur shadow-lg">
                  <div className="flex items-start gap-2">
                    <div className="h-7 w-7 rounded-md border border-neutral-900 bg-neutral-900 text-white flex items-center justify-center font-mono text-[10px]">
                      QP
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-[10px] uppercase tracking-wider text-neutral-500">
                          QuickPitik · now
                        </span>
                        <Tag tone="info">3 NEW</Tag>
                      </div>
                      <div className="mt-1 text-sm font-semibold">
                        We found you in 3 photos
                      </div>
                      <div className="text-[12px] text-neutral-700">
                        Cebu City Marathon 2026 · tap to view
                      </div>
                    </div>
                  </div>
                </Box>
              </div>

              {/* in-app toast variant */}
              <div className="absolute left-3 right-3 bottom-32">
                <Box dashed>
                  <Caption>A1 · App in foreground (toast)</Caption>
                  <div className="mt-1 flex items-center justify-between">
                    <span className="text-[12px] text-neutral-800">
                      3 new matches in your event
                    </span>
                    <Btn small>View</Btn>
                  </div>
                </Box>
              </div>

              <div className="absolute left-0 right-0 bottom-6 text-center font-mono text-[10px] uppercase tracking-wider text-neutral-500">
                ↑ swipe up to unlock
              </div>
            </div>
          </PhoneFrame>
        </div>

        {/* deep-link gallery */}
        <div className="col-span-12 lg:col-span-5 flex justify-center">
          <PhoneFrame statusLabel="QUICKPITIK · GALLERY">
            <div className="flex flex-col gap-3 p-4">
              <Box className="!bg-neutral-900 !text-white !border-neutral-900">
                <Caption><span className="text-neutral-400">Deep-link landing · MSS step 9</span></Caption>
                <div className="mt-1 text-base font-semibold">
                  Cebu City Marathon 2026
                </div>
                <div className="text-[11px] text-neutral-300">
                  3 new matches · runner #4218
                </div>
              </Box>

              <div className="grid grid-cols-2 gap-2">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Placeholder
                    key={i}
                    label={`Match ${i + 1}\nWATERMARKED`}
                    height="h-28"
                  />
                ))}
              </div>

              <Box>
                <Caption>E3 · OS notifications disabled</Caption>
                <div className="mt-1 flex items-center justify-between">
                  <span className="text-[11px] text-neutral-800">In-app inbox · 3 unread</span>
                  <Btn small>Open inbox</Btn>
                </div>
              </Box>

              <div className="space-y-2 pt-1">
                <Annot n={1}>Lock-screen copy = event name + count, no biometric data.</Annot>
                <Annot n={2}>Toast (A1) when foregrounded — refreshes gallery inline.</Annot>
                <Annot n={3}>Deep link → filtered gallery scoped to runner + event.</Annot>
                <Annot n={4}>Fallback in-app inbox (E3) for OS-disabled notifications.</Annot>
              </div>
            </div>
          </PhoneFrame>
        </div>
      </div>
    </WireframePage>
  );
}
