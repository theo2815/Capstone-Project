import { WireframePage } from "../_components/WireframePage";
import { PhoneFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Tag } from "../_components/primitives";

export default function M11() {
  return (
    <WireframePage
      module="M1"
      ucId="UC-M1-1.1"
      title="Tether Camera to Mobile App"
      tracesTo="SO1.1 · GO1"
      mustShow={[
        "Active-event banner",
        "‘Connect camera’ primary CTA",
        "Camera-detection in-progress state",
        "Camera model + battery + mode (USB / Wi-Fi) on success",
        "Troubleshoot dialog on failure",
        "‘Reconnect’ affordance for alternative flow A2",
      ]}
    >
      <div className="grid grid-cols-12 gap-8">
        <div className="col-span-12 lg:col-span-7 flex justify-center">
          <PhoneFrame statusLabel="QUICKPITIK · TETHER">
            <div className="flex flex-col gap-3 p-4">
              {/* active-event banner */}
              <Box className="!p-3 border-neutral-900 !bg-neutral-900 text-white">
                <Caption>
                  <span className="text-neutral-400">Active event</span>
                </Caption>
                <div className="mt-1 text-sm font-semibold">
                  Cebu City Marathon 2026
                </div>
                <div className="text-[11px] text-neutral-300">
                  Photographer · @theo · Booth&nbsp;A
                </div>
              </Box>

              {/* connect camera card */}
              <Box dashed className="!bg-neutral-50">
                <Caption>Step 1 · Connect camera</Caption>
                <div className="mt-2 flex items-center justify-between">
                  <div className="font-mono text-[11px] text-neutral-700">
                    USB OTG · Wi-Fi pairing
                  </div>
                  <Tag tone="warn">Awaiting</Tag>
                </div>
                <div className="mt-3 flex gap-2">
                  <Btn primary>Connect camera</Btn>
                  <Btn>Wi-Fi mode</Btn>
                </div>
              </Box>

              {/* in-progress */}
              <Box>
                <Caption>Detecting camera…</Caption>
                <div className="mt-2 h-1.5 w-full overflow-hidden bg-neutral-200">
                  <div className="h-full w-2/3 bg-neutral-900" />
                </div>
                <div className="mt-1 font-mono text-[10px] text-neutral-500">
                  PTP handshake in progress · 4 s elapsed
                </div>
              </Box>

              {/* success state */}
              <Box className="border-emerald-700">
                <div className="flex items-center justify-between">
                  <Caption>Tether session · ACTIVE</Caption>
                  <Tag tone="ok">CONNECTED</Tag>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
                  <div>
                    <div className="text-neutral-500">Model</div>
                    <div className="font-medium">Canon EOS R6</div>
                  </div>
                  <div>
                    <div className="text-neutral-500">Mode</div>
                    <div className="font-medium">USB OTG</div>
                  </div>
                  <div>
                    <div className="text-neutral-500">Battery</div>
                    <div className="font-medium">82 %</div>
                  </div>
                  <div>
                    <div className="text-neutral-500">Captures</div>
                    <div className="font-medium">Ready · 0</div>
                  </div>
                </div>
              </Box>

              {/* failure / troubleshoot dialog */}
              <Box className="border-rose-700 !bg-rose-50">
                <div className="flex items-center justify-between">
                  <Caption>Troubleshoot · failure</Caption>
                  <Tag tone="err">E3</Tag>
                </div>
                <div className="mt-1 text-[11px] text-neutral-800">
                  Handshake timed out after 10 s. Check cable, retry, or switch
                  to Wi-Fi.
                </div>
                <div className="mt-2 flex gap-2">
                  <Btn small>Reconnect</Btn>
                  <Btn small>Help</Btn>
                </div>
              </Box>
            </div>
          </PhoneFrame>
        </div>

        <div className="col-span-12 lg:col-span-5">
          <div className="mb-3">
            <Caption>Annotations</Caption>
          </div>
          <div className="space-y-3">
            <Annot n={1}>
              Active-event banner — read from M3-side event the photographer is
              joined to.
            </Annot>
            <Annot n={2}>
              Primary CTA &quot;Connect camera&quot; — opens OS USB / Wi-Fi
              permission prompt (steps 3-4 of MSS).
            </Annot>
            <Annot n={3}>
              Detection in-progress — PTP/SDK handshake, 10 s timeout (special
              requirement).
            </Annot>
            <Annot n={4}>
              Success card — exposes camera model, battery, mode (USB / Wi-Fi)
              per MSS step 6.
            </Annot>
            <Annot n={5}>
              Troubleshoot dialog (E3) and Reconnect (A2) — auto-retry every 5 s
              for up to 60 s.
            </Annot>
          </div>
        </div>
      </div>
    </WireframePage>
  );
}
