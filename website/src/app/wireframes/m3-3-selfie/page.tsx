import { WireframePage } from "../_components/WireframePage";
import { BrowserFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Placeholder, Tag } from "../_components/primitives";

export default function M33() {
  return (
    <WireframePage
      module="M3"
      ucId="UC-M3-3.3"
      title="Search by Selfie"
      tracesTo="SO3.1, SO3.2 · GO3"
      mustShow={[
        "Selfie capture / upload step with camera preview",
        "In-progress state with elapsed time",
        "Result gallery with similarity badges",
        "No-match / multiple-faces states",
        "Revocation-redirect to bib search",
      ]}
    >
      <BrowserFrame url="quickpitik.ph/events/EV-118/search/selfie">
        <div className="grid grid-cols-12 gap-0">
          {/* capture pane */}
          <div className="col-span-5 border-r border-neutral-200 p-6">
            <Caption>Step 1 · Capture or upload selfie</Caption>
            <h3 className="mt-1 font-display text-xl font-semibold">
              Cebu City Marathon 2026
            </h3>
            <p className="mt-1 text-[11px] text-neutral-600">
              We&apos;ll match your face to event photos. Selfie is sent over
              HTTPS, the embedding is stored, and the original is deleted after
              search.
            </p>

            <div className="mt-4 relative">
              <Placeholder label="CAMERA PREVIEW · 1:1" height="h-64" />
              {/* face guide */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="h-40 w-40 rounded-full border-2 border-dashed border-neutral-900/40" />
              </div>
            </div>

            <div className="mt-3 flex gap-2">
              <Btn primary>Capture</Btn>
              <Btn>Upload photo</Btn>
              <Btn>Switch camera</Btn>
            </div>

            <Box className="mt-4 !bg-amber-50 !border-amber-700">
              <Caption>A1 · multiple faces detected</Caption>
              <div className="mt-1 text-[11px] text-neutral-800">
                Crop to a single face or retake.
              </div>
            </Box>

            <Box className="mt-2 !bg-rose-50 !border-rose-700">
              <Caption>E3 · consent revoked</Caption>
              <div className="mt-1 flex items-center justify-between">
                <span className="text-[11px] text-neutral-800">
                  Selfie search disabled · use bib search →
                </span>
                <Btn small>Bib search</Btn>
              </div>
            </Box>
          </div>

          {/* in-progress + results */}
          <div className="col-span-7 p-6">
            <div className="flex items-center justify-between">
              <Caption>Step 2 · Search</Caption>
              <Tag tone="info">backend → ai-api /faces/search?event_id=EV-118</Tag>
            </div>

            <Box className="mt-2 !bg-neutral-900 !text-white !border-neutral-900">
              <div className="flex items-center justify-between">
                <Caption><span className="text-neutral-400">In progress</span></Caption>
                <Tag tone="ok">SO3.2 budget · 4 s</Tag>
              </div>
              <div className="mt-2 h-2 w-full overflow-hidden bg-neutral-700">
                <div className="h-full w-[55%] bg-emerald-400" />
              </div>
              <div className="mt-2 flex justify-between font-mono text-[10px] text-neutral-300">
                <span>extracting embedding · pgvector cosine</span>
                <span>elapsed 1.2 s</span>
              </div>
            </Box>

            <div className="mt-4 flex items-center justify-between">
              <Caption>Step 3 · Matches · 14 found · sorted by score</Caption>
              <span className="font-mono text-[10px] text-neutral-500">
                threshold · 0.62 (per-event)
              </span>
            </div>
            <div className="mt-2 grid grid-cols-3 gap-2">
              {[
                { score: 0.93 },
                { score: 0.89 },
                { score: 0.86 },
                { score: 0.81 },
                { score: 0.74 },
                { score: 0.68 },
              ].map((m, i) => (
                <div
                  key={i}
                  className="relative aspect-[4/3] border border-neutral-300 bg-neutral-100"
                >
                  <Placeholder label={`MATCH ${i + 1}\nWATERMARKED`} height="h-full" />
                  <div className="absolute left-1 top-1">
                    <Tag tone="ok">score {m.score.toFixed(2)}</Tag>
                  </div>
                  <div className="absolute right-1 bottom-1">
                    <Tag>preview</Tag>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-3 flex items-center gap-2 text-[11px] text-neutral-600">
              <Btn small>View next page</Btn>
              <span>· total 14 · showing 1–6</span>
            </div>

            {/* no-match */}
            <Box className="mt-4 !bg-neutral-50">
              <Caption>A2 · below-threshold (no match)</Caption>
              <div className="mt-1 text-[11px] text-neutral-800">
                We didn&apos;t find any photos above the confidence threshold.
                Try bib search instead.
              </div>
              <div className="mt-2 flex gap-2">
                <Btn small primary>Try bib search</Btn>
                <Btn small>Retake selfie</Btn>
              </div>
            </Box>

            <div className="mt-4 grid grid-cols-2 gap-2">
              <Annot n={1}>Camera preview with face guide; capture or upload.</Annot>
              <Annot n={2}>HTTPS-only transit; selfie deleted after search; embedding stored.</Annot>
              <Annot n={3}>In-progress card with SO3.2 budget timer.</Annot>
              <Annot n={4}>Result gallery — watermarked previews, similarity badges, descending score.</Annot>
              <Annot n={5}>A1 / A2 / E3 surfaced with redirect to bib search where applicable.</Annot>
            </div>
          </div>
        </div>
      </BrowserFrame>
    </WireframePage>
  );
}
