import { WireframePage } from "../_components/WireframePage";
import { BrowserFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Field, Placeholder, Tag } from "../_components/primitives";

export default function M34() {
  return (
    <WireframePage
      module="M3"
      ucId="UC-M3-3.4"
      title="Search by Bib Number"
      tracesTo="SO3.1, SO3.2 · GO3"
      mustShow={[
        "Input switcher (text vs photo)",
        "Bib-image upload with OCR preview",
        "Multi-candidate disambiguation",
        "Result gallery",
        "No-match state",
        "OCR failure prompt",
      ]}
    >
      <BrowserFrame url="quickpitik.ph/events/EV-118/search/bib">
        <div className="grid grid-cols-12 gap-0">
          {/* input pane */}
          <div className="col-span-5 border-r border-neutral-200 p-6">
            <Caption>Step 1 · Input bib number</Caption>
            <h3 className="mt-1 font-display text-xl font-semibold">
              Cebu City Marathon 2026
            </h3>

            {/* input switcher */}
            <div className="mt-4 grid grid-cols-2 border border-neutral-900">
              <span className="border-r border-neutral-900 bg-neutral-900 py-2 text-center text-[11px] font-medium text-white">
                Type number
              </span>
              <span className="py-2 text-center text-[11px] text-neutral-700">
                Upload bib photo
              </span>
            </div>

            {/* type variant */}
            <div className="mt-4">
              <Field label="Bib number" value="4218" />
              <div className="mt-2 flex gap-2">
                <Btn primary>Find my photos</Btn>
                <Btn>Clear</Btn>
              </div>
            </div>

            <div className="my-4 h-px bg-neutral-200" />

            {/* image variant */}
            <div>
              <Caption>Or · upload bib photo (OCR via PaddleOCR)</Caption>
              <Box dashed className="!bg-neutral-50 !mt-2">
                <Placeholder label="BIB PHOTO" height="h-32" />
                <div className="mt-2 flex gap-2">
                  <Btn small>Upload</Btn>
                  <Btn small>Take photo</Btn>
                </div>
              </Box>
            </div>

            <Box className="mt-3 !bg-amber-50 !border-amber-700">
              <Caption>E2 · OCR failed</Caption>
              <div className="mt-1 text-[11px] text-neutral-800">
                We couldn&apos;t read a bib number. Retake the photo or type
                the number manually.
              </div>
            </Box>
          </div>

          {/* OCR + candidates + results */}
          <div className="col-span-7 p-6">
            <div className="flex items-center justify-between">
              <Caption>Step 2 · OCR result</Caption>
              <Tag tone="info">backend → ai-api /bibs/recognize</Tag>
            </div>

            <Box className="mt-2">
              <div className="grid grid-cols-3 gap-3">
                <Placeholder label="CROP · BIB" height="h-24" />
                <div className="col-span-2">
                  <Caption>Detected digits · A1 multi-candidate</Caption>
                  <div className="mt-2 grid grid-cols-3 gap-2">
                    {[
                      { v: "4218", conf: 0.96, sel: true },
                      { v: "4213", conf: 0.71 },
                      { v: "4218B", conf: 0.42 },
                    ].map((c) => (
                      <span
                        key={c.v}
                        className={[
                          "border px-2 py-2 text-center font-mono text-sm",
                          c.sel
                            ? "border-neutral-900 bg-neutral-900 text-white"
                            : "border-neutral-300 bg-white",
                        ].join(" ")}
                      >
                        {c.v}
                        <div className="mt-0.5 text-[9px] opacity-70">
                          conf {c.conf.toFixed(2)}
                        </div>
                      </span>
                    ))}
                  </div>
                  <div className="mt-2 flex gap-2">
                    <Btn small primary>Use 4218</Btn>
                    <Btn small>Type manually</Btn>
                  </div>
                </div>
              </div>
            </Box>

            <div className="mt-4 flex items-center justify-between">
              <Caption>Step 3 · Matches for #4218 · 11 photos · capture-time order</Caption>
              <Tag tone="ok">SO3.2 · 2.4 s</Tag>
            </div>
            <div className="mt-2 grid grid-cols-3 gap-2">
              {Array.from({ length: 6 }).map((_, i) => (
                <div
                  key={i}
                  className="relative aspect-[4/3] border border-neutral-300 bg-neutral-100"
                >
                  <Placeholder label={`PHOTO ${i + 1}\nWATERMARKED`} height="h-full" />
                  <div className="absolute left-1 top-1">
                    <Tag>#4218</Tag>
                  </div>
                </div>
              ))}
            </div>

            <Box className="mt-4 !bg-rose-50 !border-rose-700">
              <Caption>E1 · no bib match</Caption>
              <div className="mt-1 flex items-center justify-between">
                <span className="text-[11px] text-neutral-800">
                  No photos for #4218. Try selfie search?
                </span>
                <Btn small>Selfie search</Btn>
              </div>
            </Box>

            <div className="mt-4 grid grid-cols-2 gap-2">
              <Annot n={1}>Input switcher · text vs bib-image upload.</Annot>
              <Annot n={2}>OCR via ai-api · PaddleOCR PP-OCRv5.</Annot>
              <Annot n={3}>Multi-candidate (A1) — user picks correct digits.</Annot>
              <Annot n={4}>Result gallery — capture-time order, watermarked.</Annot>
              <Annot n={5}>E1 / E2 redirect to selfie search or retake.</Annot>
            </div>
          </div>
        </div>
      </BrowserFrame>
    </WireframePage>
  );
}
