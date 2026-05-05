import { WireframePage } from "../_components/WireframePage";
import { BrowserFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Placeholder, Tag } from "../_components/primitives";

export default function M35() {
  return (
    <WireframePage
      module="M3"
      ucId="UC-M3-3.5"
      title="Preview Photo (Watermarked)"
      tracesTo="SO3.3 · GO3"
      mustShow={[
        "Watermarked photo at high res",
        "Lightbox with prev / next",
        "‘Add to cart’ CTA",
        "Preview-unavailable state",
        "URL-expired re-fetch state",
        "Mobile swipe affordance",
      ]}
    >
      <BrowserFrame url="quickpitik.ph/photos/PH-91204/preview">
        <div className="grid grid-cols-12">
          <div className="col-span-12 flex items-center justify-between border-b border-neutral-200 px-6 py-3">
            <div>
              <span className="font-mono text-[10px] text-neutral-500">
                Cebu City Marathon 2026 · result 4 of 14
              </span>
              <div className="text-sm font-semibold">
                Lightbox preview · GET /v1/photos/PH-91204/preview
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Tag tone="info">short-TTL signed URL</Tag>
              <Btn small>Close ✕</Btn>
            </div>
          </div>

          {/* lightbox stage */}
          <div className="col-span-9 p-6">
            <div className="relative aspect-[16/10] border border-neutral-900 bg-neutral-100">
              <Placeholder label="WATERMARKED PHOTO" height="h-full" diagonals />
              {/* watermark overlay */}
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                <div className="-rotate-12 text-center">
                  <div className="font-display text-5xl font-bold text-neutral-900/15">
                    QUICKPITIK
                  </div>
                  <div className="font-display text-3xl font-bold text-neutral-900/15">
                    QUICKPITIK
                  </div>
                  <div className="font-display text-5xl font-bold text-neutral-900/15">
                    QUICKPITIK
                  </div>
                </div>
              </div>
              {/* prev / next */}
              <button className="absolute left-3 top-1/2 -translate-y-1/2 h-10 w-10 border border-neutral-900 bg-white font-mono">
                ‹
              </button>
              <button className="absolute right-3 top-1/2 -translate-y-1/2 h-10 w-10 border border-neutral-900 bg-white font-mono">
                ›
              </button>
              <div className="absolute left-3 top-3 flex gap-1">
                <Tag tone="ok">match · 0.93</Tag>
                <Tag>#4218</Tag>
              </div>
            </div>

            {/* thumbnail strip */}
            <div className="mt-3 grid grid-cols-8 gap-1">
              {Array.from({ length: 8 }).map((_, i) => (
                <div
                  key={i}
                  className={[
                    "aspect-[4/3] border bg-neutral-100",
                    i === 3
                      ? "border-neutral-900 ring-2 ring-neutral-900"
                      : "border-neutral-300",
                  ].join(" ")}
                />
              ))}
            </div>
            <div className="mt-2 font-mono text-[10px] text-neutral-500">
              ← swipe / arrow keys to navigate · A1 adjacent navigation
            </div>
          </div>

          {/* sidebar */}
          <aside className="col-span-3 border-l border-neutral-200 p-5">
            <Caption>Photo · PH-91204</Caption>
            <ul className="mt-2 space-y-1 font-mono text-[11px] text-neutral-700">
              <li>capture · 06:42:18</li>
              <li>photographer · @theo</li>
              <li>resolution · 6000 × 4000</li>
              <li>price · ₱ 199.00</li>
            </ul>

            <Btn primary className="!mt-4">
              Add to cart →
            </Btn>
            <Btn className="!mt-2">Add all 14 to cart</Btn>

            <Box className="mt-4 !bg-amber-50 !border-amber-700">
              <Caption>E1 · preview not ready</Caption>
              <div className="mt-1 flex items-center justify-between">
                <span className="text-[11px] text-neutral-800">404 / 410</span>
                <Btn small>Retry</Btn>
              </div>
            </Box>

            <Box className="mt-2 !bg-neutral-100">
              <Caption>E2 · URL expired</Caption>
              <div className="mt-1 text-[11px] text-neutral-800">
                Re-fetching signed URL transparently…
              </div>
            </Box>

            <div className="mt-4 space-y-2">
              <Annot n={1}>Watermark overlay · server-side at preview generation.</Annot>
              <Annot n={2}>Un-watermarked original NOT served before payment.</Annot>
              <Annot n={3}>Prev / next cycle through result set; mobile swipe.</Annot>
              <Annot n={4}>Add-to-cart → UC-M3-3.6.</Annot>
            </div>
          </aside>
        </div>
      </BrowserFrame>
    </WireframePage>
  );
}
