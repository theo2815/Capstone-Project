import { WireframePage } from "../_components/WireframePage";
import { BrowserFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Placeholder, Tag } from "../_components/primitives";

export default function M36() {
  const lines = [
    { id: "PH-91204", price: 199, sel: true },
    { id: "PH-91207", price: 199, sel: true },
    { id: "PH-91212", price: 199, sel: false },
    { id: "PH-91219", price: 249, sel: true },
  ];
  const total = lines.reduce((s, l) => s + l.price, 0);

  return (
    <WireframePage
      module="M3"
      ucId="UC-M3-3.6"
      title="Add to Cart"
      tracesTo="SO3.3 · GO3"
      mustShow={[
        "‘Add to cart’ CTA on preview and on gallery cards",
        "Multi-select bulk-add affordance",
        "Cart counter",
        "Cart drawer / page with line items, prices, and remove control",
        "Error toasts",
      ]}
    >
      <BrowserFrame url="quickpitik.ph/events/EV-118/results">
        <div className="grid grid-cols-12">
          {/* nav with cart counter */}
          <div className="col-span-12 flex items-center justify-between border-b border-neutral-200 px-6 py-3">
            <span className="font-display text-lg font-semibold">QuickPitik</span>
            <div className="flex items-center gap-3">
              <Tag tone="info">runner · @maria</Tag>
              <span className="relative inline-flex items-center gap-1 border border-neutral-900 px-2 py-1 font-mono text-[11px]">
                Cart
                <span className="ml-1 inline-flex h-5 min-w-[20px] items-center justify-center rounded-full bg-neutral-900 px-1 text-[10px] text-white">
                  4
                </span>
              </span>
            </div>
          </div>

          {/* gallery with multi-select */}
          <main className="col-span-8 p-5">
            <div className="flex items-center justify-between border-b border-neutral-200 pb-3">
              <Caption>Selfie matches · 14 photos · multi-select mode</Caption>
              <div className="flex items-center gap-2">
                <Tag>3 selected</Tag>
                <Btn small primary>Bulk add → cart (A1)</Btn>
              </div>
            </div>

            <div className="mt-3 grid grid-cols-4 gap-2">
              {Array.from({ length: 8 }).map((_, i) => {
                const selected = [0, 1, 3].includes(i);
                return (
                  <div
                    key={i}
                    className={[
                      "relative aspect-[4/3] border bg-neutral-100",
                      selected
                        ? "border-neutral-900 ring-2 ring-neutral-900"
                        : "border-neutral-300",
                    ].join(" ")}
                  >
                    <Placeholder label={`PH-9120${i}`} height="h-full" />
                    {/* check */}
                    <div
                      className={[
                        "absolute right-1 top-1 h-5 w-5 border border-neutral-900 flex items-center justify-center font-mono text-[10px]",
                        selected ? "bg-neutral-900 text-white" : "bg-white",
                      ].join(" ")}
                    >
                      {selected ? "✓" : ""}
                    </div>
                    <div className="absolute left-1 bottom-1">
                      <Btn small primary>Add</Btn>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* error toasts */}
            <div className="mt-4 grid grid-cols-2 gap-3">
              <Box className="!bg-amber-50 !border-amber-700">
                <Caption>E1 · already in cart (409)</Caption>
                <div className="mt-1 text-[11px] text-neutral-800">
                  PH-91204 is already in your cart.
                </div>
              </Box>
              <Box className="!bg-rose-50 !border-rose-700">
                <Caption>E2 · photo retired (410)</Caption>
                <div className="mt-1 text-[11px] text-neutral-800">
                  This photo is no longer available.
                </div>
              </Box>
            </div>
          </main>

          {/* cart drawer */}
          <aside className="col-span-4 border-l border-neutral-200 bg-neutral-50 p-5">
            <div className="flex items-center justify-between">
              <Caption>Cart drawer</Caption>
              <span className="font-mono text-[10px] text-neutral-500">
                4 items · ₱ {total}.00
              </span>
            </div>

            <div className="mt-3 space-y-2">
              {lines.map((l) => (
                <Box key={l.id} className="!p-2 flex items-center gap-2">
                  <div className="h-12 w-16 border border-neutral-300 bg-neutral-100" />
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[11px]">{l.id}</span>
                      <span className="font-mono text-[11px]">
                        ₱ {l.price.toFixed(2)}
                      </span>
                    </div>
                    <div className="mt-0.5 flex items-center justify-between">
                      <Tag tone={l.sel ? "ok" : "neutral"}>
                        {l.sel ? "in cart" : "removed"}
                      </Tag>
                      <button className="font-mono text-[10px] text-neutral-500 underline">
                        Remove
                      </button>
                    </div>
                  </div>
                </Box>
              ))}
            </div>

            <div className="mt-4 border-t border-neutral-300 pt-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-neutral-600">Subtotal</span>
                <span className="font-mono">₱ {total}.00</span>
              </div>
              <div className="mt-1 flex items-center justify-between text-sm">
                <span className="text-neutral-600">VAT (incl.)</span>
                <span className="font-mono">₱ 0.00</span>
              </div>
              <div className="mt-2 flex items-center justify-between border-t border-neutral-300 pt-2 font-semibold">
                <span>Total · PHP</span>
                <span className="font-mono">₱ {total}.00</span>
              </div>
              <Btn primary className="!mt-3 w-full">
                Checkout →
              </Btn>
            </div>

            <div className="mt-4 space-y-2">
              <Annot n={1}>Cart counter in nav · server-side cart per JWT subject.</Annot>
              <Annot n={2}>Bulk-add (A1) from multi-select gallery.</Annot>
              <Annot n={3}>Per-line remove (A2) · DELETE /v1/cart/items/&lt;id&gt;.</Annot>
              <Annot n={4}>Pricing snapshot at add-time honoured at checkout.</Annot>
            </div>
          </aside>
        </div>
      </BrowserFrame>
    </WireframePage>
  );
}
