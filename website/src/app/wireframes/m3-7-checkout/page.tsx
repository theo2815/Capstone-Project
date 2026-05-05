import { WireframePage } from "../_components/WireframePage";
import { BrowserFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Field, Tag } from "../_components/primitives";

export default function M37() {
  return (
    <WireframePage
      module="M3"
      ucId="UC-M3-3.7"
      title="Checkout & Pay"
      tracesTo="SO3.3 · GO3"
      mustShow={[
        "Order summary (line items + total in PHP)",
        "Payment-method selector",
        "PayMongo authorisation surface (card form / e-wallet redirect)",
        "3DS / OTP challenge screen",
        "Decline / retry state",
        "Success / receipt screen",
      ]}
    >
      <BrowserFrame url="quickpitik.ph/checkout/ord_91A42">
        <div className="grid grid-cols-12">
          <div className="col-span-12 flex items-center justify-between border-b border-neutral-200 px-6 py-3">
            <div>
              <span className="font-mono text-[10px] text-neutral-500">
                Order ord_91A42 · status pending → paid
              </span>
              <div className="text-sm font-semibold">Checkout</div>
            </div>
            <Tag tone="info">PayMongo · PHP</Tag>
          </div>

          {/* method selector + card form */}
          <main className="col-span-8 p-6">
            <Caption>Step 1 · Payment method</Caption>
            <div className="mt-2 grid grid-cols-4 gap-2">
              {[
                { name: "Card", note: "Visa · MC · JCB", sel: true },
                { name: "GCash", note: "e-wallet" },
                { name: "Maya", note: "e-wallet" },
                { name: "Bank", note: "QR Ph" },
              ].map((m) => (
                <span
                  key={m.name}
                  className={[
                    "border px-3 py-3 text-center",
                    m.sel
                      ? "border-neutral-900 bg-neutral-900 text-white"
                      : "border-neutral-300 bg-white",
                  ].join(" ")}
                >
                  <div className="text-sm font-semibold">{m.name}</div>
                  <div className="font-mono text-[10px] opacity-70">{m.note}</div>
                </span>
              ))}
            </div>

            {/* card form */}
            <Box className="mt-4">
              <Caption>Step 2 · Authorise (PayMongo card)</Caption>
              <div className="mt-3 grid grid-cols-2 gap-3">
                <Field label="Cardholder name" placeholder="Maria Dela Cruz" className="!col-span-2" />
                <Field label="Card number" value="4242 4242 4242 4242" className="!col-span-2" />
                <Field label="Expiry" value="04 / 28" />
                <Field label="CVC" value="•••" />
              </div>
              <div className="mt-3 flex items-center gap-2">
                <Tag tone="info">tokenised by PayMongo</Tag>
                <Tag tone="info">3DS / OTP supported</Tag>
              </div>
              <Btn primary className="!mt-3">
                Pay ₱ 846.00
              </Btn>
            </Box>

            {/* 3DS challenge */}
            <Box className="mt-4 !bg-neutral-50">
              <Caption>Step 3 · A1 · 3DS / OTP challenge</Caption>
              <div className="mt-2 grid grid-cols-2 gap-3">
                <div>
                  <Caption>Issuer challenge</Caption>
                  <div className="mt-1 text-[12px] text-neutral-800">
                    Your bank sent a 6-digit code to ••• 4218.
                  </div>
                  <div className="mt-2 grid grid-cols-6 gap-1">
                    {["7", "3", "9", "_", "_", "_"].map((d, i) => (
                      <span
                        key={i}
                        className="border border-neutral-700 px-2 py-2 text-center font-mono"
                      >
                        {d}
                      </span>
                    ))}
                  </div>
                  <div className="mt-2 flex gap-2">
                    <Btn small primary>Verify</Btn>
                    <Btn small>Resend</Btn>
                  </div>
                </div>
                <div>
                  <Caption>Webhook · payment.succeeded</Caption>
                  <ul className="mt-2 space-y-1 font-mono text-[10px] text-neutral-700">
                    <li>· receive PayMongo webhook</li>
                    <li>· verify HMAC signature (E2)</li>
                    <li>· transition order pending → paid</li>
                    <li>· emit purchase succeeded event</li>
                  </ul>
                </div>
              </div>
            </Box>

            {/* states */}
            <div className="mt-4 grid grid-cols-3 gap-3">
              <Box className="!bg-rose-50 !border-rose-700">
                <Caption>E1 · payment declined</Caption>
                <div className="mt-1 text-[11px] text-neutral-800">
                  Your bank declined this card.
                </div>
                <Btn small className="!mt-2">Try another method</Btn>
              </Box>
              <Box className="!bg-amber-50 !border-amber-700">
                <Caption>E2 · webhook signature invalid</Caption>
                <div className="mt-1 text-[11px] text-neutral-800">
                  Order remains pending; user shown a retry CTA.
                </div>
              </Box>
              <Box className="!bg-amber-50 !border-amber-700">
                <Caption>E3 · cart mutated mid-checkout (409)</Caption>
                <div className="mt-1 text-[11px] text-neutral-800">
                  Sent back to review the cart.
                </div>
              </Box>
            </div>
          </main>

          {/* order summary + receipt */}
          <aside className="col-span-4 border-l border-neutral-200 bg-neutral-50 p-6">
            <Caption>Order summary</Caption>
            <div className="mt-2 space-y-1.5 font-mono text-[11px] text-neutral-800">
              {[
                ["PH-91204", 199],
                ["PH-91207", 199],
                ["PH-91219", 249],
                ["PH-91224", 199],
              ].map(([id, p]) => (
                <div
                  key={id as string}
                  className="flex items-center justify-between"
                >
                  <span>{id}</span>
                  <span>₱ {(p as number).toFixed(2)}</span>
                </div>
              ))}
            </div>
            <div className="mt-3 border-t border-neutral-300 pt-2 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-neutral-600">Subtotal</span>
                <span className="font-mono">₱ 846.00</span>
              </div>
              <div className="mt-2 flex items-center justify-between font-semibold">
                <span>Total · PHP</span>
                <span className="font-mono">₱ 846.00</span>
              </div>
            </div>

            <Box className="mt-4 !bg-emerald-50 !border-emerald-700">
              <Caption>Step 4 · receipt screen</Caption>
              <div className="mt-1 text-[11px] text-neutral-800">
                Payment received. Your downloads are being prepared (M3.8).
              </div>
              <div className="mt-2 flex gap-2">
                <Btn small primary>Go to downloads</Btn>
                <Btn small>Email receipt</Btn>
              </div>
            </Box>

            <div className="mt-4 space-y-2">
              <Annot n={1}>Method selector → PayMongo card / GCash / Maya / QR Ph.</Annot>
              <Annot n={2}>3DS / OTP challenge (A1) honoured.</Annot>
              <Annot n={3}>Webhook signature verified before pending → paid.</Annot>
              <Annot n={4}>PHP-only · explicit currency in totals.</Annot>
            </div>
          </aside>
        </div>
      </BrowserFrame>
    </WireframePage>
  );
}
