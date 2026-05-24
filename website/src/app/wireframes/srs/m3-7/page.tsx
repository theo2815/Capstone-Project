import { BrowserFrame } from "../../_components/BrowserFrame";
import { EventBackdrop } from "../../_components/EventBackdrop";

const mono = "var(--font-mono), ui-monospace, monospace";
const sans = "var(--font-sans), system-ui, sans-serif";
const display = "var(--font-display), system-ui, sans-serif";

const PAYMENT_OPTIONS = [
  { id: "gcash", label: "GCash", hint: "Open the GCash app to confirm.", active: true },
  { id: "maya", label: "Maya", hint: "Open the Maya app to confirm.", active: false },
  { id: "card", label: "Credit / Debit card", hint: "Visa, Mastercard, or JCB.", active: false },
];

export default function M3_7_Checkout() {
  return (
    <BrowserFrame
      ucId="UC-M3-3.7"
      screenName="Checkout & Payment · PayMongo"
      url="quickpitik.ph/events/run-cebu-2026?checkout=1"
    >
      <div style={{ height: "100%", position: "relative", overflow: "hidden" }}>
        <EventBackdrop cartCount={3} dim />

        {/* Scrim */}
        <div
          aria-hidden
          style={{
            position: "absolute",
            inset: 0,
            background: "rgba(0,0,0,0.45)",
          }}
        />

        {/* Drawer */}
        <aside
          style={{
            position: "absolute",
            top: 0,
            right: 0,
            bottom: 0,
            width: 384,
            background: "#fff",
            borderLeft: "1.5px solid #000",
            display: "flex",
            flexDirection: "column",
            boxShadow: "-20px 0 40px rgba(0,0,0,0.25)",
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: "22px 24px 18px",
              borderBottom: "1px solid #000",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "flex-start",
              gap: 12,
            }}
          >
            <div style={{ minWidth: 0 }}>
              <div
                style={{
                  fontFamily: mono,
                  fontSize: 10,
                  letterSpacing: "0.3em",
                  textTransform: "uppercase",
                  color: "#666",
                }}
              >
                Checkout
              </div>
              <div
                style={{
                  fontFamily: display,
                  fontSize: 26,
                  fontWeight: 500,
                  letterSpacing: "-0.01em",
                  color: "#000",
                  marginTop: 6,
                  lineHeight: 1.1,
                }}
              >
                Pick how to pay.
              </div>
              <StepIndicator step={1} total={3} />
            </div>
            <span
              aria-hidden
              style={{
                width: 32,
                height: 32,
                borderRadius: 999,
                border: "1px solid #000",
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                fontFamily: mono,
                fontSize: 12,
                color: "#000",
              }}
            >
              ✕
            </span>
          </div>

          {/* Body */}
          <div style={{ flex: 1, overflow: "hidden", padding: "20px 24px", display: "flex", flexDirection: "column", gap: 22 }}>
            {/* Order summary */}
            <section>
              <Kicker>Order Summary</Kicker>
              <div
                style={{
                  marginTop: 6,
                  border: "1px solid #000",
                  borderRadius: 10,
                  background: "#FAFAFA",
                  padding: "14px 18px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "baseline",
                }}
              >
                <span style={{ fontFamily: sans, fontSize: 13, color: "#333" }}>
                  <span style={{ color: "#000", fontVariantNumeric: "tabular-nums" }}>3</span> photos · full resolution
                </span>
                <span
                  style={{
                    fontFamily: display,
                    fontSize: 24,
                    fontWeight: 500,
                    letterSpacing: "-0.01em",
                    color: "#000",
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  ₱150
                </span>
              </div>
            </section>

            {/* Send to */}
            <section>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <Kicker>Send To</Kicker>
                <span
                  style={{
                    fontFamily: mono,
                    fontSize: 9,
                    letterSpacing: "0.22em",
                    textTransform: "uppercase",
                    color: "#666",
                  }}
                >
                  Edit
                </span>
              </div>
              <div style={{ marginTop: 6, fontFamily: sans, fontSize: 13, color: "#000" }}>
                theo@quickpitik.ph
              </div>
              <div
                style={{
                  marginTop: 2,
                  fontFamily: mono,
                  fontSize: 9,
                  letterSpacing: "0.22em",
                  textTransform: "uppercase",
                  color: "#666",
                }}
              >
                Signed-in account
              </div>
            </section>

            {/* Payment method */}
            <section style={{ flex: 1, minHeight: 0 }}>
              <Kicker>Payment Method</Kicker>
              <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 8 }}>
                {PAYMENT_OPTIONS.map((opt) => (
                  <PaymentOption key={opt.id} {...opt} />
                ))}
              </div>
            </section>
          </div>

          {/* Footer CTA */}
          <div style={{ padding: "16px 24px 22px", borderTop: "1px solid #000" }}>
            <button
              type="button"
              style={{
                width: "100%",
                padding: "14px 18px",
                borderRadius: 999,
                background: "#000",
                color: "#fff",
                fontFamily: mono,
                fontSize: 11,
                letterSpacing: "0.22em",
                textTransform: "uppercase",
                border: "1.5px solid #000",
                cursor: "default",
              }}
            >
              Pay ₱150 →
            </button>
            <div
              style={{
                marginTop: 10,
                textAlign: "center",
                fontFamily: mono,
                fontSize: 9,
                letterSpacing: "0.22em",
                textTransform: "uppercase",
                color: "#666",
              }}
            >
              Secure transaction · Watermark removed on download
            </div>
            <div
              style={{
                marginTop: 10,
                fontFamily: mono,
                fontSize: 9,
                letterSpacing: "0.22em",
                textTransform: "uppercase",
                color: "#666",
              }}
            >
              ← Back to cart
            </div>
          </div>
        </aside>
      </div>
    </BrowserFrame>
  );
}

function PaymentOption({ label, hint, active }: { label: string; hint: string; active: boolean }) {
  return (
    <div
      style={{
        display: "flex",
        gap: 12,
        padding: "14px 16px",
        borderRadius: 12,
        border: active ? "1.5px solid #000" : "1px solid #DDD",
        background: active ? "#FAFAFA" : "#fff",
        alignItems: "flex-start",
      }}
    >
      <span
        aria-hidden
        style={{
          width: 16,
          height: 16,
          borderRadius: 999,
          border: "2px solid #000",
          background: active ? "#000" : "#fff",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
          marginTop: 2,
        }}
      >
        {active && <span style={{ width: 5, height: 5, borderRadius: 999, background: "#fff" }} />}
      </span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div
          style={{
            fontFamily: display,
            fontSize: 15,
            fontWeight: 500,
            color: "#000",
            letterSpacing: "-0.005em",
          }}
        >
          {label}
        </div>
        <div style={{ fontFamily: sans, fontSize: 12, color: "#666", marginTop: 2 }}>{hint}</div>
      </div>
    </div>
  );
}

function StepIndicator({ step, total }: { step: number; total: number }) {
  return (
    <div style={{ marginTop: 12, display: "flex", gap: 6 }} aria-hidden>
      {Array.from({ length: total }).map((_, i) => (
        <span
          key={i}
          style={{
            height: 3,
            width: 28,
            borderRadius: 999,
            background: i <= step ? "#000" : "#DDD",
          }}
        />
      ))}
    </div>
  );
}

function Kicker({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        fontFamily: mono,
        fontSize: 10,
        letterSpacing: "0.3em",
        textTransform: "uppercase",
        color: "#666",
      }}
    >
      {children}
    </div>
  );
}
