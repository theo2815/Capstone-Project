import { BrowserFrame } from "../../_components/BrowserFrame";
import { EventBackdrop } from "../../_components/EventBackdrop";
import { Hatch } from "../../_components/WireUI";

const mono = "var(--font-mono), ui-monospace, monospace";
const sans = "var(--font-sans), system-ui, sans-serif";
const display = "var(--font-display), system-ui, sans-serif";

const CART_ITEMS: Array<{ bib: string; time: string; price: number }> = [
  { bib: "1247", time: "01:42:14", price: 50 },
  { bib: "1247", time: "01:44:32", price: 50 },
  { bib: "1247", time: "01:48:07", price: 50 },
];

export default function M3_6_CartDrawer() {
  const total = CART_ITEMS.reduce((s, i) => s + i.price, 0);

  return (
    <BrowserFrame
      ucId="UC-M3-3.6"
      screenName="Cart Drawer · Photo Review & Subtotal"
      url="quickpitik.ph/events/run-cebu-2026?cart=1"
    >
      <div style={{ height: "100%", position: "relative", overflow: "hidden" }}>
        {/* Backdrop with dimmed event page */}
        <EventBackdrop cartCount={CART_ITEMS.length} dim />

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
            <div>
              <div
                style={{
                  fontFamily: mono,
                  fontSize: 10,
                  letterSpacing: "0.3em",
                  textTransform: "uppercase",
                  color: "#666",
                }}
              >
                Your cart
              </div>
              <div
                style={{
                  fontFamily: display,
                  fontSize: 28,
                  fontWeight: 500,
                  letterSpacing: "-0.01em",
                  color: "#000",
                  marginTop: 6,
                  lineHeight: 1.1,
                }}
              >
                {CART_ITEMS.length} photos.
              </div>
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

          {/* Item list */}
          <div style={{ flex: 1, overflow: "hidden" }}>
            {CART_ITEMS.map((item, i) => (
              <CartRow key={i} {...item} last={i === CART_ITEMS.length - 1} />
            ))}
          </div>

          {/* Footer */}
          <div
            style={{
              borderTop: "1px solid #000",
              background: "#FAFAFA",
              padding: "20px 24px",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
                marginBottom: 14,
              }}
            >
              <span
                style={{
                  fontFamily: mono,
                  fontSize: 10,
                  letterSpacing: "0.3em",
                  textTransform: "uppercase",
                  color: "#666",
                }}
              >
                Subtotal
              </span>
              <span
                style={{
                  fontFamily: display,
                  fontSize: 32,
                  fontWeight: 500,
                  letterSpacing: "-0.01em",
                  color: "#000",
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                ₱{total.toLocaleString()}
              </span>
            </div>
            <div
              style={{
                fontFamily: mono,
                fontSize: 9,
                letterSpacing: "0.22em",
                textTransform: "uppercase",
                color: "#666",
                marginBottom: 16,
              }}
            >
              Watermarked previews are free. Pay once, download forever.
            </div>
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
              Continue to checkout →
            </button>
            <div
              style={{
                marginTop: 14,
                display: "flex",
                justifyContent: "space-between",
                fontFamily: mono,
                fontSize: 9,
                letterSpacing: "0.22em",
                textTransform: "uppercase",
                color: "#666",
              }}
            >
              <span>← Keep browsing</span>
              <span>Clear cart</span>
            </div>
          </div>
        </aside>
      </div>
    </BrowserFrame>
  );
}

function CartRow({ bib, time, price, last }: { bib: string; time: string; price: number; last: boolean }) {
  return (
    <div
      style={{
        display: "flex",
        gap: 14,
        padding: "16px 24px",
        borderBottom: last ? "none" : "1px solid #DDDDDD",
        alignItems: "flex-start",
        background: "#fff",
        position: "relative",
      }}
    >
      <div style={{ position: "relative", width: 72, height: 72, flexShrink: 0 }}>
        <Hatch width={72} height={72} density={4} />
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontFamily: mono,
            fontSize: 8,
            letterSpacing: "0.22em",
            color: "#000",
            opacity: 0.32,
            fontWeight: 700,
            transform: "rotate(-18deg)",
          }}
        >
          QP/WM
        </div>
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div
          style={{
            fontFamily: mono,
            fontSize: 9,
            letterSpacing: "0.22em",
            textTransform: "uppercase",
            color: "#666",
          }}
        >
          Run Cebu 2026
        </div>
        <div
          style={{
            fontFamily: display,
            fontSize: 20,
            fontWeight: 500,
            letterSpacing: "-0.01em",
            color: "#000",
            marginTop: 2,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {bib}
        </div>
        <div
          style={{
            fontFamily: mono,
            fontSize: 9,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "#666",
            fontVariantNumeric: "tabular-nums",
            marginTop: 2,
          }}
        >
          {time}
        </div>
        <div
          style={{
            marginTop: 8,
            fontFamily: mono,
            fontSize: 9,
            letterSpacing: "0.22em",
            textTransform: "uppercase",
            color: "#666",
          }}
        >
          ✕ Remove
        </div>
      </div>
      <div
        style={{
          fontFamily: mono,
          fontSize: 12,
          fontVariantNumeric: "tabular-nums",
          color: "#000",
        }}
      >
        ₱{price}
      </div>
    </div>
  );
}
