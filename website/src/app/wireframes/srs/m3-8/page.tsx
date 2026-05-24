import { BrowserFrame } from "../../_components/BrowserFrame";
import { Hatch } from "../../_components/WireUI";

const mono = "var(--font-mono), ui-monospace, monospace";
const sans = "var(--font-sans), system-ui, sans-serif";
const display = "var(--font-display), system-ui, sans-serif";

const BUNDLE_ITEMS: Array<{ bib: string; time: string; size: string }> = [
  { bib: "1247", time: "01:42:14", size: "15.8 MB" },
  { bib: "1247", time: "01:44:32", size: "16.1 MB" },
  { bib: "1247", time: "01:48:07", size: "15.3 MB" },
];

export default function M3_8_BundleDownload() {
  const total = BUNDLE_ITEMS.length;

  return (
    <BrowserFrame
      ucId="UC-3.3"
      screenName="Receipt & Bundle Download · Post-Payment"
      url="quickpitik.ph/orders/QP-2026-0318-1247"
    >
      <div
        style={{
          height: "100%",
          background: "#fff",
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
          boxSizing: "border-box",
        }}
      >
        <SiteHeader />

        <div
          style={{
            flex: 1,
            overflow: "hidden",
            padding: "28px 64px 0",
            display: "grid",
            gridTemplateColumns: "1fr 360px",
            gap: 40,
          }}
        >
          {/* LEFT — confirmation + items */}
          <main>
            <ConfirmationHeader />

            <section style={{ marginTop: 24 }}>
              <SectionLabel>Your Photos · {total} files</SectionLabel>
              <div
                style={{
                  marginTop: 10,
                  border: "1px solid #000",
                  borderRadius: 4,
                }}
              >
                {BUNDLE_ITEMS.map((it, i) => (
                  <PhotoRow
                    key={i}
                    {...it}
                    last={i === BUNDLE_ITEMS.length - 1}
                  />
                ))}
              </div>
              <FootNote>
                Watermark removed on download · Signed URL · expires in 14:52
              </FootNote>
            </section>
          </main>

          {/* RIGHT — bundle CTA + receipt + audit */}
          <aside style={{ display: "flex", flexDirection: "column", gap: 22 }}>
            <BundleCard total={total} />
            <ReceiptCard />
            <ShareTokenCard />
          </aside>
        </div>

        <NfrFooter />
      </div>
    </BrowserFrame>
  );
}

/* ─────────────────────────────────────────────────────────── */

function SiteHeader() {
  return (
    <div
      style={{
        height: 52,
        borderBottom: "1px solid #000",
        display: "flex",
        alignItems: "center",
        padding: "0 32px",
        boxSizing: "border-box",
        justifyContent: "space-between",
        background: "#fff",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 28 }}>
        <span
          style={{
            fontFamily: display,
            fontSize: 18,
            fontWeight: 700,
            letterSpacing: "-0.01em",
            color: "#000",
          }}
        >
          QuickPitik
        </span>
        <nav style={{ display: "flex", gap: 22 }}>
          {["Events", "Photographers", "How it works"].map((item) => (
            <span
              key={item}
              style={{
                fontFamily: mono,
                fontSize: 10,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                color: "#666",
              }}
            >
              {item}
            </span>
          ))}
        </nav>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <span
          style={{
            fontFamily: mono,
            fontSize: 10,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "#666",
          }}
        >
          Theo
        </span>
        <span
          aria-hidden
          style={{
            width: 28,
            height: 28,
            borderRadius: 999,
            border: "1.5px solid #000",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            fontFamily: mono,
            fontSize: 10,
            fontWeight: 700,
            color: "#000",
          }}
        >
          TC
        </span>
      </div>
    </div>
  );
}

function ConfirmationHeader() {
  return (
    <header>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <CheckBadge />
        <span
          style={{
            fontFamily: mono,
            fontSize: 10,
            letterSpacing: "0.3em",
            textTransform: "uppercase",
            color: "#000",
          }}
        >
          Payment Confirmed
        </span>
      </div>

      <h1
        style={{
          fontFamily: display,
          fontSize: 40,
          fontWeight: 500,
          color: "#000",
          margin: "10px 0 0",
          letterSpacing: "-0.02em",
          lineHeight: 1.1,
        }}
      >
        Your photos are ready.
      </h1>

      <div
        style={{
          marginTop: 12,
          display: "flex",
          gap: 20,
          alignItems: "baseline",
          flexWrap: "wrap",
        }}
      >
        <Meta label="Order" value="QP-2026-0318-1247" />
        <Meta label="Event" value="Run Cebu 2026" />
        <Meta label="Paid" value="₱150 · GCash" />
        <Meta label="Confirmed" value="08 Mar 2026 · 14:02:11" />
      </div>
    </header>
  );
}

function CheckBadge() {
  return (
    <span
      aria-hidden
      style={{
        width: 22,
        height: 22,
        borderRadius: 999,
        background: "#000",
        color: "#fff",
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: mono,
        fontSize: 12,
        fontWeight: 700,
      }}
    >
      ✓
    </span>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div
        style={{
          fontFamily: mono,
          fontSize: 9,
          letterSpacing: "0.22em",
          textTransform: "uppercase",
          color: "#666",
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: mono,
          fontSize: 12,
          color: "#000",
          marginTop: 2,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */

function PhotoRow({
  bib,
  time,
  size,
  last,
}: {
  bib: string;
  time: string;
  size: string;
  last: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        gap: 16,
        padding: "14px 16px",
        borderBottom: last ? "none" : "1px solid #DDDDDD",
        alignItems: "center",
        background: "#fff",
      }}
    >
      {/* Un-watermarked thumb (paid) */}
      <div style={{ position: "relative", width: 72, height: 54, flexShrink: 0 }}>
        <Hatch width={72} height={54} density={5} />
        <div
          style={{
            position: "absolute",
            top: 4,
            left: 4,
            padding: "1px 6px",
            background: "#000",
            color: "#fff",
            fontFamily: mono,
            fontSize: 8,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
          }}
        >
          Full Res
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
          Bib {bib} · {time}
        </div>
        <div
          style={{
            fontFamily: sans,
            fontSize: 13,
            color: "#000",
            marginTop: 2,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          IMG_{bib}_{time.replaceAll(":", "")}.jpg
        </div>
      </div>

      <div
        style={{
          fontFamily: mono,
          fontSize: 10,
          color: "#666",
          fontVariantNumeric: "tabular-nums",
          minWidth: 60,
          textAlign: "right",
        }}
      >
        {size}
      </div>

      <div
        style={{
          padding: "6px 12px",
          border: "1px solid #000",
          borderRadius: 999,
          fontFamily: mono,
          fontSize: 9,
          letterSpacing: "0.22em",
          textTransform: "uppercase",
          color: "#000",
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
        }}
      >
        <span aria-hidden>↓</span>
        <span>JPG</span>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */

function BundleCard({ total }: { total: number }) {
  return (
    <section
      style={{
        border: "1.5px solid #000",
        borderRadius: 10,
        padding: 20,
        background: "#fff",
      }}
    >
      <SectionLabel>Download All</SectionLabel>

      <div
        style={{
          marginTop: 8,
          fontFamily: display,
          fontSize: 24,
          fontWeight: 500,
          letterSpacing: "-0.01em",
          color: "#000",
          lineHeight: 1.15,
        }}
      >
        {total} photos · ZIP archive
      </div>
      <div
        style={{
          marginTop: 4,
          fontFamily: mono,
          fontSize: 10,
          color: "#666",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        47.2 MB · streams over HTTPS
      </div>

      <button
        type="button"
        style={{
          marginTop: 16,
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
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 10,
        }}
      >
        <span aria-hidden>↓</span>
        <span>Download ZIP</span>
      </button>

      <div
        style={{
          marginTop: 10,
          fontFamily: mono,
          fontSize: 9,
          letterSpacing: "0.22em",
          textTransform: "uppercase",
          color: "#666",
          textAlign: "center",
        }}
      >
        Ready in &lt; 10 s · NFR-P-9
      </div>
    </section>
  );
}

function ReceiptCard() {
  return (
    <section
      style={{
        border: "1px solid #000",
        borderRadius: 10,
        padding: 18,
        background: "#FAFAFA",
      }}
    >
      <SectionLabel>Receipt Sent</SectionLabel>
      <div
        style={{
          marginTop: 8,
          fontFamily: sans,
          fontSize: 13,
          color: "#000",
        }}
      >
        theo@quickpitik.ph
      </div>
      <div
        style={{
          marginTop: 4,
          fontFamily: mono,
          fontSize: 9,
          letterSpacing: "0.22em",
          textTransform: "uppercase",
          color: "#666",
        }}
      >
        Includes recovery link · Resend
      </div>
    </section>
  );
}

function ShareTokenCard() {
  return (
    <section
      style={{
        border: "1px dashed #000",
        borderRadius: 10,
        padding: 18,
        background: "#fff",
      }}
    >
      <SectionLabel>Guest Recovery Link</SectionLabel>
      <div
        style={{
          marginTop: 8,
          fontFamily: mono,
          fontSize: 10,
          color: "#000",
          background: "#fff",
          border: "1px solid #000",
          padding: "8px 10px",
          borderRadius: 4,
          wordBreak: "break-all",
          lineHeight: 1.45,
        }}
      >
        quickpitik.ph/r/QP-2026-0318-1247?t=…a9f3
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
        Per-order share token · Re-download anytime
      </div>
    </section>
  );
}

/* ─────────────────────────────────────────────────────────── */

function NfrFooter() {
  return (
    <div
      style={{
        borderTop: "1px solid #000",
        padding: "10px 64px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        fontFamily: mono,
        fontSize: 9,
        letterSpacing: "0.22em",
        textTransform: "uppercase",
        color: "#666",
        background: "#fff",
      }}
    >
      <span>UC-3.3 · anchors SO3.3</span>
      <span>JWT (runner) · share token (guest)</span>
      <span>Signed S3 URL · TTL ≤ 15 min</span>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
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

function FootNote({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        marginTop: 10,
        display: "flex",
        justifyContent: "space-between",
        fontFamily: mono,
        fontSize: 9,
        letterSpacing: "0.22em",
        textTransform: "uppercase",
        color: "#666",
      }}
    >
      <span>{children}</span>
    </div>
  );
}
