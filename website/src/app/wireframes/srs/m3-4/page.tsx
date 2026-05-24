import { BrowserFrame } from "../../_components/BrowserFrame";
import { EventBackdrop } from "../../_components/EventBackdrop";
import { Hatch } from "../../_components/WireUI";

const mono = "var(--font-mono), ui-monospace, monospace";
const sans = "var(--font-sans), system-ui, sans-serif";
const display = "var(--font-display), system-ui, sans-serif";

export default function M3_4_BibSearch() {
  return (
    <BrowserFrame
      ucId="UC-M3-3.4"
      screenName="Search by Bib Number · AI OCR"
      url="quickpitik.ph/events/run-cebu-2026?search=bib&q=1247"
    >
      <EventBackdrop activeTab="bib" panel={<BibPanel />} />
    </BrowserFrame>
  );
}

function BibPanel() {
  return (
    <div style={{ paddingTop: 18, display: "flex", flexDirection: "column", gap: 18 }}>
      {/* Input row */}
      <div
        style={{
          display: "flex",
          gap: 10,
          alignItems: "center",
          padding: 18,
          border: "1.5px solid #000",
          background: "#FAFAFA",
        }}
      >
        <div style={{ flex: 1 }}>
          <div
            style={{
              fontFamily: mono,
              fontSize: 10,
              letterSpacing: "0.22em",
              textTransform: "uppercase",
              color: "#666",
            }}
          >
            Enter Bib Number
          </div>
          <div
            style={{
              marginTop: 6,
              border: "1.5px solid #000",
              background: "#fff",
              padding: "12px 16px",
              fontFamily: mono,
              fontSize: 32,
              fontVariantNumeric: "tabular-nums",
              letterSpacing: "0.04em",
              color: "#000",
              display: "flex",
              alignItems: "center",
              gap: 10,
            }}
          >
            <span>1247</span>
            <span
              aria-hidden
              style={{
                display: "inline-block",
                width: 2,
                height: 28,
                background: "#000",
                animation: "blink 1s steps(2) infinite",
              }}
            />
          </div>
          <div
            style={{
              marginTop: 6,
              fontFamily: mono,
              fontSize: 9,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              color: "#666",
            }}
          >
            3–6 digits · numbers only
          </div>
        </div>
        <button
          type="button"
          style={{
            padding: "14px 24px",
            borderRadius: 999,
            background: "#000",
            color: "#fff",
            fontFamily: mono,
            fontSize: 12,
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            border: "1.5px solid #000",
            cursor: "default",
          }}
        >
          Search →
        </button>
      </div>

      {/* Results header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <div>
          <span
            style={{
              fontFamily: mono,
              fontSize: 10,
              letterSpacing: "0.22em",
              textTransform: "uppercase",
              color: "#666",
            }}
          >
            Bib · 1247
          </span>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginTop: 4 }}>
            <span
              style={{
                fontFamily: display,
                fontSize: 28,
                fontWeight: 600,
                letterSpacing: "-0.01em",
                color: "#000",
              }}
            >
              18 photos
            </span>
            <span
              style={{
                fontFamily: mono,
                fontSize: 10,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                padding: "3px 8px",
                border: "1px solid #000",
                color: "#000",
              }}
            >
              OCR match · 0.6 s
            </span>
          </div>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <Pill label="Sort · Time ↑" />
          <Pill label="Add all to cart" filled />
        </div>
      </div>

      {/* Results grid 4x2 */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10 }}>
        {Array.from({ length: 10 }).map((_, i) => (
          <ResultCard key={i} bib="1247" time={timeFor(i)} />
        ))}
      </div>

      {/* Footer hint */}
      <div
        style={{
          marginTop: 4,
          display: "flex",
          justifyContent: "space-between",
          fontFamily: mono,
          fontSize: 10,
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          color: "#666",
        }}
      >
        <span>Showing 10 of 18 · scroll for more</span>
        <span>SO3.1 · bib OCR ≥ 85%</span>
      </div>
    </div>
  );
}

function timeFor(i: number) {
  const base = 41 + i * 2;
  return `01:${String(base).padStart(2, "0")}:14`;
}

function ResultCard({ bib, time }: { bib: string; time: string }) {
  return (
    <div style={{ position: "relative" }}>
      <Hatch width="100%" height={120} density={4} />
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: mono,
          fontSize: 14,
          letterSpacing: "0.22em",
          color: "#000",
          opacity: 0.32,
          fontWeight: 700,
          transform: "rotate(-16deg)",
        }}
      >
        QP / WM
      </div>
      <div
        style={{
          position: "absolute",
          top: 6,
          left: 6,
          padding: "2px 6px",
          background: "#000",
          color: "#fff",
          fontFamily: mono,
          fontSize: 9,
          letterSpacing: "0.12em",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        Bib {bib}
      </div>
      <div
        style={{
          marginTop: 6,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span
          style={{
            fontFamily: mono,
            fontSize: 9,
            color: "#666",
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {time}
        </span>
        <span
          style={{
            fontFamily: mono,
            fontSize: 9,
            color: "#000",
            border: "1px solid #000",
            padding: "1px 6px",
          }}
        >
          + Cart
        </span>
      </div>
    </div>
  );
}

function Pill({ label, filled = false }: { label: string; filled?: boolean }) {
  return (
    <span
      style={{
        padding: "4px 12px",
        border: "1.5px solid #000",
        borderRadius: 999,
        fontFamily: mono,
        fontSize: 9,
        letterSpacing: "0.18em",
        textTransform: "uppercase",
        color: filled ? "#fff" : "#000",
        background: filled ? "#000" : "#fff",
      }}
    >
      {label}
    </span>
  );
}
