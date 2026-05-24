import { BrowserFrame } from "../../_components/BrowserFrame";
import { EventBackdrop } from "../../_components/EventBackdrop";
import { Hatch } from "../../_components/WireUI";

const mono = "var(--font-mono), ui-monospace, monospace";
const sans = "var(--font-sans), system-ui, sans-serif";
const display = "var(--font-display), system-ui, sans-serif";

export default function M3_3_SelfieSearch() {
  return (
    <BrowserFrame
      ucId="UC-M3-3.3"
      screenName="Search by Selfie · AI Face Match"
      url="quickpitik.ph/events/run-cebu-2026?search=face"
    >
      <EventBackdrop
        activeTab="face"
        panel={<SelfiePanel />}
      />
    </BrowserFrame>
  );
}

function SelfiePanel() {
  return (
    <div style={{ paddingTop: 18 }}>
      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 24 }}>
        {/* LEFT — selfie capture */}
        <div>
          <div
            style={{
              fontFamily: mono,
              fontSize: 10,
              letterSpacing: "0.22em",
              textTransform: "uppercase",
              color: "#666",
              marginBottom: 8,
            }}
          >
            Step 1 · Reference Selfie
          </div>

          {/* Selfie preview */}
          <div style={{ position: "relative" }}>
            <Hatch width="100%" height={240} label="Selfie Preview" />
            <span
              aria-hidden
              style={{
                position: "absolute",
                top: 8,
                right: 8,
                width: 22,
                height: 22,
                borderRadius: 999,
                background: "#fff",
                border: "1.5px solid #000",
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                fontFamily: mono,
                fontSize: 11,
                fontWeight: 700,
              }}
            >
              ✕
            </span>
            <div
              style={{
                position: "absolute",
                bottom: 8,
                left: 8,
                padding: "3px 8px",
                background: "#fff",
                border: "1px solid #000",
                fontFamily: mono,
                fontSize: 9,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
              }}
            >
              Face Detected
            </div>
          </div>

          {/* Source buttons */}
          <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
            <ActionButton label="Upload Photo" icon="↑" filled />
            <ActionButton label="Use Webcam" icon="◉" />
          </div>

          {/* Primary CTA */}
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
              fontSize: 12,
              letterSpacing: "0.2em",
              textTransform: "uppercase",
              border: "1.5px solid #000",
              cursor: "default",
            }}
          >
            Search Photos →
          </button>

          {/* Consent reminder */}
          <div
            style={{
              marginTop: 12,
              padding: 10,
              border: "1px dashed #000",
              fontFamily: sans,
              fontSize: 11,
              color: "#333",
              lineHeight: 1.45,
            }}
          >
            <span
              style={{
                display: "inline-block",
                fontFamily: mono,
                fontSize: 9,
                letterSpacing: "0.22em",
                textTransform: "uppercase",
                color: "#000",
                marginBottom: 4,
              }}
            >
              RA 10173 · Privacy
            </span>
            <div>Your selfie is matched against this event only and is not stored beyond the session unless you enrol.</div>
          </div>
        </div>

        {/* RIGHT — results */}
        <div>
          <div
            style={{
              display: "flex",
              alignItems: "baseline",
              justifyContent: "space-between",
              marginBottom: 10,
            }}
          >
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
                Step 2 · Matches
              </span>
              <div style={{ marginTop: 4, display: "flex", alignItems: "baseline", gap: 10 }}>
                <span
                  style={{
                    fontFamily: display,
                    fontSize: 26,
                    fontWeight: 600,
                    letterSpacing: "-0.01em",
                    color: "#000",
                  }}
                >
                  12 photos
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
                  Face match · 0.4 s
                </span>
              </div>
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <Pill label="Sort · Confidence ↓" />
              <Pill label="Bib · All" />
            </div>
          </div>

          {/* Results grid */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
            {RESULTS.map((r, i) => (
              <ResultCard key={i} bib={r.bib} score={r.score} />
            ))}
          </div>

          {/* Confidence note */}
          <div
            style={{
              marginTop: 12,
              display: "flex",
              justifyContent: "space-between",
              fontFamily: mono,
              fontSize: 10,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              color: "#666",
            }}
          >
            <span>Threshold · 0.55 (event default)</span>
            <span>SO3.1 · combined ≥ 85%</span>
          </div>
        </div>
      </div>
    </div>
  );
}

const RESULTS: Array<{ bib: string; score: number }> = [
  { bib: "1247", score: 0.94 },
  { bib: "1247", score: 0.91 },
  { bib: "1247", score: 0.88 },
  { bib: "1247", score: 0.83 },
  { bib: "1247", score: 0.78 },
  { bib: "1247", score: 0.72 },
  { bib: "—", score: 0.69 },
  { bib: "1247", score: 0.61 },
];

function ResultCard({ bib, score }: { bib: string; score: number }) {
  return (
    <div style={{ position: "relative" }}>
      <Hatch width="100%" height={120} density={4} />
      {/* Watermark */}
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
      {/* Score chip */}
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
        {(score * 100).toFixed(0)}%
      </div>
      {/* Footer */}
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
            fontSize: 10,
            letterSpacing: "0.12em",
            color: "#000",
            fontVariantNumeric: "tabular-nums",
          }}
        >
          Bib {bib}
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

function ActionButton({ label, icon, filled = false }: { label: string; icon: string; filled?: boolean }) {
  return (
    <div
      style={{
        flex: 1,
        padding: "10px 12px",
        border: "1.5px solid #000",
        background: filled ? "#000" : "#fff",
        color: filled ? "#fff" : "#000",
        fontFamily: mono,
        fontSize: 10,
        letterSpacing: "0.16em",
        textTransform: "uppercase",
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}
    >
      <span style={{ fontSize: 14 }}>{icon}</span>
      <span>{label}</span>
    </div>
  );
}

function Pill({ label }: { label: string }) {
  return (
    <span
      style={{
        padding: "3px 10px",
        border: "1px solid #000",
        borderRadius: 999,
        fontFamily: mono,
        fontSize: 9,
        letterSpacing: "0.18em",
        textTransform: "uppercase",
        color: "#000",
      }}
    >
      {label}
    </span>
  );
}
