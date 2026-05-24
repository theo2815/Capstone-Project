import { PhoneFrame } from "../../_components/PhoneFrame";
import {
  AppBar,
  Body,
  Chip,
  Hatch,
  Kicker,
  MonoNum,
} from "../../_components/WireUI";

export default function M1_5_NotificationGallery() {
  return (
    <PhoneFrame
      ucId="UC-M1-1.5"
      screenName="Photo-Found Notification + Filtered Gallery"
    >
      <AppBar
        title="Run Cebu 2026"
        trailing={
          <span style={{ position: "relative", display: "inline-flex", alignItems: "center" }}>
            <span
              aria-hidden
              style={{
                width: 18,
                height: 14,
                border: "1.5px solid #000",
                borderRadius: 2,
                display: "inline-block",
                position: "relative",
              }}
            />
            <span
              style={{
                position: "absolute",
                top: -5,
                right: -6,
                width: 14,
                height: 14,
                borderRadius: 999,
                background: "#000",
                color: "#fff",
                fontFamily: "var(--font-mono), monospace",
                fontSize: 9,
                fontWeight: 700,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              3
            </span>
          </span>
        }
      />

      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
        {/* System notification banner */}
        <div
          style={{
            border: "1.5px solid #000",
            padding: 12,
            background: "#fff",
            position: "relative",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div
                style={{
                  width: 24,
                  height: 24,
                  background: "#000",
                  color: "#fff",
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontFamily: "var(--font-mono), monospace",
                  fontSize: 11,
                  fontWeight: 700,
                }}
              >
                QP
              </div>
              <div>
                <div style={{ fontFamily: "var(--font-mono), monospace", fontSize: 9, letterSpacing: "0.22em", color: "#666" }}>
                  QUICKPITIK · NOW
                </div>
                <div style={{ marginTop: 2 }}>
                  <Body size={12} weight={600}>3 new matches</Body>
                </div>
              </div>
            </div>
            <span
              aria-hidden
              style={{
                width: 8,
                height: 8,
                borderRight: "1.5px solid #000",
                borderBottom: "1.5px solid #000",
                transform: "rotate(-45deg)",
              }}
            />
          </div>
          <div style={{ marginTop: 6 }}>
            <Body size={11} color="#333">Run Cebu 2026 — tap to view your photos.</Body>
          </div>
          <div
            style={{
              position: "absolute",
              top: -8,
              left: 10,
              background: "#fff",
              padding: "0 4px",
              fontFamily: "var(--font-mono), monospace",
              fontSize: 8,
              letterSpacing: "0.22em",
              color: "#666",
            }}
          >
            LOCK-SCREEN / IN-APP TOAST
          </div>
        </div>

        {/* Gallery header */}
        <div>
          <Kicker>Your Matches</Kicker>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: 2 }}>
            <MonoNum size={28} style={{ letterSpacing: "-0.02em", lineHeight: 1 }}>3 photos</MonoNum>
            <Chip filled>Watermarked</Chip>
          </div>
        </div>

        {/* Filtered gallery 2x3 */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          {[1, 2, 3, 4].map((i) => (
            <GalleryCell key={i} index={i} isNew={i <= 3} />
          ))}
        </div>

        {/* Inbox fallback annotation */}
        <div
          style={{
            border: "1px dashed #000",
            padding: 10,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div>
            <Kicker>Fallback</Kicker>
            <div style={{ marginTop: 2 }}>
              <Body size={11}>If OS notifications are disabled, matches appear in the in-app inbox.</Body>
            </div>
          </div>
        </div>
      </div>
    </PhoneFrame>
  );
}

function GalleryCell({ index, isNew }: { index: number; isNew: boolean }) {
  return (
    <div style={{ position: "relative" }}>
      <Hatch height={120} density={4.5} />
      {/* Watermark overlay */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "var(--font-mono), monospace",
          fontSize: 22,
          letterSpacing: "0.2em",
          color: "#000",
          opacity: 0.35,
          fontWeight: 700,
          transform: "rotate(-18deg)",
        }}
      >
        QP / WM
      </div>
      {/* New badge */}
      {isNew && (
        <div style={{ position: "absolute", top: 6, left: 6 }}>
          <Chip filled style={{ padding: "2px 6px", fontSize: 8 }}>New</Chip>
        </div>
      )}
      {/* Photo id */}
      <div
        style={{
          position: "absolute",
          bottom: 4,
          right: 6,
          fontFamily: "var(--font-mono), monospace",
          fontSize: 9,
          color: "#000",
          background: "rgba(255,255,255,0.85)",
          padding: "1px 4px",
        }}
      >
        #021{index}
      </div>
    </div>
  );
}
