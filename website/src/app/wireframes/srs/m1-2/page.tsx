import { PhoneFrame } from "../../_components/PhoneFrame";
import {
  AppBar,
  Body,
  Chip,
  Frame,
  Hatch,
  Kicker,
  MonoNum,
} from "../../_components/WireUI";

const THUMBS: Array<{ label: string; state: "uploaded" | "uploading" | "queued" | "failed" }> = [
  { label: "img_0147", state: "uploading" },
  { label: "img_0146", state: "queued" },
  { label: "img_0145", state: "failed" },
  { label: "img_0144", state: "uploaded" },
  { label: "img_0143", state: "uploaded" },
];

export default function M1_2_Capture() {
  return (
    <PhoneFrame ucId="UC-M1-1.2" screenName="Capture & Queue Screen">
      <AppBar title="Capture · Live" trailing={<Chip filled>● REC</Chip>} />

      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 14 }}>
        {/* Hero counter */}
        <div>
          <Kicker>Photos Captured</Kicker>
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginTop: 2 }}>
            <MonoNum size={48} style={{ letterSpacing: "-0.02em", lineHeight: 1 }}>147</MonoNum>
            <div style={{ textAlign: "right" }}>
              <div><Body size={11} color="#666">Session</Body></div>
              <MonoNum size={14}>02:14</MonoNum>
            </div>
          </div>
        </div>

        {/* Counter row */}
        <div style={{ display: "flex", borderTop: "1px solid #000", borderBottom: "1px solid #000" }}>
          <div style={{ flex: 1, padding: "10px 0", borderRight: "1px solid #000" }}>
            <Kicker>In Queue</Kicker>
            <div style={{ marginTop: 2 }}><MonoNum size={20}>23</MonoNum></div>
          </div>
          <div style={{ flex: 1, padding: "10px 0 10px 12px", borderRight: "1px solid #000" }}>
            <Kicker>Uploaded</Kicker>
            <div style={{ marginTop: 2 }}><MonoNum size={20}>124</MonoNum></div>
          </div>
          <div style={{ flex: 1, padding: "10px 0 10px 12px" }}>
            <Kicker>Failed</Kicker>
            <div style={{ marginTop: 2 }}><MonoNum size={20}>0</MonoNum></div>
          </div>
        </div>

        {/* Last capture preview */}
        <div>
          <Kicker>Last Capture</Kicker>
          <div style={{ marginTop: 6 }}>
            <Hatch height={150} label="img_0147.jpg" />
          </div>
        </div>

        {/* Thumb strip */}
        <div>
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
            <Kicker>Recent</Kicker>
            <Body size={10} color="#666">Swipe →</Body>
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(5, 1fr)",
              gap: 6,
              marginTop: 6,
            }}
          >
            {THUMBS.map((t) => (
              <ThumbCell key={t.label} state={t.state} />
            ))}
          </div>
        </div>

        {/* Storage warning */}
        <Frame padding={10} style={{ borderStyle: "dashed" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <Kicker style={{ color: "#000" }}>⚠ Storage Low</Kicker>
              <div style={{ marginTop: 2 }}>
                <Body size={11}>487 MB free · </Body>
                <Body size={11} weight={600} style={{ textDecoration: "underline" }}>Free up space</Body>
              </div>
            </div>
            <MonoNum size={11}>32GB / 31.5GB</MonoNum>
          </div>
        </Frame>
      </div>
    </PhoneFrame>
  );
}

function ThumbCell({ state }: { state: "uploaded" | "uploading" | "queued" | "failed" }) {
  const badge = {
    uploaded: { text: "✓", filled: true },
    uploading: { text: "↑", filled: true },
    queued: { text: "·", filled: false },
    failed: { text: "!", filled: false },
  }[state];
  return (
    <div style={{ position: "relative" }}>
      <Hatch height={56} density={4} />
      <div
        style={{
          position: "absolute",
          top: 3,
          right: 3,
          width: 14,
          height: 14,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: badge.filled ? "#000" : "#fff",
          color: badge.filled ? "#fff" : "#000",
          border: "1px solid #000",
          fontFamily: "var(--font-mono), monospace",
          fontSize: 9,
          fontWeight: 700,
        }}
      >
        {badge.text}
      </div>
    </div>
  );
}
