import { PhoneFrame } from "../../_components/PhoneFrame";
import {
  AppBar,
  Body,
  Button,
  Chip,
  Frame,
  Hatch,
  Kicker,
  MonoNum,
} from "../../_components/WireUI";

export default function M1_1_TetherCamera() {
  return (
    <PhoneFrame ucId="UC-M1-1.1" screenName="Tether Camera Screen">
      <AppBar
        title="Connect Camera"
        trailing={<Chip>Event Setup</Chip>}
      />

      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 14 }}>
        {/* Active event banner */}
        <Frame padding={12} style={{ borderStyle: "dashed" }}>
          <Kicker>Active Event</Kicker>
          <div style={{ marginTop: 6, display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <Body size={14} weight={600}>Run Cebu 2026 — 21 K</Body>
            <MonoNum size={10} color="#666">EVT-0421</MonoNum>
          </div>
          <div style={{ marginTop: 4 }}>
            <Body size={11} color="#666">Photographer · Theo C. Chan</Body>
          </div>
        </Frame>

        {/* Camera illustration */}
        <Hatch height={150} label="Camera Illustration" />

        {/* Connection mode selector */}
        <div>
          <Kicker>Connection Mode</Kicker>
          <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
            <div style={{ flex: 1, border: "1.5px solid #000", padding: "10px 12px", background: "#000", color: "#fff" }}>
              <div style={{ fontFamily: "var(--font-mono), monospace", fontSize: 10, letterSpacing: "0.2em" }}>USB OTG</div>
              <div style={{ fontFamily: "var(--font-sans), sans-serif", fontSize: 10, marginTop: 2, opacity: 0.7 }}>Cable tether</div>
            </div>
            <div style={{ flex: 1, border: "1.5px solid #000", padding: "10px 12px" }}>
              <div style={{ fontFamily: "var(--font-mono), monospace", fontSize: 10, letterSpacing: "0.2em" }}>WI-FI</div>
              <div style={{ fontFamily: "var(--font-sans), sans-serif", fontSize: 10, marginTop: 2, color: "#666" }}>Vendor SDK</div>
            </div>
          </div>
        </div>

        {/* Primary CTA */}
        <Button variant="primary">Connect Camera</Button>

        {/* Connected session card (success state) */}
        <Frame padding={12}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <Kicker>Session · Active</Kicker>
            <MonoNum size={10}>02:14:08</MonoNum>
          </div>
          <div style={{ marginTop: 8, display: "flex", justifyContent: "space-between" }}>
            <div>
              <Body size={11} color="#666">Model</Body>
              <div><Body size={12} weight={600}>Canon EOS R7</Body></div>
            </div>
            <div style={{ textAlign: "right" }}>
              <Body size={11} color="#666">Battery</Body>
              <div><MonoNum size={12}>87%</MonoNum></div>
            </div>
            <div style={{ textAlign: "right" }}>
              <Body size={11} color="#666">Mode</Body>
              <div><MonoNum size={12}>USB</MonoNum></div>
            </div>
          </div>
        </Frame>

        {/* Trouble row */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingTop: 4 }}>
          <Body size={11} color="#666">Camera not detected?</Body>
          <Body size={11} weight={600} style={{ textDecoration: "underline" }}>Troubleshoot</Body>
        </div>
      </div>
    </PhoneFrame>
  );
}
