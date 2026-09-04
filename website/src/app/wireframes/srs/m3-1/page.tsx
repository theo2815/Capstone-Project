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

export default function M3_1_MobileAuth() {
  return (
    <PhoneFrame ucId="UC-M3-1.1" screenName="Mobile Authentication & Verification">
      <AppBar
        title="Sign In / Register"
        trailing={<Chip>Auth Flow</Chip>}
      />

      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 14 }}>
        {/* Brand Kicker Header */}
        <Frame padding={12} style={{ borderStyle: "dashed" }}>
          <Kicker>QuickPitik Mobile</Kicker>
          <div style={{ marginTop: 6, display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <Body size={14} weight={600}>Marathon Photography</Body>
            <MonoNum size={10} color="#666">v1.0-mobile</MonoNum>
          </div>
          <div style={{ marginTop: 4 }}>
            <Body size={11} color="#666">Cebu Marathon Ecosystem</Body>
          </div>
        </Frame>

        {/* Auth form illustration */}
        <Hatch height={120} label="Auth & Deep-Link Wireframe" />

        {/* Role selection tab */}
        <div>
          <Kicker>Select Account Role</Kicker>
          <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
            <div style={{ flex: 1, border: "1.5px solid #000", padding: "10px 12px", background: "#000", color: "#fff" }}>
              <div style={{ fontFamily: "var(--font-mono), monospace", fontSize: 10, letterSpacing: "0.2em" }}>RUNNER</div>
              <div style={{ fontFamily: "var(--font-sans), sans-serif", fontSize: 10, marginTop: 2, opacity: 0.7 }}>Find & Buy Photos</div>
            </div>
            <div style={{ flex: 1, border: "1.5px solid #000", padding: "10px 12px" }}>
              <div style={{ fontFamily: "var(--font-mono), monospace", fontSize: 10, letterSpacing: "0.2em" }}>PHOTOGRAPHER</div>
              <div style={{ fontFamily: "var(--font-sans), sans-serif", fontSize: 10, marginTop: 2, color: "#666" }}>Tether & Sell</div>
            </div>
          </div>
        </div>

        {/* Form Fields */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <Frame padding={10}>
            <Kicker>Email Address</Kicker>
            <Body size={12} color="#333" style={{ marginTop: 4 }}>runner@quickpitik.com</Body>
          </Frame>
          <Frame padding={10}>
            <Kicker>Password</Kicker>
            <Body size={12} color="#333" style={{ marginTop: 4 }}>••••••••••••</Body>
          </Frame>
        </div>

        {/* Primary Submit Button */}
        <Button variant="primary">Sign In / Register</Button>

        {/* Deep link verification notification banner */}
        <Frame padding={12}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <Kicker>Email Verification</Kicker>
            <Chip>Deep Link</Chip>
          </div>
          <div style={{ marginTop: 6 }}>
            <Body size={11} color="#333">Tapping email link auto-routes to <b>VerifyEmailScreen</b></Body>
          </div>
        </Frame>
      </div>
    </PhoneFrame>
  );
}
