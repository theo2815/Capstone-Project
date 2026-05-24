import { DesktopFrame } from "../../_components/DesktopFrame";
import {
  Body,
  Button,
  Chip,
  Frame,
  Hatch,
  Kicker,
  MonoNum,
} from "../../_components/WireUI";

const mono = "var(--font-mono), ui-monospace, monospace";
const sans = "var(--font-sans), system-ui, sans-serif";

type BlurType = "PORTRAIT" | "GENERAL" | "MOTION";

const FLAGGED: Array<{
  name: string;
  score: number;
  type: BlurType;
  overridden?: boolean;
}> = [
  { name: "DSC_04127.JPG", score: 78.4, type: "MOTION" },
  { name: "DSC_04128.JPG", score: 102.3, type: "MOTION" },
  { name: "DSC_04131.JPG", score: 41.7, type: "PORTRAIT" },
  { name: "DSC_04138.JPG", score: 187.2, type: "GENERAL", overridden: true },
  { name: "DSC_04142.JPG", score: 64.0, type: "PORTRAIT" },
  { name: "DSC_04145.JPG", score: 33.9, type: "MOTION" },
];

export default function M2_2_BlurDetection() {
  return (
    <DesktopFrame
      ucId="UC-M2-2.2"
      screenName="Blur Analysis & Manual Override"
      windowTitle="BatchMyPhotos · RunCebu2026"
    >
      <AppHeader subtitle="Post-event culling — AI blur detection station" />

      <div
        style={{
          padding: "20px 32px 24px",
          height: "calc(100% - 88px)",
          overflow: "hidden",
          display: "flex",
          gap: 20,
        }}
      >
        {/* LEFT COLUMN — stats + culled list */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 14, minWidth: 0 }}>
          {/* Folder header */}
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
            <div>
              <span
                style={{
                  fontFamily: "var(--font-display), system-ui",
                  fontSize: 18,
                  fontWeight: 600,
                  color: "#000",
                  letterSpacing: "-0.01em",
                }}
              >
                RunCebu2026_Photos
              </span>
              <div style={{ marginTop: 2 }}>
                <span style={{ fontFamily: mono, fontSize: 10, color: "#666" }}>
                  D:\Events\RunCebu2026\
                </span>
              </div>
            </div>
            <Chip>Job · blur_b2c4a · 100%</Chip>
          </div>

          {/* Stats grid */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
            <StatCard label="Total Files" value="14,283" />
            <StatCard label="Clean" value="13,436" />
            <StatCard label="Culled" value="847" highlight />
          </div>

          {/* Tabs */}
          <div style={{ display: "flex", borderBottom: "1.5px solid #000" }}>
            <Tab label="Culled (847)" active />
            <Tab label="Clean (13,436)" />
            <Tab label="Failed (12)" />
          </div>

          {/* Culled list header */}
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
            <Kicker>Flagged Photos · Manual Override Enabled</Kicker>
            <Body size={10} color="#666">Showing 6 of 847</Body>
          </div>

          {/* List */}
          <div
            style={{
              border: "1.5px solid #000",
              flex: 1,
              overflow: "hidden",
              display: "flex",
              flexDirection: "column",
            }}
          >
            {/* Column headers */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "40px 1fr 80px 90px 110px",
                gap: 12,
                padding: "8px 12px",
                borderBottom: "1px solid #000",
                background: "#F2F2F2",
                alignItems: "center",
              }}
            >
              <span />
              <Kicker>Filename</Kicker>
              <Kicker style={{ textAlign: "right" }}>Score</Kicker>
              <Kicker>Type</Kicker>
              <Kicker style={{ textAlign: "right" }}>Override</Kicker>
            </div>

            {/* Rows */}
            <div style={{ flex: 1, overflow: "hidden" }}>
              {FLAGGED.map((f, i) => (
                <FlaggedRow key={f.name} {...f} last={i === FLAGGED.length - 1} />
              ))}
            </div>
          </div>

          {/* Failed scoring panel */}
          <Frame padding={10} style={{ borderStyle: "dashed" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <Kicker>Failed Scoring · 12 files</Kicker>
                <div style={{ marginTop: 2 }}>
                  <Body size={11}>Files that ai-api could not score — left at pending_blur_check.</Body>
                </div>
              </div>
              <Button variant="secondary" fullWidth={false}>Retry</Button>
            </div>
          </Frame>
        </div>

        {/* RIGHT COLUMN — controls */}
        <div style={{ width: 280, display: "flex", flexDirection: "column", gap: 14 }}>
          {/* Run Blur action card */}
          <Frame padding={14}>
            <Kicker>AI Blur Detection</Kicker>
            <div style={{ marginTop: 8, marginBottom: 10 }}>
              <Body size={11} color="#333">
                Submits the synced library to ai-api with scopes blur:read + jobs:read.
              </Body>
            </div>
            <Button variant="primary">Re-Detect Blur</Button>
            <div style={{ marginTop: 8, display: "flex", justifyContent: "space-between" }}>
              <Body size={10} color="#666">Last run · 2 min ago</Body>
              <MonoNum size={10} color="#666">847 / 14,283</MonoNum>
            </div>
          </Frame>

          {/* Threshold slider */}
          <Frame padding={14}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <Kicker>Blur Threshold</Kicker>
              <MonoNum size={14}>120</MonoNum>
            </div>
            <div style={{ marginTop: 10 }}>
              <Slider value={0.55} />
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                <Body size={9} color="#666">Lenient · 30</Body>
                <Body size={9} color="#666">Strict · 250</Body>
              </div>
            </div>
            <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px dashed #000" }}>
              <Body size={11} color="#333">
                Re-classifies locally without re-calling ai-api.
              </Body>
            </div>
          </Frame>

          {/* Blur type */}
          <Frame padding={14}>
            <Kicker>Blur Type</Kicker>
            <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
              <RadioRow label="General" />
              <RadioRow label="Portrait" checked />
              <RadioRow label="Motion" />
            </div>
          </Frame>

          {/* Cull summary */}
          <Frame padding={14}>
            <Kicker>Cull Summary</Kicker>
            <div style={{ marginTop: 8, display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <Body size={11} color="#666">Cull rate</Body>
              <MonoNum size={20}>5.93%</MonoNum>
            </div>
            <div style={{ marginTop: 4, display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <Body size={11} color="#666">Manual overrides</Body>
              <MonoNum size={14}>1</MonoNum>
            </div>
          </Frame>
        </div>
      </div>
    </DesktopFrame>
  );
}

function AppHeader({ subtitle }: { subtitle: string }) {
  return (
    <div
      style={{
        height: 56,
        borderBottom: "1.5px solid #000",
        display: "flex",
        alignItems: "center",
        padding: "0 32px",
        boxSizing: "border-box",
        background: "#fff",
        justifyContent: "space-between",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span
          aria-hidden
          style={{
            display: "inline-flex",
            width: 28,
            height: 28,
            border: "1.5px solid #000",
            borderRadius: 4,
            alignItems: "center",
            justifyContent: "center",
            fontFamily: mono,
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: "0.1em",
          }}
        >
          BMP
        </span>
        <div>
          <div
            style={{
              fontFamily: "var(--font-display), system-ui",
              fontSize: 16,
              fontWeight: 700,
              color: "#000",
              letterSpacing: "-0.01em",
            }}
          >
            BatchMyPhotos
          </div>
          <div style={{ fontFamily: sans, fontSize: 10, color: "#666", marginTop: 1 }}>{subtitle}</div>
        </div>
      </div>
      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        <IconBtn label="HIST" />
        <IconBtn label="THM" />
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "4px 10px",
            border: "1px solid #000",
            borderRadius: 999,
          }}
        >
          <span
            style={{
              width: 22,
              height: 22,
              borderRadius: 999,
              border: "1.5px solid #000",
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              fontFamily: mono,
              fontSize: 9,
              fontWeight: 700,
            }}
          >
            TC
          </span>
          <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: "0.1em", color: "#000" }}>PRO</span>
        </div>
      </div>
    </div>
  );
}

function IconBtn({ label }: { label: string }) {
  return (
    <span
      style={{
        width: 30,
        height: 26,
        border: "1px solid #000",
        borderRadius: 4,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: mono,
        fontSize: 9,
        letterSpacing: "0.1em",
        color: "#000",
      }}
    >
      {label}
    </span>
  );
}

function StatCard({ label, value, highlight = false }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div
      style={{
        border: "1.5px solid #000",
        padding: "10px 14px",
        background: highlight ? "#000" : "#fff",
        color: highlight ? "#fff" : "#000",
      }}
    >
      <div
        style={{
          fontFamily: mono,
          fontSize: 9,
          letterSpacing: "0.22em",
          textTransform: "uppercase",
          color: highlight ? "rgba(255,255,255,0.7)" : "#666",
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: mono,
          fontSize: 24,
          fontWeight: 600,
          marginTop: 4,
          fontVariantNumeric: "tabular-nums",
          letterSpacing: "-0.01em",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function Tab({ label, active = false }: { label: string; active?: boolean }) {
  return (
    <div
      style={{
        padding: "8px 14px",
        marginBottom: -1.5,
        borderBottom: active ? "2px solid #000" : "2px solid transparent",
        fontFamily: mono,
        fontSize: 11,
        letterSpacing: "0.16em",
        textTransform: "uppercase",
        color: active ? "#000" : "#666",
        fontWeight: active ? 600 : 400,
      }}
    >
      {label}
    </div>
  );
}

function FlaggedRow({
  name,
  score,
  type,
  overridden = false,
  last,
}: {
  name: string;
  score: number;
  type: BlurType;
  overridden?: boolean;
  last: boolean;
}) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "40px 1fr 80px 90px 110px",
        gap: 12,
        padding: "8px 12px",
        borderBottom: last ? "none" : "1px solid #000",
        alignItems: "center",
        background: overridden ? "#F7F7F7" : "#fff",
      }}
    >
      <Hatch width={30} height={30} density={3.5} />
      <div style={{ minWidth: 0 }}>
        <div style={{ fontFamily: mono, fontSize: 11, color: "#000" }}>{name}</div>
        {overridden && (
          <div style={{ fontFamily: mono, fontSize: 9, color: "#666", marginTop: 2 }}>
            ↶ MARKED CLEAN BY USER
          </div>
        )}
      </div>
      <div style={{ textAlign: "right" }}>
        <MonoNum size={12}>{score.toFixed(1)}</MonoNum>
      </div>
      <Chip filled={type === "MOTION"}>{type}</Chip>
      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <ToggleSwitch on={overridden} />
      </div>
    </div>
  );
}

function ToggleSwitch({ on }: { on: boolean }) {
  return (
    <div
      style={{
        position: "relative",
        width: 40,
        height: 20,
        border: "1.5px solid #000",
        borderRadius: 999,
        background: on ? "#000" : "#fff",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 1.5,
          left: on ? 21 : 1.5,
          width: 14,
          height: 14,
          background: on ? "#fff" : "#000",
          borderRadius: 999,
          transition: "left 0.15s ease",
        }}
      />
    </div>
  );
}

function Slider({ value }: { value: number }) {
  return (
    <div style={{ position: "relative", height: 16, display: "flex", alignItems: "center" }}>
      <div style={{ height: 2, background: "#000", width: "100%" }} />
      <div
        style={{
          position: "absolute",
          left: `calc(${value * 100}% - 7px)`,
          width: 14,
          height: 14,
          background: "#000",
          borderRadius: 999,
          border: "1.5px solid #000",
        }}
      />
    </div>
  );
}

function RadioRow({ label, checked = false }: { label: string; checked?: boolean }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <span
        style={{
          width: 14,
          height: 14,
          borderRadius: 999,
          border: "1.5px solid #000",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {checked && <span style={{ width: 6, height: 6, borderRadius: 999, background: "#000" }} />}
      </span>
      <span style={{ fontFamily: sans, fontSize: 12, color: "#000" }}>{label}</span>
    </div>
  );
}
