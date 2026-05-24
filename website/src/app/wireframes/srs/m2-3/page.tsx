import { DesktopFrame } from "../../_components/DesktopFrame";
import {
  Body,
  Button,
  Chip,
  Frame,
  Kicker,
  MonoNum,
} from "../../_components/WireUI";

const mono = "var(--font-mono), ui-monospace, monospace";
const sans = "var(--font-sans), system-ui, sans-serif";

const BATCHES = [
  { name: "Batch_001", files: 500 },
  { name: "Batch_002", files: 500 },
  { name: "Batch_003", files: 500 },
  { name: "Batch_004", files: 500 },
  { name: "Batch_005", files: 487 },
];

export default function M2_3_BatchSort() {
  return (
    <DesktopFrame
      ucId="UC-M2-2.3"
      screenName="Batch Sort & Folder Output"
      windowTitle="BatchMyPhotos · RunCebu2026"
    >
      <AppHeader subtitle="Post-event culling — sort clean subset into batch folders" />

      <div
        style={{
          padding: "20px 32px 24px",
          height: "calc(100% - 88px)",
          overflow: "hidden",
          display: "flex",
          gap: 20,
        }}
      >
        {/* LEFT — folder + stats + sort settings */}
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
            <Chip>Sort · complete · 00:08.4</Chip>
          </div>

          {/* Stats */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
            <StatCard label="Clean Files" value="13,436" />
            <StatCard label="Batches" value="28" highlight />
            <StatCard label="Per Batch" value="500" />
          </div>

          {/* Settings panel — sort config */}
          <Frame padding={14}>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 12 }}>
              <Kicker>Sort Settings</Kicker>
              <Chip>Preset · Marathon Default</Chip>
            </div>

            <SettingRow label="Max Photos Per Batch">
              <Input value="500" hint="default" />
            </SettingRow>
            <SettingRow label="Folder Name Prefix">
              <Input value="Batch" hint="" />
            </SettingRow>
            <SettingRow label="Sort Photos By">
              <Select value="Date · Capture Time (ascending)" />
            </SettingRow>
            <SettingRow label="Route Blurry Photos">
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <ToggleSwitch on />
                <Body size={11} color="#666">847 culled → Blurry/</Body>
              </div>
            </SettingRow>
            <SettingRow label="Sort Mode" last>
              <SegmentedControl options={["Move", "Hardlink", "Copy"]} active={1} />
            </SettingRow>
          </Frame>

          {/* Confirmation */}
          <Frame padding={12}>
            <Body size={12} color="#000">
              28 folders created — <strong>Batch_001</strong> through <strong>Batch_028</strong> — under{" "}
              <span style={{ fontFamily: mono, fontSize: 11 }}>D:\Events\RunCebu2026\Output\</span>
            </Body>
            <div style={{ marginTop: 4 }}>
              <Body size={10} color="#666">
                Filesystem-level hardlinks · originals untouched · ready for upload (M2.4).
              </Body>
            </div>
          </Frame>

          {/* Actions */}
          <div style={{ display: "flex", gap: 10 }}>
            <Button variant="secondary" fullWidth>Re-Sort After Overrides</Button>
            <Button variant="primary" fullWidth>Continue to Upload →</Button>
          </div>
        </div>

        {/* RIGHT — folder tree preview */}
        <div style={{ width: 320, display: "flex", flexDirection: "column", gap: 10 }}>
          <Kicker>Output Folder Tree</Kicker>
          <div
            style={{
              flex: 1,
              border: "1.5px solid #000",
              padding: "12px 14px",
              background: "#FAFAFA",
              overflow: "hidden",
              fontFamily: mono,
              fontSize: 11,
              color: "#000",
              boxSizing: "border-box",
            }}
          >
            <TreeRow icon="▾" label="RunCebu2026\" bold />
            <TreeRow indent={1} icon="▾" label="Output\" />
            {BATCHES.map((b) => (
              <TreeRow
                key={b.name}
                indent={2}
                icon="▸"
                label={`${b.name}\\`}
                trailing={`${b.files} files`}
              />
            ))}
            <TreeRow indent={2} icon=" " label="…" muted />
            <TreeRow indent={2} icon="▸" label="Batch_028\" trailing="436 files" />
            <TreeRow indent={1} icon="▸" label="Blurry\" trailing="847 files" muted />
            <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px dashed #000" }}>
              <div
                style={{
                  fontFamily: mono,
                  fontSize: 9,
                  letterSpacing: "0.22em",
                  textTransform: "uppercase",
                  color: "#666",
                }}
              >
                Summary
              </div>
              <div style={{ marginTop: 6, display: "flex", justifyContent: "space-between" }}>
                <Body size={11} color="#000">Batches created</Body>
                <MonoNum size={11}>28</MonoNum>
              </div>
              <div style={{ marginTop: 2, display: "flex", justifyContent: "space-between" }}>
                <Body size={11} color="#000">Photos sorted</Body>
                <MonoNum size={11}>13,436</MonoNum>
              </div>
              <div style={{ marginTop: 2, display: "flex", justifyContent: "space-between" }}>
                <Body size={11} color="#000">Routed to Blurry/</Body>
                <MonoNum size={11}>847</MonoNum>
              </div>
              <div style={{ marginTop: 2, display: "flex", justifyContent: "space-between" }}>
                <Body size={11} color="#000">Elapsed time</Body>
                <MonoNum size={11}>00:08.4</MonoNum>
              </div>
            </div>
          </div>
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

function SettingRow({ label, children, last = false }: { label: string; children: React.ReactNode; last?: boolean }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "8px 0",
        borderBottom: last ? "none" : "1px solid #E0E0E0",
        gap: 16,
      }}
    >
      <span style={{ fontFamily: sans, fontSize: 12, color: "#000" }}>{label}</span>
      {children}
    </div>
  );
}

function Input({ value, hint }: { value: string; hint: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      {hint && (
        <span style={{ fontFamily: mono, fontSize: 9, color: "#666", letterSpacing: "0.12em", textTransform: "uppercase" }}>
          {hint}
        </span>
      )}
      <div
        style={{
          border: "1.5px solid #000",
          padding: "5px 10px",
          minWidth: 90,
          textAlign: "right",
          fontFamily: mono,
          fontSize: 12,
          color: "#000",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function Select({ value }: { value: string }) {
  return (
    <div
      style={{
        border: "1.5px solid #000",
        padding: "5px 10px",
        minWidth: 220,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 10,
        fontFamily: sans,
        fontSize: 12,
        color: "#000",
      }}
    >
      <span>{value}</span>
      <span
        aria-hidden
        style={{
          width: 7,
          height: 7,
          borderRight: "1.5px solid #000",
          borderBottom: "1.5px solid #000",
          transform: "rotate(45deg)",
          marginTop: -3,
        }}
      />
    </div>
  );
}

function SegmentedControl({ options, active }: { options: string[]; active: number }) {
  return (
    <div style={{ display: "inline-flex", border: "1.5px solid #000" }}>
      {options.map((opt, i) => (
        <div
          key={opt}
          style={{
            padding: "5px 12px",
            borderLeft: i === 0 ? "none" : "1.5px solid #000",
            background: i === active ? "#000" : "#fff",
            color: i === active ? "#fff" : "#000",
            fontFamily: mono,
            fontSize: 11,
            letterSpacing: "0.12em",
            textTransform: "uppercase",
          }}
        >
          {opt}
        </div>
      ))}
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
        }}
      />
    </div>
  );
}

function TreeRow({
  icon,
  label,
  trailing,
  indent = 0,
  bold = false,
  muted = false,
}: {
  icon: string;
  label: string;
  trailing?: string;
  indent?: number;
  bold?: boolean;
  muted?: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        paddingLeft: indent * 14,
        paddingTop: 2,
        paddingBottom: 2,
        color: muted ? "#666" : "#000",
        fontWeight: bold ? 600 : 400,
      }}
    >
      <span style={{ display: "flex", gap: 6 }}>
        <span style={{ display: "inline-block", width: 10 }}>{icon}</span>
        <span>{label}</span>
      </span>
      {trailing && (
        <span style={{ fontFamily: mono, fontSize: 10, color: muted ? "#666" : "#444", fontVariantNumeric: "tabular-nums" }}>
          {trailing}
        </span>
      )}
    </div>
  );
}
