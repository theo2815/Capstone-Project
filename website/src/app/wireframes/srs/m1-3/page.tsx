import { PhoneFrame } from "../../_components/PhoneFrame";
import {
  AppBar,
  Body,
  Button,
  Chip,
  Hatch,
  Kicker,
  MonoNum,
} from "../../_components/WireUI";

const ROWS: Array<{
  name: string;
  size: string;
  state: "uploading" | "queued" | "uploaded" | "failed";
}> = [
  { name: "img_0147.jpg", size: "8.2 MB", state: "uploading" },
  { name: "img_0146.jpg", size: "7.9 MB", state: "uploading" },
  { name: "img_0145.jpg", size: "8.4 MB", state: "queued" },
  { name: "img_0144.jpg", size: "8.0 MB", state: "failed" },
  { name: "img_0143.jpg", size: "7.6 MB", state: "uploaded" },
  { name: "img_0142.jpg", size: "8.1 MB", state: "uploaded" },
  { name: "img_0141.jpg", size: "7.8 MB", state: "uploaded" },
];

const PROGRESS = 0.159; // 23 / 145

export default function M1_3_UploadQueue() {
  return (
    <PhoneFrame ucId="UC-M1-1.3" screenName="Upload Queue Screen">
      <AppBar
        title="Upload Queue"
        trailing={<Chip>23 / 145</Chip>}
      />

      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
        {/* ETA + rate */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <div>
            <Kicker>ETA Remaining</Kicker>
            <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginTop: 2 }}>
              <MonoNum size={32} style={{ letterSpacing: "-0.02em", lineHeight: 1 }}>~4</MonoNum>
              <Body size={11} color="#666">MIN</Body>
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <Kicker>Sync Rate</Kicker>
            <div style={{ marginTop: 2 }}>
              <MonoNum size={16}>3.2 ph/s</MonoNum>
            </div>
            <div style={{ marginTop: 4 }}>
              <Chip>Slow link</Chip>
            </div>
          </div>
        </div>

        {/* Progress bar */}
        <div>
          <div style={{ border: "1.5px solid #000", height: 14, position: "relative", boxSizing: "border-box" }}>
            <div style={{ width: `${PROGRESS * 100}%`, height: "100%", background: "#000" }} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
            <MonoNum size={10} color="#666">23 of 145 uploaded</MonoNum>
            <MonoNum size={10} color="#666">15.9%</MonoNum>
          </div>
        </div>

        {/* Filter chips */}
        <div style={{ display: "flex", gap: 6 }}>
          <Chip filled>All</Chip>
          <Chip>Uploading</Chip>
          <Chip>Queued</Chip>
          <Chip>Failed (1)</Chip>
        </div>

        {/* Row list */}
        <div style={{ border: "1px solid #000" }}>
          {ROWS.map((r, i) => (
            <Row key={r.name} {...r} last={i === ROWS.length - 1} />
          ))}
        </div>

        {/* Retry CTA */}
        <Button variant="secondary">Retry All Failed</Button>
      </div>
    </PhoneFrame>
  );
}

function Row({
  name,
  size,
  state,
  last,
}: {
  name: string;
  size: string;
  state: "uploading" | "queued" | "uploaded" | "failed";
  last: boolean;
}) {
  const stateLabel = {
    uploading: "UPLOADING",
    queued: "QUEUED",
    uploaded: "DONE",
    failed: "FAILED",
  }[state];
  const stateFilled = state === "uploading" || state === "uploaded";
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "8px 10px",
        borderBottom: last ? "none" : "1px solid #000",
        boxSizing: "border-box",
      }}
    >
      <Hatch width={32} height={32} density={3.5} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontFamily: "var(--font-mono), monospace", fontSize: 11 }}>{name}</div>
        <div style={{ fontFamily: "var(--font-mono), monospace", fontSize: 9, color: "#666" }}>{size}</div>
      </div>
      <Chip filled={stateFilled}>{stateLabel}</Chip>
      {state === "failed" && (
        <span
          aria-hidden
          style={{
            width: 16,
            height: 16,
            border: "1.5px solid #000",
            borderRadius: 999,
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            fontFamily: "var(--font-mono), monospace",
            fontSize: 10,
            fontWeight: 700,
          }}
        >
          ↻
        </span>
      )}
    </div>
  );
}
