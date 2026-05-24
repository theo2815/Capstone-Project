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

const CACHED: Array<{ name: string; size: string }> = [
  { name: "img_0147.jpg", size: "8.2 MB" },
  { name: "img_0146.jpg", size: "7.9 MB" },
  { name: "img_0145.jpg", size: "8.4 MB" },
  { name: "img_0144.jpg", size: "8.0 MB" },
  { name: "img_0143.jpg", size: "7.6 MB" },
];

export default function M1_4_Offline() {
  return (
    <PhoneFrame ucId="UC-M1-1.4" screenName="Offline · Local-Cache State">
      <AppBar
        title="Upload Queue"
        trailing={<Chip>● OFFLINE</Chip>}
      />

      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
        {/* Offline banner */}
        <div
          style={{
            border: "1.5px solid #000",
            background: "#000",
            color: "#fff",
            padding: 14,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <div style={{ fontFamily: "var(--font-mono), monospace", fontSize: 10, letterSpacing: "0.22em" }}>
              ● OFFLINE — CACHING LOCALLY
            </div>
            <div style={{ fontFamily: "var(--font-mono), monospace", fontSize: 10 }}>00:04:21</div>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 12 }}>
            <div>
              <div style={{ fontFamily: "var(--font-sans), sans-serif", fontSize: 10, opacity: 0.6 }}>Cached on device</div>
              <div style={{ fontFamily: "var(--font-mono), monospace", fontSize: 22, fontVariantNumeric: "tabular-nums" }}>34</div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontFamily: "var(--font-sans), sans-serif", fontSize: 10, opacity: 0.6 }}>Storage remaining</div>
              <div style={{ fontFamily: "var(--font-mono), monospace", fontSize: 22, fontVariantNumeric: "tabular-nums" }}>3.8 GB</div>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div style={{ display: "flex", gap: 8 }}>
          <Button variant="primary" fullWidth>Retry Now</Button>
          <Button variant="secondary" fullWidth>Diagnostics</Button>
        </div>

        {/* Cached list */}
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <Kicker>Cached photos</Kicker>
            <MonoNum size={10} color="#666">5 of 34</MonoNum>
          </div>
          <div style={{ border: "1px solid #000", marginTop: 6 }}>
            {CACHED.map((c, i) => (
              <CachedRow key={c.name} {...c} last={i === CACHED.length - 1} />
            ))}
          </div>
        </div>

        {/* Storage-low dialog annotation */}
        <Frame padding={10} style={{ borderStyle: "dashed" }}>
          <Kicker>State Variant · E1</Kicker>
          <div style={{ marginTop: 4 }}>
            <Body size={11}>Free-space blocking dialog appears when storage &lt; 500 MB.</Body>
          </div>
          <div style={{ marginTop: 8, display: "flex", gap: 6 }}>
            <Chip>Block uploads</Chip>
            <Chip>List uploaded</Chip>
          </div>
        </Frame>
      </div>
    </PhoneFrame>
  );
}

function CachedRow({ name, size, last }: { name: string; size: string; last: boolean }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "8px 10px",
        borderBottom: last ? "none" : "1px solid #000",
      }}
    >
      <Hatch width={28} height={28} density={3.5} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontFamily: "var(--font-mono), monospace", fontSize: 11 }}>{name}</div>
        <div style={{ fontFamily: "var(--font-mono), monospace", fontSize: 9, color: "#666" }}>{size}</div>
      </div>
      <Chip>Cached</Chip>
    </div>
  );
}
