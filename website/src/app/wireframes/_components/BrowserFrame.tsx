import type { ReactNode } from "react";

const mono = "var(--font-mono), ui-monospace, monospace";

type BrowserFrameProps = {
  ucId: string;
  screenName: string;
  url?: string;
  children: ReactNode;
  width?: number;
  height?: number;
};

export function BrowserFrame({
  ucId,
  screenName,
  url = "quickpitik.ph/events/run-cebu-2026",
  children,
  width = 1100,
  height = 780,
}: BrowserFrameProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 20 }}>
      <div
        style={{
          width,
          height,
          border: "1.5px solid #000",
          borderRadius: 10,
          background: "#fff",
          overflow: "hidden",
          boxSizing: "border-box",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <BrowserChrome url={url} />
        <div style={{ flex: 1, overflow: "hidden", background: "#fff", position: "relative" }}>{children}</div>
      </div>

      <div style={{ textAlign: "center", fontFamily: mono }}>
        <p
          style={{
            fontSize: 11,
            letterSpacing: "0.22em",
            textTransform: "uppercase",
            color: "#000",
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {ucId}
        </p>
        <p
          style={{
            fontSize: 10,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "#666",
            marginTop: 4,
          }}
        >
          {screenName}
        </p>
      </div>
    </div>
  );
}

function BrowserChrome({ url }: { url: string }) {
  return (
    <div
      style={{
        height: 64,
        borderBottom: "1.5px solid #000",
        background: "#fff",
        display: "flex",
        flexDirection: "column",
        boxSizing: "border-box",
      }}
    >
      {/* tab strip */}
      <div
        style={{
          height: 30,
          display: "flex",
          alignItems: "center",
          paddingLeft: 14,
          gap: 10,
          borderBottom: "1px solid #000",
        }}
      >
        <span aria-hidden style={{ width: 10, height: 10, borderRadius: 999, border: "1.5px solid #000" }} />
        <span aria-hidden style={{ width: 10, height: 10, borderRadius: 999, border: "1.5px solid #000" }} />
        <span aria-hidden style={{ width: 10, height: 10, borderRadius: 999, border: "1.5px solid #000" }} />
        <div
          style={{
            marginLeft: 12,
            padding: "4px 14px",
            border: "1.5px solid #000",
            borderBottom: "none",
            borderRadius: "6px 6px 0 0",
            fontFamily: mono,
            fontSize: 10,
            letterSpacing: "0.16em",
            textTransform: "uppercase",
            background: "#fff",
            color: "#000",
          }}
        >
          QuickPitik
        </div>
      </div>
      {/* url bar */}
      <div style={{ flex: 1, display: "flex", alignItems: "center", padding: "0 14px", gap: 10 }}>
        <NavButton dir="left" />
        <NavButton dir="right" />
        <span
          aria-hidden
          style={{
            width: 16,
            height: 16,
            borderRadius: 999,
            border: "1.5px solid #000",
            display: "inline-block",
          }}
        />
        <div
          style={{
            flex: 1,
            border: "1.5px solid #000",
            borderRadius: 999,
            height: 24,
            padding: "0 12px",
            display: "flex",
            alignItems: "center",
            gap: 8,
            fontFamily: mono,
            fontSize: 10,
            color: "#000",
            background: "#FAFAFA",
          }}
        >
          <span aria-hidden style={{ display: "inline-block", width: 8, height: 8, border: "1.5px solid #000" }} />
          <span>https://{url}</span>
        </div>
        <span
          aria-hidden
          style={{
            width: 24,
            height: 24,
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
      </div>
    </div>
  );
}

function NavButton({ dir }: { dir: "left" | "right" }) {
  return (
    <span
      aria-hidden
      style={{
        width: 18,
        height: 18,
        border: "1.5px solid #000",
        borderRadius: 999,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <span
        style={{
          width: 5,
          height: 5,
          borderRight: "1.5px solid #000",
          borderBottom: "1.5px solid #000",
          transform: dir === "right" ? "rotate(-45deg)" : "rotate(135deg)",
          marginLeft: dir === "right" ? -2 : 2,
        }}
      />
    </span>
  );
}
