import type { ReactNode } from "react";

type DesktopFrameProps = {
  ucId: string;
  screenName: string;
  windowTitle?: string;
  children: ReactNode;
  width?: number;
  height?: number;
};

export function DesktopFrame({
  ucId,
  screenName,
  windowTitle = "BatchMyPhotos",
  children,
  width = 1100,
  height = 780,
}: DesktopFrameProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 20 }}>
      <div
        style={{
          width,
          height,
          border: "1.5px solid #000",
          borderRadius: 8,
          background: "#fff",
          overflow: "hidden",
          boxSizing: "border-box",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <TitleBar title={windowTitle} />
        <div style={{ flex: 1, background: "#fff", overflow: "hidden" }}>{children}</div>
      </div>

      <div style={{ textAlign: "center", fontFamily: "var(--font-mono), monospace" }}>
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

function TitleBar({ title }: { title: string }) {
  return (
    <div
      style={{
        height: 32,
        borderBottom: "1.5px solid #000",
        display: "flex",
        alignItems: "center",
        padding: "0 12px",
        boxSizing: "border-box",
        background: "#fff",
        gap: 12,
      }}
    >
      <span
        style={{
          fontFamily: "var(--font-mono), monospace",
          fontSize: 11,
          letterSpacing: "0.2em",
          textTransform: "uppercase",
          color: "#000",
        }}
      >
        {title}
      </span>
      <div style={{ marginLeft: "auto", display: "flex", gap: 10, alignItems: "center" }}>
        <WinBtn type="min" />
        <WinBtn type="max" />
        <WinBtn type="close" />
      </div>
    </div>
  );
}

function WinBtn({ type }: { type: "min" | "max" | "close" }) {
  const common = { width: 14, height: 14, display: "inline-flex" as const, alignItems: "center" as const, justifyContent: "center" as const };
  if (type === "min") {
    return (
      <span aria-hidden style={common}>
        <span style={{ display: "block", width: 10, height: 1.5, background: "#000" }} />
      </span>
    );
  }
  if (type === "max") {
    return <span aria-hidden style={{ ...common, border: "1.5px solid #000", width: 10, height: 10 }} />;
  }
  return (
    <span aria-hidden style={{ ...common, position: "relative" }}>
      <span
        style={{
          position: "absolute",
          width: 12,
          height: 1.5,
          background: "#000",
          transform: "rotate(45deg)",
        }}
      />
      <span
        style={{
          position: "absolute",
          width: 12,
          height: 1.5,
          background: "#000",
          transform: "rotate(-45deg)",
        }}
      />
    </span>
  );
}
