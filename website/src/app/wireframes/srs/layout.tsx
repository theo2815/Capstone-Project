import type { ReactNode } from "react";

export default function WireframesSrsLayout({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#F2F2F2",
        color: "#000",
        padding: "48px 24px 64px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 1200,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 32,
          fontFamily: "var(--font-mono), ui-monospace, monospace",
          fontSize: 10,
          letterSpacing: "0.22em",
          textTransform: "uppercase",
          color: "#000",
        }}
      >
        <span>QuickPitik / SRS Wireframes</span>
        <span style={{ color: "#666" }}>Module 1 — Mobile Application</span>
      </div>
      {children}
    </div>
  );
}
