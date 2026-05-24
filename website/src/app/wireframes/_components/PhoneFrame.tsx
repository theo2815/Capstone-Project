import type { ReactNode } from "react";

type PhoneFrameProps = {
  ucId: string;
  screenName: string;
  children: ReactNode;
};

export function PhoneFrame({ ucId, screenName, children }: PhoneFrameProps) {
  return (
    <div className="flex flex-col items-center gap-5">
      <div
        className="relative bg-white"
        style={{
          width: 360,
          height: 780,
          border: "1.5px solid #000",
          borderRadius: 28,
          boxSizing: "border-box",
          overflow: "hidden",
        }}
      >
        <StatusBar />
        <div
          style={{
            height: 693,
            background: "#fff",
            overflow: "hidden",
            boxSizing: "border-box",
          }}
        >
          {children}
        </div>
        <NavBar />
      </div>

      <div className="text-center font-mono">
        <p
          className="tabular-nums"
          style={{
            fontSize: 11,
            letterSpacing: "0.22em",
            textTransform: "uppercase",
            color: "#000",
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

function StatusBar() {
  return (
    <div
      style={{
        height: 36,
        borderBottom: "1.5px solid #000",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 16px",
        fontFamily: "var(--font-mono), ui-monospace, monospace",
        fontSize: 11,
        fontVariantNumeric: "tabular-nums",
        color: "#000",
        boxSizing: "border-box",
      }}
    >
      <span>9:41</span>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ letterSpacing: "0.05em" }}>5G</span>
        <span
          style={{
            display: "inline-flex",
            alignItems: "flex-end",
            gap: 1.5,
          }}
        >
          <span style={{ width: 2, height: 3, background: "#000" }} />
          <span style={{ width: 2, height: 5, background: "#000" }} />
          <span style={{ width: 2, height: 7, background: "#000" }} />
          <span style={{ width: 2, height: 9, background: "#000" }} />
        </span>
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            position: "relative",
            marginLeft: 2,
          }}
        >
          <span
            style={{
              width: 22,
              height: 11,
              border: "1px solid #000",
              borderRadius: 2,
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "flex-start",
              padding: 1,
              boxSizing: "border-box",
            }}
          >
            <span style={{ display: "block", width: 16, height: 7, background: "#000" }} />
          </span>
          <span
            style={{
              width: 2,
              height: 5,
              background: "#000",
              marginLeft: 1,
              borderRadius: "0 1px 1px 0",
            }}
          />
        </span>
      </div>
    </div>
  );
}

function NavBar() {
  return (
    <div
      style={{
        height: 48,
        borderTop: "1.5px solid #000",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-around",
        boxSizing: "border-box",
      }}
    >
      <span
        aria-hidden
        style={{
          width: 14,
          height: 14,
          borderLeft: "1.5px solid #000",
          borderBottom: "1.5px solid #000",
          transform: "rotate(45deg)",
        }}
      />
      <span
        aria-hidden
        style={{
          width: 14,
          height: 14,
          borderRadius: "9999px",
          border: "1.5px solid #000",
        }}
      />
      <span
        aria-hidden
        style={{
          width: 13,
          height: 13,
          border: "1.5px solid #000",
        }}
      />
    </div>
  );
}
