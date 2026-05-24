import type { ReactNode } from "react";
import { Hatch } from "./WireUI";

const mono = "var(--font-mono), ui-monospace, monospace";
const display = "var(--font-display), system-ui, sans-serif";
const sans = "var(--font-sans), system-ui, sans-serif";

type EventBackdropProps = {
  cartCount?: number;
  activeTab?: "face" | "bib";
  panel?: ReactNode;
  dim?: boolean;
};

export function EventBackdrop({
  cartCount = 3,
  activeTab,
  panel,
  dim = false,
}: EventBackdropProps) {
  return (
    <div
      style={{
        height: "100%",
        background: "#fff",
        overflow: "hidden",
        position: "relative",
        boxSizing: "border-box",
        display: "flex",
        flexDirection: "column",
        opacity: dim ? 0.35 : 1,
        filter: dim ? "grayscale(1)" : undefined,
      }}
    >
      <SiteHeader cartCount={cartCount} />
      <div style={{ flex: 1, overflow: "hidden", padding: "20px 40px" }}>
        <EventHero />
        {(activeTab || panel) && (
          <div style={{ marginTop: 18 }}>
            {activeTab && <SearchTabs active={activeTab} />}
            {panel}
          </div>
        )}
        {!panel && <PhotoGrid />}
      </div>
    </div>
  );
}

function SiteHeader({ cartCount }: { cartCount: number }) {
  return (
    <div
      style={{
        height: 52,
        borderBottom: "1px solid #000",
        display: "flex",
        alignItems: "center",
        padding: "0 32px",
        boxSizing: "border-box",
        justifyContent: "space-between",
        background: "#fff",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 28 }}>
        <span
          style={{
            fontFamily: display,
            fontSize: 18,
            fontWeight: 700,
            letterSpacing: "-0.01em",
            color: "#000",
          }}
        >
          QuickPitik
        </span>
        <nav style={{ display: "flex", gap: 22 }}>
          {["Events", "Photographers", "How it works"].map((item, i) => (
            <span
              key={item}
              style={{
                fontFamily: mono,
                fontSize: 10,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                color: i === 0 ? "#000" : "#666",
                borderBottom: i === 0 ? "1px solid #000" : "none",
                paddingBottom: 2,
              }}
            >
              {item}
            </span>
          ))}
        </nav>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <span
          style={{
            fontFamily: mono,
            fontSize: 10,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "#666",
          }}
        >
          Theo
        </span>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            padding: "4px 12px 4px 10px",
            border: "1.5px solid #000",
            borderRadius: 999,
            background: "#000",
            color: "#fff",
          }}
        >
          <span
            aria-hidden
            style={{
              width: 14,
              height: 14,
              borderRadius: 999,
              border: "1.5px solid #fff",
              display: "inline-block",
            }}
          />
          <span
            style={{
              fontFamily: mono,
              fontSize: 10,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            Cart · {cartCount}
          </span>
        </div>
      </div>
    </div>
  );
}

function EventHero() {
  return (
    <div style={{ display: "flex", gap: 28, alignItems: "flex-end" }}>
      <div style={{ flex: 1 }}>
        <div
          style={{
            fontFamily: mono,
            fontSize: 10,
            letterSpacing: "0.3em",
            textTransform: "uppercase",
            color: "#666",
          }}
        >
          21 K · March 8, 2026 · Cebu City
        </div>
        <h1
          style={{
            fontFamily: display,
            fontSize: 48,
            fontWeight: 500,
            color: "#000",
            margin: "8px 0 0",
            letterSpacing: "-0.02em",
            lineHeight: 1.05,
          }}
        >
          Run Cebu 2026
        </h1>
        <div
          style={{
            marginTop: 12,
            display: "flex",
            gap: 24,
            alignItems: "baseline",
          }}
        >
          <Stat label="Photos" value="14,283" />
          <Stat label="Photographers" value="6" />
          <Stat label="Runners enrolled" value="2,418" />
        </div>
      </div>
      <Hatch width={220} height={96} label="Event Cover Photo" />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div
        style={{
          fontFamily: mono,
          fontSize: 9,
          letterSpacing: "0.22em",
          textTransform: "uppercase",
          color: "#666",
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: mono,
          fontSize: 18,
          fontVariantNumeric: "tabular-nums",
          fontWeight: 600,
          marginTop: 2,
        }}
      >
        {value}
      </div>
    </div>
  );
}

function SearchTabs({ active }: { active: "face" | "bib" }) {
  return (
    <div style={{ display: "flex", borderBottom: "1.5px solid #000" }}>
      {(
        [
          { id: "face", label: "Search by Selfie" },
          { id: "bib", label: "Search by Bib Number" },
        ] as const
      ).map((t) => (
        <div
          key={t.id}
          style={{
            padding: "10px 18px",
            marginBottom: -1.5,
            borderBottom: active === t.id ? "2.5px solid #000" : "2.5px solid transparent",
            fontFamily: mono,
            fontSize: 11,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: active === t.id ? "#000" : "#666",
            fontWeight: active === t.id ? 600 : 400,
          }}
        >
          {t.label}
        </div>
      ))}
    </div>
  );
}

function PhotoGrid() {
  return (
    <div style={{ marginTop: 18 }}>
      <div
        style={{
          fontFamily: mono,
          fontSize: 10,
          letterSpacing: "0.22em",
          textTransform: "uppercase",
          color: "#666",
          marginBottom: 10,
        }}
      >
        All Photos · 14,283
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10 }}>
        {Array.from({ length: 10 }).map((_, i) => (
          <WatermarkedThumb key={i} />
        ))}
      </div>
    </div>
  );
}

function WatermarkedThumb() {
  return (
    <div style={{ position: "relative", aspectRatio: "4 / 3" }}>
      <Hatch width="100%" height="100%" density={4} />
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: mono,
          fontSize: 12,
          letterSpacing: "0.22em",
          color: "#000",
          opacity: 0.3,
          fontWeight: 700,
          transform: "rotate(-16deg)",
        }}
      >
        QP · WM
      </div>
      <div
        style={{
          position: "absolute",
          bottom: 4,
          right: 6,
          fontFamily: mono,
          fontSize: 9,
          background: "rgba(255,255,255,0.85)",
          padding: "1px 4px",
          color: "#000",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        ₱50
      </div>
    </div>
  );
}

export { sans, mono, display };
