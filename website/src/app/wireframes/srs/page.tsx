import Link from "next/link";

const ROUTES = [
  { idx: "01", uc: "UC-M1-1.1", screen: "Tether Camera Screen", href: "/wireframes/srs/m1-1", group: "Module 1 · Mobile App" },
  { idx: "02", uc: "UC-M1-1.2", screen: "Capture & Queue Screen", href: "/wireframes/srs/m1-2", group: "Module 1 · Mobile App" },
  { idx: "03", uc: "UC-M1-1.3", screen: "Upload Queue Screen", href: "/wireframes/srs/m1-3", group: "Module 1 · Mobile App" },
  { idx: "04", uc: "UC-M1-1.4", screen: "Offline / Local-Cache State", href: "/wireframes/srs/m1-4", group: "Module 1 · Mobile App" },
  { idx: "05", uc: "UC-M1-1.5", screen: "Notification + Filtered Gallery", href: "/wireframes/srs/m1-5", group: "Module 1 · Mobile App" },
  { idx: "06", uc: "UC-M2-2.2", screen: "Blur Analysis & Manual Override", href: "/wireframes/srs/m2-2", group: "Module 2 · Desktop (BatchMyPhotos)" },
  { idx: "07", uc: "UC-M2-2.3", screen: "Batch Sort & Folder Output", href: "/wireframes/srs/m2-3", group: "Module 2 · Desktop (BatchMyPhotos)" },
  { idx: "08", uc: "UC-M3-3.3", screen: "Search by Selfie · AI Face Match", href: "/wireframes/srs/m3-3", group: "Module 3 · Marketplace (Web)" },
  { idx: "09", uc: "UC-M3-3.4", screen: "Search by Bib Number · AI OCR", href: "/wireframes/srs/m3-4", group: "Module 3 · Marketplace (Web)" },
  { idx: "10", uc: "UC-M3-3.6", screen: "Cart Drawer · Photo Review", href: "/wireframes/srs/m3-6", group: "Module 3 · Marketplace (Web)" },
  { idx: "11", uc: "UC-M3-3.7", screen: "Checkout & Payment · PayMongo", href: "/wireframes/srs/m3-7", group: "Module 3 · Marketplace (Web)" },
  { idx: "12", uc: "UC-3.3",    screen: "Receipt & Bundle Download · Post-Payment", href: "/wireframes/srs/m3-8", group: "Module 3 · Marketplace (Web)" },
];

export default function WireframesIndex() {
  return (
    <div
      style={{
        width: "100%",
        maxWidth: 720,
        background: "#fff",
        border: "1.5px solid #000",
        padding: 32,
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono), ui-monospace, monospace",
          fontSize: 10,
          letterSpacing: "0.22em",
          textTransform: "uppercase",
          color: "#666",
          marginBottom: 8,
        }}
      >
        SRS § 3.2 — Modules 1, 2 & 3
      </div>
      <h1
        style={{
          fontFamily: "var(--font-display), system-ui, sans-serif",
          fontSize: 28,
          fontWeight: 600,
          color: "#000",
          margin: "0 0 8px 0",
          letterSpacing: "-0.01em",
        }}
      >
        QuickPitik Wireframes
      </h1>
      <p
        style={{
          fontFamily: "var(--font-sans), system-ui, sans-serif",
          fontSize: 13,
          color: "#333",
          margin: "0 0 32px 0",
          maxWidth: 540,
          lineHeight: 1.55,
        }}
      >
        Mobile (Android phone), desktop (BatchMyPhotos window), and web
        (browser) mockups — pure black and white, sized for direct
        screenshot into the SRS document. One screen per route.
      </p>

      <div style={{ borderTop: "1px solid #000" }}>
        {ROUTES.map((r, i) => (
          <Link
            key={r.uc}
            href={r.href}
            style={{
              display: "flex",
              alignItems: "center",
              padding: "16px 0",
              borderBottom: "1px solid #000",
              borderTop: i > 0 && r.group !== ROUTES[i - 1].group ? "1.5px solid #000" : "none",
              color: "#000",
              textDecoration: "none",
              gap: 16,
            }}
          >
            <span
              style={{
                fontFamily: "var(--font-mono), ui-monospace, monospace",
                fontSize: 11,
                color: "#666",
                width: 24,
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {r.idx}
            </span>
            <span
              style={{
                fontFamily: "var(--font-mono), ui-monospace, monospace",
                fontSize: 11,
                letterSpacing: "0.22em",
                textTransform: "uppercase",
                color: "#000",
                width: 90,
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {r.uc}
            </span>
            <span
              style={{
                fontFamily: "var(--font-sans), system-ui, sans-serif",
                fontSize: 14,
                color: "#000",
                flex: 1,
              }}
            >
              {r.screen}
            </span>
            <span
              aria-hidden
              style={{
                width: 8,
                height: 8,
                borderRight: "1.5px solid #000",
                borderBottom: "1.5px solid #000",
                transform: "rotate(-45deg)",
              }}
            />
          </Link>
        ))}
      </div>
    </div>
  );
}
