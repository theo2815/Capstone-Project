"use client";

import { useEffect, useState } from "react";
import {
  WireframeNav,
  WireframeStamp,
} from "../_components/WireframeNav";

type Panel = "none" | "runner" | "photographer";

interface Chapter {
  n: string;
  title: string;
  cap: string;
  meta: Record<string, string>;
  body: string;
  time: string;
  cta?: string;
  fork?: boolean;
  margin?: string;
}

interface Persona {
  name: string;
  tagline: string;
  stat: string;
  total: string;
  page: string;
  date: string;
  chapters: Chapter[];
}

const INK = "#1F1A12";
const PAPER = "#EFE5D0";
const PAPER_DEEP = "#E5D9BD";
const SEPIA = "#9C6B3F";
const ANNOTATION = "#B23A2C";
const SAGE = "#6F7C5A";

const RUNNER: Persona = {
  name: "The Runner",
  tagline: "Find yourself in the crowd.",
  stat: "≈ 2.8 sec average search",
  total: "≈ 2 minutes, end to end",
  page: "p. 47",
  date: "entry · iv · viii · mmxxvi",
  chapters: [
    {
      n: "I",
      title: "Find your race",
      cap: "Enter the gallery of your event.",
      meta: { events: "200+ hosted", coverage: "Cebu · Manila · Davao" },
      body:
        "Filter by city, distance, or date. Most runners land here from a QR code shared at packet pickup or on the race's official page — no account needed to browse.",
      time: "5 seconds",
      cta: "Browse events",
      margin: "most arrive here from a QR at packet pickup ↘",
    },
    {
      n: "II",
      title: "Selfie or bib",
      cap: "Two ways to find yourself in the crowd.",
      meta: { time: "30 seconds", "selfie storage": "never" },
      body:
        "Take a quick selfie, or just type your bib number — both work. The selfie is used once for matching only. It's never stored, never shared, never seen by anyone but the matching engine itself.",
      time: "30 seconds",
      cta: "Try a sample search",
      margin: "selfie is single-use only — never stored.",
    },
    {
      n: "III",
      title: "The match",
      cap: "Live thumbnails as the AI finds you.",
      meta: { "avg search": "2.8 sec", accuracy: "98.6%" },
      body:
        "Photos appear in a flowing grid as they're found. Watch the count climb from zero to dozens in seconds. Every face match shows you the bib number it confirmed against, so you know the result is yours.",
      time: "3 seconds",
      margin: "the count keeps climbing — quietly addictive.",
    },
    {
      n: "IV",
      title: "Take them home",
      cap: "GCash, Maya, or card.",
      meta: { format: "high-res JPEG", license: "personal use" },
      body:
        "Pick the keepers, pay once, download immediately. No watermarks, no shipping, no waiting weeks for the photographer to email you. The full-resolution file is yours the moment payment clears.",
      time: "60 seconds",
      cta: "See sample pricing",
      margin: "no watermark. ever.",
    },
  ],
};

const PHOTOGRAPHER: Persona = {
  name: "The Photographer",
  tagline: "From shutter to gallery in minutes.",
  stat: "≈ 10 sec sort, no blur",
  total: "≈ 2 min setup · then automatic",
  page: "p. 48",
  date: "entry · iv · viii · mmxxvi",
  chapters: [
    {
      n: "I",
      title: "Choose your scale",
      cap: "Three tools — one gallery — picked by event size.",
      meta: { choice: "by volume", install: "optional" },
      body:
        "How big is the event you're shooting? Your answer tells us which tool to put in your hands. All three feed the same gallery, so you can mix and match within an event.",
      time: "10 seconds",
      fork: true,
      margin: "all three feed the same gallery — choose by volume.",
    },
    {
      n: "II",
      title: "Connect or upload",
      cap: "The tool guides you through.",
      meta: { tether: "wi-fi / usb", "auto-batch": "500 / chunk" },
      body:
        "Mobile uploads in real time as you shoot. Web works in any browser, drag-and-drop. Desktop chops 10,000-photo events into stable 500-photo batches automatically — no more upload failures three hours into a sync.",
      time: "1 minute",
      margin: "desktop solves the 10k-photo crash problem ↘",
    },
    {
      n: "III",
      title: "The cull",
      cap: "AI removes the blurs as photos arrive.",
      meta: { precision: "99.4%", latency: "< 1 sec / photo" },
      body:
        "You shoot a thousand frames. By the time you're back to the bag, the misses are gone. You only see your keepers — already sorted, already tagged with the bib numbers visible in each frame.",
      time: "0 seconds (background)",
      margin: "the cull happens while you walk back.",
    },
    {
      n: "IV",
      title: "Your gallery is live",
      cap: "Searchable to runners in minutes.",
      meta: { "live time": "minutes", search: "face + bib" },
      body:
        "The minute photos finish processing, runners can find themselves with a selfie or bib number. You earn while you're still shooting the back half of the race.",
      time: "0 seconds (automatic)",
      cta: "Apply to host an event",
      margin: "earnings start while you're still shooting.",
    },
  ],
};

const VOLUMES = [
  {
    label: "< 500",
    tool: "Web upload",
    desc: "Drag-drop in your browser. No install, no setup.",
    note: "smallest events",
  },
  {
    label: "500 – 10,000",
    tool: "Mobile app",
    desc: "Tether your camera. Auto-uploads as you shoot.",
    note: "most marathons",
  },
  {
    label: "10,000+",
    tool: "Desktop app",
    desc: "Auto-batches into stable 500-photo chunks.",
    note: "majors / nationals",
  },
];

export default function DualSidebarWireframe() {
  const [active, setActive] = useState<Panel>("none");

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setActive("none");
      if (e.key === "ArrowLeft") setActive("runner");
      if (e.key === "ArrowRight") setActive("photographer");
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <main
      className="relative min-h-screen"
      style={{
        background: PAPER,
        color: INK,
        fontFamily: "var(--font-notebook-body), Georgia, serif",
      }}
    >
      <PaperBackground />
      <WireframeNav active="dual-sidebar" />
      <WireframeStamp label="WF · Dual Sidebar · Field Notebook · Variant B" />

      <NotebookHeader active={active} onReset={() => setActive("none")} />

      <Spread active={active} setActive={setActive} />

      <KeyboardHints />
      <FieldNotes />
    </main>
  );
}

/* ─────────── Background paper texture ─────────── */

function PaperBackground() {
  return (
    <>
      {/* Faint horizontal ruled lines */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 z-0"
        style={{
          backgroundImage: `repeating-linear-gradient(
            to bottom,
            transparent 0,
            transparent 31px,
            rgba(31, 26, 18, 0.045) 31px,
            rgba(31, 26, 18, 0.045) 32px
          )`,
        }}
      />
      {/* Subtle paper grain */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 z-0"
        style={{
          opacity: 0.18,
          mixBlendMode: "multiply",
          backgroundImage:
            "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 240 240'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.59 0 0 0 0 0.42 0 0 0 0 0.21 0 0 0 0.6 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>\")",
        }}
      />
    </>
  );
}

/* ─────────── Header ─────────── */

function NotebookHeader({
  active,
  onReset,
}: {
  active: Panel;
  onReset: () => void;
}) {
  return (
    <header
      className="relative z-10 border-b"
      style={{ borderColor: `${INK}29` }}
    >
      <div className="mx-auto flex max-w-7xl flex-wrap items-baseline justify-between gap-4 px-8 py-7">
        <div className="flex items-baseline gap-6">
          <span
            className="text-[34px] font-light italic leading-none tracking-tight"
            style={{ fontFamily: "var(--font-notebook-display)" }}
          >
            Quick Pitik
          </span>
          <span
            className="text-[12.5px] italic"
            style={{
              fontFamily: "var(--font-notebook-body)",
              color: `${INK}99`,
            }}
          >
            — Field Notebook · Vol. IV · Spread III · MMXXVI
          </span>
        </div>

        <div className="relative flex items-center gap-7">
          {active !== "none" && (
            <button
              onClick={onReset}
              className="text-[16px] italic transition-colors hover:no-underline"
              style={{
                fontFamily: "var(--font-notebook-display)",
                color: ANNOTATION,
                textDecoration: "underline",
                textUnderlineOffset: "4px",
                textDecorationStyle: "wavy",
                textDecorationThickness: "1px",
              }}
            >
              ← back to the open spread
            </button>
          )}
          <span
            className="inline-block text-[20px]"
            style={{
              fontFamily: "var(--font-notebook-hand)",
              color: `${INK}A8`,
              transform: "rotate(-3deg)",
            }}
          >
            Cebu, PH
          </span>
          <CoffeeRing className="absolute -right-6 -top-3" size={70} />
        </div>
      </div>
    </header>
  );
}

/* ─────────── The spread ─────────── */

function Spread({
  active,
  setActive,
}: {
  active: Panel;
  setActive: (p: Panel) => void;
}) {
  return (
    <section className="relative z-10 mx-auto max-w-7xl px-4 sm:px-8">
      {active === "none" && <SpreadIntro />}

      <div
        className="relative flex min-h-[78vh] items-stretch"
        style={{
          border: `1px solid ${INK}33`,
          background: `${PAPER_DEEP}40`,
          boxShadow: `0 1px 0 0 ${INK}10, inset 0 0 0 1px ${INK}05`,
        }}
      >
        <RunnerPage
          active={active}
          onClick={() =>
            setActive(active === "runner" ? "none" : "runner")
          }
        />
        <PhotographerPage
          active={active}
          onClick={() =>
            setActive(active === "photographer" ? "none" : "photographer")
          }
        />
      </div>
    </section>
  );
}

function SpreadIntro() {
  return (
    <div className="relative py-12 text-center">
      <p
        className="text-[12px] tracking-[0.04em]"
        style={{
          fontFamily: "var(--font-notebook-body)",
          color: `${INK}80`,
          fontStyle: "italic",
        }}
      >
        — pick a page · the chosen one opens, the other folds aside —
      </p>
      <h2
        className="mt-5 text-[40px] font-light italic leading-[1.05] md:text-[58px]"
        style={{
          fontFamily: "var(--font-notebook-display)",
          color: `${INK}E6`,
          letterSpacing: "-0.01em",
        }}
      >
        Are you here to{" "}
        <em style={{ color: SEPIA, position: "relative" }}>
          find
          <HandUnderline color={SEPIA} />
        </em>{" "}
        photos,
        <br className="hidden sm:block" />
        or to{" "}
        <em style={{ color: SEPIA, position: "relative" }}>
          sell
          <HandUnderline color={SEPIA} />
        </em>{" "}
        them?
      </h2>
      <div
        className="mt-6 inline-flex items-center gap-3 text-[20px]"
        style={{
          fontFamily: "var(--font-notebook-hand)",
          color: ANNOTATION,
          transform: "rotate(-1.6deg)",
        }}
      >
        <ScribbleArrow />
        either side opens up — promise.
      </div>
    </div>
  );
}

/* ─────────── Page wrappers ─────────── */

function RunnerPage({
  active,
  onClick,
}: {
  active: Panel;
  onClick: () => void;
}) {
  const width =
    active === "none" ? "50%" : active === "runner" ? "80%" : "20%";

  return (
    <div
      className="relative flex shrink-0 flex-col overflow-hidden transition-[width] duration-700"
      style={{
        width,
        transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)",
        boxShadow: `inset -14px 0 22px -14px ${INK}28`,
      }}
    >
      <StitchSeam side="right" />
      {active === "runner" ? (
        <OpenSpread persona={RUNNER} side="left" />
      ) : active === "photographer" ? (
        <ClosedEdgeRail persona={RUNNER} onClick={onClick} />
      ) : (
        <PreviewCard persona={RUNNER} side="left" onClick={onClick} />
      )}
    </div>
  );
}

function PhotographerPage({
  active,
  onClick,
}: {
  active: Panel;
  onClick: () => void;
}) {
  const width =
    active === "none" ? "50%" : active === "photographer" ? "80%" : "20%";

  return (
    <div
      className="relative flex shrink-0 flex-col overflow-hidden transition-[width] duration-700"
      style={{
        width,
        transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)",
        boxShadow: `inset 14px 0 22px -14px ${INK}28`,
      }}
    >
      <StitchSeam side="left" />
      {active === "photographer" ? (
        <OpenSpread persona={PHOTOGRAPHER} side="right" />
      ) : active === "runner" ? (
        <ClosedEdgeRail persona={PHOTOGRAPHER} onClick={onClick} />
      ) : (
        <PreviewCard
          persona={PHOTOGRAPHER}
          side="right"
          onClick={onClick}
        />
      )}
    </div>
  );
}

/* ─────────── Preview card (idle / 50%) ─────────── */

function PreviewCard({
  persona,
  side,
  onClick,
}: {
  persona: Persona;
  side: "left" | "right";
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="group relative flex h-full w-full flex-col p-8 text-left transition-colors lg:p-14"
      style={{
        background: "transparent",
      }}
    >
      <DogEar />

      {/* Top header: page no. + spread label */}
      <header className="flex items-baseline justify-between">
        <span
          className="text-[22px] italic"
          style={{
            fontFamily: "var(--font-notebook-hand)",
            color: `${INK}9C`,
            transform: "rotate(-2deg)",
            display: "inline-block",
          }}
        >
          {persona.page}
        </span>
        <span
          className="text-[11px] italic tracking-[0.16em]"
          style={{
            fontFamily: "var(--font-notebook-body)",
            color: `${INK}80`,
            textTransform: "uppercase",
          }}
        >
          Spread III · Page {side === "left" ? "A" : "B"}
        </span>
      </header>

      <div
        className="my-6 h-px w-full"
        style={{
          background: `repeating-linear-gradient(to right, ${INK}55 0, ${INK}55 6px, transparent 6px, transparent 12px)`,
        }}
      />

      {/* Botanical sketch */}
      <div className="flex items-start justify-between">
        <PressedFern />
        <span
          className="text-[14px] italic"
          style={{
            fontFamily: "var(--font-notebook-body)",
            color: `${INK}75`,
          }}
        >
          fig. {side === "left" ? "i" : "ii"} —{" "}
          {persona.tagline.toLowerCase().replace(/\.$/, "")}
        </span>
      </div>

      {/* Persona name */}
      <h2
        className="mt-8 text-[58px] font-light italic leading-[0.95] tracking-tight lg:text-[72px]"
        style={{ fontFamily: "var(--font-notebook-display)" }}
      >
        {persona.name}
      </h2>
      <p
        className="mt-2 text-[20px] italic lg:text-[24px]"
        style={{
          fontFamily: "var(--font-notebook-display)",
          color: `${INK}B5`,
        }}
      >
        {persona.tagline}
      </p>

      {/* Stat in handwritten ink */}
      <div className="mt-5 flex items-center gap-3">
        <span
          className="text-[18px]"
          style={{
            fontFamily: "var(--font-notebook-hand)",
            color: ANNOTATION,
            transform: "rotate(-1.5deg)",
            display: "inline-block",
          }}
        >
          {persona.stat}
        </span>
        <ScribbleArrow color={ANNOTATION} small />
      </div>

      {/* Chapter list */}
      <div className="mt-10">
        <span
          className="text-[20px] italic"
          style={{
            fontFamily: "var(--font-notebook-hand)",
            color: `${INK}88`,
          }}
        >
          ── what you&apos;ll do here ──
        </span>
        <ol className="mt-4 space-y-3">
          {persona.chapters.map((c) => (
            <li
              key={c.n}
              className="flex items-baseline gap-4 border-b pb-2"
              style={{
                borderColor: `${INK}26`,
                borderStyle: "dashed",
              }}
            >
              <span
                className="text-[15px] italic"
                style={{
                  fontFamily: "var(--font-notebook-display)",
                  color: SEPIA,
                  minWidth: "1.4em",
                }}
              >
                {c.n}.
              </span>
              <span
                className="text-[17px]"
                style={{
                  fontFamily: "var(--font-notebook-body)",
                  color: `${INK}E0`,
                }}
              >
                {c.title}
              </span>
              <span
                className="ml-auto text-[15px] italic"
                style={{
                  fontFamily: "var(--font-notebook-hand)",
                  color: `${INK}80`,
                }}
              >
                {c.time}
              </span>
            </li>
          ))}
        </ol>
      </div>

      {/* Footer */}
      <div className="mt-auto flex items-baseline justify-between pt-10">
        <span
          className="text-[14px] italic"
          style={{
            fontFamily: "var(--font-notebook-body)",
            color: `${INK}80`,
          }}
        >
          {persona.total}
        </span>
        <span
          className="flex items-center gap-2 text-[22px] italic transition-all group-hover:gap-4"
          style={{
            fontFamily: "var(--font-notebook-display)",
            color: ANNOTATION,
            position: "relative",
          }}
        >
          Read the page <span style={{ fontSize: "26px" }}>→</span>
          <HandUnderline color={ANNOTATION} />
        </span>
      </div>

      <CoffeeRing
        className="pointer-events-none absolute"
        style={{ bottom: "8%", left: side === "left" ? "12%" : "auto", right: side === "right" ? "12%" : "auto" }}
        size={120}
      />
    </button>
  );
}

/* ─────────── Closed page rail (compressed / 20%) ─────────── */

function ClosedEdgeRail({
  persona,
  onClick,
}: {
  persona: Persona;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="group relative flex h-full w-full flex-col items-center justify-between px-3 py-8 transition-colors hover:bg-[rgba(229,217,189,0.45)]"
      style={{
        background: `${PAPER_DEEP}55`,
      }}
    >
      <span
        className="text-[18px] italic"
        style={{
          fontFamily: "var(--font-notebook-hand)",
          color: `${INK}A0`,
          transform: "rotate(-3deg)",
          display: "inline-block",
        }}
      >
        {persona.page}
      </span>

      <div
        className="text-[34px] font-light italic leading-none lg:text-[42px]"
        style={{
          fontFamily: "var(--font-notebook-display)",
          color: INK,
          writingMode: "vertical-rl",
          transform: "rotate(180deg)",
        }}
      >
        {persona.name}
      </div>

      <div
        className="flex flex-col items-center gap-4"
        style={{ fontFamily: "var(--font-notebook-body)" }}
      >
        <span
          className="text-[11px] italic tracking-[0.18em]"
          style={{
            color: `${INK}80`,
            writingMode: "vertical-rl",
            transform: "rotate(180deg)",
            textTransform: "uppercase",
          }}
        >
          {persona.stat}
        </span>
        <span
          className="border px-3 py-1 text-[15px] italic transition-colors group-hover:bg-[var(--rail-ann)] group-hover:text-[var(--rail-paper)]"
          style={
            {
              fontFamily: "var(--font-notebook-display)",
              color: ANNOTATION,
              borderColor: `${ANNOTATION}80`,
              borderStyle: "dashed",
              "--rail-ann": ANNOTATION,
              "--rail-paper": PAPER,
            } as React.CSSProperties
          }
        >
          ↻ open page
        </span>
      </div>
    </button>
  );
}

/* ─────────── Open spread (expanded / 80%) ─────────── */

function OpenSpread({
  persona,
  side,
}: {
  persona: Persona;
  side: "left" | "right";
}) {
  return (
    <div className="relative h-full overflow-y-auto px-8 py-10 lg:px-16">
      <DogEar />

      <header
        className="flex items-baseline justify-between border-b pb-6"
        style={{ borderColor: `${INK}33`, borderStyle: "dashed" }}
      >
        <div className="relative">
          <div className="flex items-baseline gap-4">
            <span
              className="text-[24px] italic"
              style={{
                fontFamily: "var(--font-notebook-hand)",
                color: ANNOTATION,
                transform: "rotate(-2deg)",
                display: "inline-block",
              }}
            >
              {persona.page}
            </span>
            <span
              className="text-[11px] italic tracking-[0.18em]"
              style={{
                fontFamily: "var(--font-notebook-body)",
                color: `${INK}80`,
                textTransform: "uppercase",
              }}
            >
              Spread III · Page {side === "left" ? "A" : "B"}
            </span>
          </div>
          <h1
            className="mt-3 text-[58px] font-light italic leading-[0.95] tracking-tight lg:text-[80px]"
            style={{ fontFamily: "var(--font-notebook-display)" }}
          >
            {persona.name}
          </h1>
          <p
            className="mt-2 text-[20px] italic lg:text-[24px]"
            style={{
              fontFamily: "var(--font-notebook-display)",
              color: `${INK}B5`,
            }}
          >
            {persona.tagline}
          </p>
        </div>
        <div className="flex flex-col items-end gap-2 text-right">
          <span
            className="text-[16px] italic"
            style={{
              fontFamily: "var(--font-notebook-hand)",
              color: `${INK}AA`,
              transform: "rotate(2deg)",
              display: "inline-block",
            }}
          >
            {persona.date}
          </span>
          <DateStamp />
          <span
            className="mt-2 text-[12px] italic tracking-[0.12em]"
            style={{
              fontFamily: "var(--font-notebook-body)",
              color: `${INK}80`,
              textTransform: "uppercase",
            }}
          >
            four entries · {persona.total}
          </span>
        </div>
      </header>

      <div className="stagger-children mt-12 space-y-16">
        {persona.chapters.map((c, i) => (
          <ChapterEntry key={c.n} chapter={c} index={i} />
        ))}
      </div>

      <div
        className="mt-16 flex items-center justify-center gap-4 border-t pt-8"
        style={{ borderColor: `${INK}26`, borderStyle: "dashed" }}
      >
        <PressedFern small />
        <span
          className="text-[12px] italic tracking-[0.18em]"
          style={{
            fontFamily: "var(--font-notebook-body)",
            color: `${INK}80`,
            textTransform: "uppercase",
          }}
        >
          end of page {side === "left" ? "A" : "B"}
        </span>
        <PressedFern small flip />
      </div>
    </div>
  );
}

/* ─────────── Chapter entry ─────────── */

function ChapterEntry({
  chapter,
  index,
}: {
  chapter: Chapter;
  index: number;
}) {
  const isEven = index % 2 === 0;

  return (
    <article className="relative grid gap-10 md:grid-cols-[180px_1fr]">
      <header
        className="md:border-r md:pr-6"
        style={{ borderColor: `${INK}24`, borderStyle: "dashed" }}
      >
        <span
          className="block text-[100px] font-light italic leading-[0.85]"
          style={{
            fontFamily: "var(--font-notebook-display)",
            color: SEPIA,
          }}
        >
          {chapter.n}
        </span>
        <span
          className="mt-4 block text-[10px] italic tracking-[0.18em]"
          style={{
            fontFamily: "var(--font-notebook-body)",
            color: `${INK}88`,
            textTransform: "uppercase",
          }}
        >
          Chapter
        </span>
        <span
          className="mt-1 block text-[18px] italic"
          style={{
            fontFamily: "var(--font-notebook-hand)",
            color: ANNOTATION,
            transform: "rotate(-1.5deg)",
          }}
        >
          ≈ {chapter.time}
        </span>
      </header>

      <div className="relative">
        <h3
          className="relative inline-block text-[34px] font-light italic leading-[1.05] tracking-tight lg:text-[44px]"
          style={{ fontFamily: "var(--font-notebook-display)" }}
        >
          {chapter.title}
          <HandUnderline color={ANNOTATION} />
        </h3>
        <p
          className="mt-2 text-[19px] italic lg:text-[22px]"
          style={{
            fontFamily: "var(--font-notebook-display)",
            color: `${INK}B0`,
          }}
        >
          {chapter.cap}
        </p>

        {/* Meta as a small ledger row */}
        <dl
          className="mt-6 grid grid-cols-1 gap-y-1 border-y py-3 sm:grid-cols-2 sm:gap-x-6"
          style={{ borderColor: `${INK}30`, borderStyle: "dashed" }}
        >
          {Object.entries(chapter.meta).map(([k, v]) => (
            <div
              key={k}
              className="flex items-baseline justify-between gap-3 text-[13px]"
            >
              <dt
                className="italic"
                style={{
                  fontFamily: "var(--font-notebook-body)",
                  color: `${INK}88`,
                  textTransform: "lowercase",
                  letterSpacing: "0.02em",
                }}
              >
                {k}
              </dt>
              <dd
                style={{
                  fontFamily: "var(--font-notebook-body)",
                  color: INK,
                  fontVariantNumeric: "tabular-nums",
                  fontWeight: 500,
                }}
              >
                {v}
              </dd>
            </div>
          ))}
        </dl>

        {/* Body with illuminated drop cap + floated marginalia */}
        <div
          className="mt-7 text-[17px] leading-[1.7]"
          style={{
            fontFamily: "var(--font-notebook-body)",
            color: `${INK}E0`,
          }}
        >
          {chapter.margin && (
            <MarginNote
              text={chapter.margin}
              placement={isEven ? "left" : "right"}
            />
          )}
          <span
            style={{
              float: "left",
              fontFamily: "var(--font-notebook-display)",
              fontStyle: "italic",
              fontWeight: 300,
              fontSize: "76px",
              lineHeight: "0.78",
              color: SEPIA,
              marginRight: "10px",
              marginTop: "6px",
            }}
          >
            {chapter.body.charAt(0)}
          </span>
          {chapter.body.slice(1)}
          <span style={{ display: "block", clear: "both" }} />
        </div>

        {chapter.fork && <VolumeFork />}

        {chapter.cta && (
          <button
            className="relative mt-7 inline-block text-[20px] italic transition-colors"
            style={{
              fontFamily: "var(--font-notebook-display)",
              color: ANNOTATION,
            }}
          >
            {chapter.cta} →
            <HandUnderline color={ANNOTATION} />
          </button>
        )}
      </div>
    </article>
  );
}

/* ─────────── Volume fork (notebook sketch) ─────────── */

function VolumeFork() {
  return (
    <div
      className="relative mt-8 border p-6"
      style={{
        borderColor: `${INK}40`,
        background: `${PAPER}80`,
      }}
    >
      <span
        className="text-[18px] italic"
        style={{
          fontFamily: "var(--font-notebook-hand)",
          color: `${INK}90`,
          transform: "rotate(-1deg)",
          display: "inline-block",
        }}
      >
        ✱ pinned note · which event size?
      </span>
      <p
        className="mt-3 text-[26px] italic leading-tight"
        style={{
          fontFamily: "var(--font-notebook-display)",
          color: `${INK}E0`,
        }}
      >
        How big is the event you&apos;re shooting?
      </p>
      <div className="mt-6 grid gap-4 md:grid-cols-3">
        {VOLUMES.map((v, idx) => (
          <button
            key={v.label}
            className="group/vol relative border p-5 text-left transition-all hover:translate-y-[-2px]"
            style={{
              borderColor: `${INK}48`,
              borderStyle: "dashed",
              background: `${PAPER_DEEP}55`,
              transform: `rotate(${(idx - 1) * 0.6}deg)`,
            }}
          >
            <span
              className="text-[18px] italic"
              style={{
                fontFamily: "var(--font-notebook-hand)",
                color: ANNOTATION,
                transform: "rotate(-2deg)",
                display: "inline-block",
              }}
            >
              {v.note}
            </span>
            <p
              className="mt-1 text-[36px] font-light italic leading-tight"
              style={{ fontFamily: "var(--font-notebook-display)" }}
            >
              {v.label}
            </p>
            <p
              className="mt-3 text-[16px] italic"
              style={{
                fontFamily: "var(--font-notebook-display)",
                color: SEPIA,
              }}
            >
              → {v.tool}
            </p>
            <p
              className="mt-2 text-[14px]"
              style={{
                fontFamily: "var(--font-notebook-body)",
                color: `${INK}A8`,
              }}
            >
              {v.desc}
            </p>
          </button>
        ))}
      </div>
    </div>
  );
}

/* ─────────── Decorative atoms ─────────── */

function HandUnderline({ color = ANNOTATION }: { color?: string }) {
  return (
    <svg
      width="100%"
      height="6"
      viewBox="0 0 200 6"
      preserveAspectRatio="none"
      className="absolute -bottom-1 left-0 right-0"
      aria-hidden
      style={{ overflow: "visible" }}
    >
      <path
        d="M0 3 Q 30 0 60 3 T 120 3 T 200 3"
        fill="none"
        stroke={color}
        strokeWidth="1.4"
        strokeLinecap="round"
        opacity={0.85}
      />
    </svg>
  );
}

function ScribbleArrow({
  color = ANNOTATION,
  small = false,
}: {
  color?: string;
  small?: boolean;
}) {
  const w = small ? 28 : 36;
  const h = small ? 16 : 20;
  return (
    <svg
      width={w}
      height={h}
      viewBox="0 0 36 20"
      aria-hidden
      style={{ overflow: "visible" }}
    >
      <path
        d="M2 10 Q 12 2 22 10 T 32 8"
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <path
        d="M28 4 L 32 8 L 26 11"
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function CoffeeRing({
  size = 80,
  className = "",
  style = {},
}: {
  size?: number;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 80 80"
      className={className}
      style={{ opacity: 0.16, ...style }}
      aria-hidden
    >
      <circle
        cx="40"
        cy="40"
        r="32"
        fill="none"
        stroke={SEPIA}
        strokeWidth="3.2"
        strokeDasharray="2 6 9 4 12 3 5 8"
      />
      <circle
        cx="40"
        cy="40"
        r="26"
        fill="none"
        stroke={SEPIA}
        strokeWidth="1.5"
        strokeDasharray="3 7 2 9 4 6"
      />
      <circle
        cx="40"
        cy="40"
        r="28"
        fill={SEPIA}
        opacity="0.06"
      />
    </svg>
  );
}

function DogEar() {
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute right-0 top-0 transition-transform duration-500 ease-out group-hover:-translate-x-[2px] group-hover:translate-y-[2px]"
      style={{
        width: 32,
        height: 32,
        background: PAPER_DEEP,
        borderLeft: `1px solid ${INK}40`,
        borderBottom: `1px solid ${INK}40`,
        clipPath: "polygon(0 0, 100% 100%, 100% 0)",
        boxShadow: `-2px 2px 4px ${INK}22`,
      }}
    />
  );
}

function PressedFern({
  small = false,
  flip = false,
}: {
  small?: boolean;
  flip?: boolean;
}) {
  const w = small ? 36 : 56;
  const h = small ? 36 : 56;
  return (
    <svg
      width={w}
      height={h}
      viewBox="0 0 60 60"
      aria-hidden
      style={{
        opacity: 0.55,
        transform: flip ? "scaleX(-1)" : undefined,
      }}
    >
      <g stroke={SAGE} strokeWidth="1" fill="none" strokeLinecap="round">
        <path d="M30 56 Q 30 30 30 8" />
        <path d="M30 50 Q 22 47 16 42" />
        <path d="M30 44 Q 38 41 44 36" />
        <path d="M30 38 Q 22 35 17 30" />
        <path d="M30 32 Q 38 29 43 24" />
        <path d="M30 26 Q 23 23 19 18" />
        <path d="M30 20 Q 36 17 40 12" />
        <path d="M30 14 Q 26 12 24 8" />
      </g>
      <circle cx="30" cy="8" r="1.5" fill={SAGE} opacity={0.7} />
    </svg>
  );
}

function StitchSeam({ side }: { side: "left" | "right" }) {
  return (
    <div
      aria-hidden
      className={`pointer-events-none absolute top-0 bottom-0 ${
        side === "right" ? "right-0" : "left-0"
      }`}
      style={{
        width: "2px",
        backgroundImage: `repeating-linear-gradient(
          to bottom,
          ${SEPIA} 0,
          ${SEPIA} 7px,
          transparent 7px,
          transparent 14px
        )`,
        opacity: 0.55,
      }}
    />
  );
}

function MarginNote({
  text,
  placement,
}: {
  text: string;
  placement: "left" | "right";
}) {
  const tilt = placement === "left" ? -1.6 : 1.6;
  return (
    <aside
      className={`relative mb-3 inline-flex items-start gap-2 ${
        placement === "right" ? "float-right ml-4" : "float-left mr-4"
      }`}
      style={{
        maxWidth: 200,
        fontFamily: "var(--font-notebook-hand)",
        color: ANNOTATION,
        transform: `rotate(${tilt}deg)`,
      }}
    >
      <svg
        width="22"
        height="22"
        viewBox="0 0 22 22"
        aria-hidden
        style={{
          flexShrink: 0,
          transform: placement === "right" ? "scaleX(-1)" : undefined,
          marginTop: 4,
        }}
      >
        <path
          d="M2 11 Q 9 4 18 11"
          fill="none"
          stroke={ANNOTATION}
          strokeWidth="1.3"
          strokeLinecap="round"
        />
        <path
          d="M14 7 L 18 11 L 14 14"
          fill="none"
          stroke={ANNOTATION}
          strokeWidth="1.3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span className="text-[18px] leading-[1.3]">{text}</span>
    </aside>
  );
}

function DateStamp() {
  return (
    <div
      className="inline-flex items-center justify-center border-2 px-3 py-1"
      style={{
        borderColor: SEPIA,
        color: SEPIA,
        fontFamily: "var(--font-notebook-body)",
        fontSize: "11px",
        letterSpacing: "0.18em",
        textTransform: "uppercase",
        fontStyle: "italic",
        opacity: 0.7,
        transform: "rotate(2deg)",
        background: `${PAPER}80`,
      }}
    >
      Cebu · MMXXVI
    </div>
  );
}

/* ─────────── Keyboard hints ─────────── */

function KeyboardHints() {
  return (
    <div
      className="relative z-10 mx-auto max-w-7xl px-8 py-7 text-center text-[14px] italic"
      style={{
        fontFamily: "var(--font-notebook-hand)",
        color: `${INK}90`,
      }}
    >
      <span
        className="border px-2 py-[2px]"
        style={{ borderColor: `${INK}55`, borderStyle: "dashed" }}
      >
        ←
      </span>{" "}
      <span
        className="border px-2 py-[2px]"
        style={{ borderColor: `${INK}55`, borderStyle: "dashed" }}
      >
        →
      </span>{" "}
      flip pages &nbsp;·&nbsp;{" "}
      <span
        className="border px-2 py-[2px]"
        style={{ borderColor: `${INK}55`, borderStyle: "dashed" }}
      >
        esc
      </span>{" "}
      close the open spread
    </div>
  );
}

/* ─────────── Field notes (footer specs) ─────────── */

function FieldNotes() {
  return (
    <footer
      className="relative z-10 mt-8 border-t"
      style={{
        borderColor: `${INK}26`,
        background: `${PAPER_DEEP}40`,
      }}
    >
      <div className="mx-auto max-w-7xl px-8 py-16">
        <div className="flex items-baseline gap-4">
          <PressedFern small />
          <span
            className="text-[12px] italic tracking-[0.18em]"
            style={{
              fontFamily: "var(--font-notebook-body)",
              color: `${INK}88`,
              textTransform: "uppercase",
            }}
          >
            ─── field notes ───
          </span>
        </div>
        <h3
          className="mt-3 text-[40px] font-light italic"
          style={{ fontFamily: "var(--font-notebook-display)" }}
        >
          The four observations
        </h3>

        <div className="mt-12 grid gap-14 lg:grid-cols-2">
          <NotesBlock
            label="UX improvements over the original idea"
            items={[
              [
                "A held centre",
                "The plain question (\"are you here to find or sell?\") fills the void above the diptych — the user is never left guessing.",
              ],
              [
                "The other page never disappears",
                "When one panel opens, the other folds to a 20% closed-edge rail with vertical name + page number. One click brings it back.",
              ],
              [
                "Information scent at idle",
                "Each preview already shows persona name, tagline, headline stat, and a 4-chapter peek. No blind clicks.",
              ],
              [
                "Photographer asymmetry made plain",
                "Volume fork lives inside Chapter I — pinned-note metaphor with three index cards rather than three confusing app names.",
              ],
              [
                "Privacy in plain text",
                "Selfie privacy line embedded in Chapter II's body — not buried in a tooltip.",
              ],
              [
                "Marginalia on every chapter",
                "Hand-written annotations carry the things you'd otherwise stuff into a tooltip — they reward the reader without demanding interaction.",
              ],
            ]}
          />

          <NotesBlock
            label="Animations & transitions"
            items={[
              [
                "Slide expand · 700ms",
                "cubic-bezier(0.16, 1, 0.3, 1) on the width property. Slow enough to feel intentional, fast enough not to drag.",
              ],
              [
                "Idle hover · 280ms",
                "Selected page lifts subtly with a soft sepia inner-shadow; the other dims a hair. A held breath, not a shout.",
              ],
              [
                "Closed-edge hover · 280ms",
                "Width 20% → 25%, paper-deep tone deepens. The dashed ↻ open page badge softly pulses to advertise tappability.",
              ],
              [
                "Chapter reveal · staggered fade-up",
                "Each entry fades up 24px with 100ms stagger using the existing .stagger-children animation.",
              ],
              [
                "Drop cap · scale + fade",
                "Illuminated drop cap scales 0.94 → 1, opacity 0.6 → 1, over 600ms ease-out as it scrolls into view.",
              ],
              [
                "Hand-drawn underline · idle",
                "SVG wobble path under chapter titles and CTAs. No animation required — the irregularity itself reads as motion.",
              ],
              [
                "Dog-ear · hover lift",
                "Top-right corner shifts 2px down/right on card hover. A signal that this page can be picked up.",
              ],
              [
                "Keyboard",
                "← / → to flip pages · ESC to close the open spread. Hints visible in the footer of the wireframe.",
              ],
            ]}
          />
        </div>

        <div
          className="mt-14 border-t pt-8"
          style={{ borderColor: `${INK}26`, borderStyle: "dashed" }}
        >
          <span
            className="text-[12px] italic tracking-[0.18em]"
            style={{
              fontFamily: "var(--font-notebook-body)",
              color: `${INK}88`,
              textTransform: "uppercase",
            }}
          >
            ── mobile pattern (sketch) ──
          </span>
          <p
            className="mt-3 max-w-2xl text-[18px] italic"
            style={{
              fontFamily: "var(--font-notebook-display)",
              color: `${INK}B8`,
            }}
          >
            Stack vertically under 768px. Each persona becomes a full-width
            preview card. Tap → that card fills the screen, the other becomes a
            sticky top tab styled as a dog-eared page corner. Same content
            shape — no interaction is lost.
          </p>
          <div className="mt-7 grid gap-4 md:grid-cols-3">
            <MobileSketch
              label="Idle"
              copy="Two stacked notebook pages. Runner on top, photographer below. Same chapter peek and marginalia."
            />
            <MobileSketch
              label="Active"
              copy="Tapped page fills the screen. Other persona becomes a sticky dog-eared tab with name + page number."
            />
            <MobileSketch
              label="Switch"
              copy="Tap the top tab → horizontal page-flip (220ms). Both directions feel symmetric."
            />
          </div>
        </div>
      </div>
    </footer>
  );
}

function NotesBlock({
  label,
  items,
}: {
  label: string;
  items: [string, string][];
}) {
  return (
    <div>
      <span
        className="text-[18px] italic"
        style={{
          fontFamily: "var(--font-notebook-hand)",
          color: ANNOTATION,
          transform: "rotate(-0.6deg)",
          display: "inline-block",
        }}
      >
        ✦ {label}
      </span>
      <ul className="mt-5 space-y-5">
        {items.map(([k, v]) => (
          <li
            key={k}
            className="border-b pb-3"
            style={{ borderColor: `${INK}24`, borderStyle: "dashed" }}
          >
            <p
              className="text-[20px] italic"
              style={{ fontFamily: "var(--font-notebook-display)" }}
            >
              {k}
            </p>
            <p
              className="mt-1 text-[15px] leading-[1.6]"
              style={{
                fontFamily: "var(--font-notebook-body)",
                color: `${INK}B0`,
              }}
            >
              {v}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}

function MobileSketch({ label, copy }: { label: string; copy: string }) {
  return (
    <div
      className="border p-5"
      style={{
        borderColor: `${INK}30`,
        borderStyle: "dashed",
        background: `${PAPER}90`,
      }}
    >
      <span
        className="text-[16px] italic"
        style={{
          fontFamily: "var(--font-notebook-hand)",
          color: ANNOTATION,
          transform: "rotate(-1deg)",
          display: "inline-block",
        }}
      >
        ✱ {label}
      </span>
      <p
        className="mt-2 text-[14px] italic leading-relaxed"
        style={{
          fontFamily: "var(--font-notebook-display)",
          color: `${INK}AA`,
        }}
      >
        {copy}
      </p>
    </div>
  );
}
