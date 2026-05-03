import Link from "next/link";
import type { Metadata } from "next";
import { ROUTES } from "@/lib/constants";
import { SiteHeader } from "@/components/layout/site-header";

export const metadata: Metadata = {
  title: "QuickPitik — Find your race photos",
  description:
    "Find your marathon photos in seconds. Search by face or bib number.",
};

export default function RunnersPage() {
  return (
    <main className="snap-y-mobile bg-bone text-ink relative">
      <SiteHeader />

      <Hero />

      <StepSection
        n={1}
        title="Pick your event."
        sub="Tap one. That's it."
        screen={<EventListScreen />}
        bg="bone"
      />

      <StepSection
        n={2}
        title="Show us your face."
        sub="Or just type your bib number."
        screen={<ScanScreen />}
        bg="bone-deep"
        flip
      />

      <StepSection
        n={3}
        title="Save them all."
        sub="Yours to keep, share, or print."
        screen={<PhotosScreen />}
        bg="bone"
      />

      <PhotoTeaser />

      <FinalCta />

      <Footer />

      <MobileStickyCta />
    </main>
  );
}

function Hero() {
  return (
    <section className="snap-section min-h-[calc(100vh-3.75rem)] flex items-center px-6 md:px-10 py-16 md:py-20 bg-bone">
      <div className="max-w-7xl mx-auto w-full grid md:grid-cols-2 gap-12 md:gap-16 items-center">
        <div className="stagger-children">
          <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-slate mb-4">
            For Runners
          </p>
          <h1 className="font-display text-5xl md:text-7xl lg:text-8xl font-medium tracking-tight leading-[0.95]">
            From finish line
            <br />
            <span className="text-fresh">to your phone.</span>
          </h1>
          <p className="font-sans text-lg md:text-xl text-ink-soft max-w-md mt-8">
            Three taps. No sorting through thousands of photos.
          </p>
          <div className="mt-10 flex items-center gap-4 flex-wrap">
            <a
              href="#step-1"
              className="bg-fresh hover:bg-fresh-deep text-bone px-6 py-3 rounded-full font-mono uppercase tracking-[0.2em] text-sm transition-colors"
            >
              See how →
            </a>
            <a
              href="#cta"
              className="text-ink hover:text-fresh font-mono uppercase tracking-[0.2em] text-sm transition-colors"
            >
              Skip to search
            </a>
          </div>
        </div>
        <div className="hidden md:block">
          <HeroConnector />
        </div>
      </div>
    </section>
  );
}

function HeroConnector() {
  return (
    <svg
      viewBox="0 0 400 280"
      className="w-full h-auto"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M 60 60 Q 200 60 200 140 Q 200 220 340 220"
        stroke="var(--ink)"
        strokeWidth="2"
        strokeLinecap="round"
        className="draw-line"
      />
      <DotNode cx={60} cy={60} label="EVENT" delay="1.2s" />
      <DotNode cx={200} cy={140} label="FACE" delay="1.5s" />
      <DotNode cx={340} cy={220} label="PHOTOS" delay="1.8s" />
    </svg>
  );
}

interface DotNodeProps {
  cx: number;
  cy: number;
  label: string;
  delay: string;
}

function DotNode({ cx, cy, label, delay }: DotNodeProps) {
  return (
    <g style={{ animation: `fade-in 0.6s ${delay} both`, opacity: 0 }}>
      <circle
        cx={cx}
        cy={cy}
        r="10"
        fill="var(--bone)"
        stroke="var(--ink)"
        strokeWidth="2"
      />
      <circle cx={cx} cy={cy} r="4" fill="var(--fresh)" />
      <text
        x={cx}
        y={cy + 28}
        textAnchor="middle"
        fill="var(--slate)"
        fontFamily="var(--font-mono)"
        fontSize="10"
        letterSpacing="2"
      >
        {label}
      </text>
    </g>
  );
}

interface StepSectionProps {
  n: number;
  title: string;
  sub: string;
  screen: React.ReactNode;
  bg: "bone" | "bone-deep";
  flip?: boolean;
}

function StepSection({ n, title, sub, screen, bg, flip = false }: StepSectionProps) {
  const bgClass = bg === "bone-deep" ? "bg-bone-deep" : "bg-bone";
  return (
    <section
      id={`step-${n}`}
      className={`snap-section min-h-screen flex items-center px-6 md:px-10 py-16 md:py-20 ${bgClass}`}
    >
      <div className="max-w-7xl mx-auto w-full grid md:grid-cols-2 gap-12 md:gap-20 items-center">
        <div className={flip ? "md:order-2" : ""}>{screen && <DeviceMock>{screen}</DeviceMock>}</div>
        <div className={`space-y-6 ${flip ? "md:order-1" : ""}`}>
          <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-slate">
            Step
          </p>
          <p className="font-display text-7xl md:text-8xl lg:text-9xl font-medium text-fresh leading-[0.85] tnum">
            0{n}
          </p>
          <h2 className="font-display text-4xl md:text-5xl lg:text-6xl font-medium tracking-tight leading-[1.05]">
            {title}
          </h2>
          <p className="font-sans text-lg md:text-xl text-ink-soft max-w-md">
            {sub}
          </p>
        </div>
      </div>
    </section>
  );
}

function DeviceMock({ children }: { children: React.ReactNode }) {
  return (
    <div className="mx-auto aspect-[9/16] w-full max-w-[260px] md:max-w-[320px] rounded-[2.5rem] bg-ink p-3 shadow-2xl shadow-ink/10">
      <div className="h-full w-full rounded-[2rem] bg-bone overflow-hidden relative">
        {children}
      </div>
    </div>
  );
}

function EventListScreen() {
  const events = [
    { name: "Cebu Marathon 2026", date: "Apr 28", active: true },
    { name: "SRP Half-Marathon", date: "Apr 21" },
    { name: "Sun Run Cebu", date: "Apr 14" },
    { name: "Mactan Coastal 5K", date: "Apr 07" },
  ];
  return (
    <div className="h-full p-5 flex flex-col">
      <div className="flex items-center justify-between mb-5">
        <span className="font-mono uppercase tracking-[0.2em] text-[9px] text-slate">
          Events
        </span>
        <span className="font-mono uppercase tracking-[0.2em] text-[9px] text-slate-soft">
          This week
        </span>
      </div>
      <div className="space-y-2 flex-1">
        {events.map((e, i) => (
          <div
            key={e.name}
            className={`p-3 rounded-xl border transition-all ${
              e.active
                ? "bg-ink text-bone border-ink"
                : "bg-bone-deep/60 text-ink-soft border-line"
            }`}
            style={{ animation: `fade-up 0.5s ${0.1 * i + 0.3}s both`, opacity: 0 }}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="font-display text-sm font-medium tracking-tight">
                {e.name}
              </span>
              {e.active && (
                <span className="size-2 rounded-full bg-fresh shrink-0" />
              )}
            </div>
            <span className="font-mono text-[10px] tracking-wider opacity-70 tnum">
              {e.date}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ScanScreen() {
  return (
    <div className="h-full p-5 flex flex-col items-center justify-center text-center">
      <span className="font-mono uppercase tracking-[0.25em] text-[9px] text-slate mb-6">
        Scan your face
      </span>
      <div className="relative size-32 md:size-36 mb-6 rounded-full border-2 border-dashed border-slate flex items-center justify-center">
        <div className="size-20 md:size-24 rounded-full border-[3px] border-fresh breathe" />
      </div>
      <span className="font-mono uppercase tracking-[0.2em] text-[8px] text-slate-soft mb-3">
        — or —
      </span>
      <div className="border border-line rounded-lg px-4 py-2 font-mono text-sm tracking-widest text-ink tnum">
        BIB · B-4127
      </div>
    </div>
  );
}

function PhotosScreen() {
  return (
    <div className="h-full p-5 flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <span className="font-mono uppercase tracking-[0.2em] text-[9px] text-slate tnum">
          12 photos found
        </span>
        <span className="size-2 rounded-full bg-fresh" />
      </div>
      <div className="grid grid-cols-2 gap-2 flex-1">
        {[0, 1, 2, 3, 4, 5].map((i) => {
          const colors = ["var(--ink-soft)", "var(--slate)", "var(--bone-deep)"];
          return (
            <div
              key={i}
              className="aspect-square rounded-lg"
              style={{
                backgroundColor: colors[i % colors.length],
                animation: `fade-in 0.6s ${0.08 * i + 0.2}s both`,
                opacity: 0,
              }}
            />
          );
        })}
      </div>
      <button className="mt-4 bg-fresh text-bone py-2.5 rounded-full font-mono uppercase tracking-[0.2em] text-[10px]">
        Save all →
      </button>
    </div>
  );
}

function PhotoTeaser() {
  return (
    <section className="snap-section min-h-screen flex items-center px-6 md:px-10 py-16 md:py-20 bg-ink text-bone">
      <div className="max-w-7xl mx-auto w-full">
        <div className="text-center mb-12 md:mb-16">
          <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-slate-soft mb-4">
            Your moments. Captured.
          </p>
          <h2 className="font-display text-4xl md:text-6xl lg:text-7xl font-medium tracking-tight">
            Hundreds of photos.
            <br />
            <span className="text-fresh">Yours in seconds.</span>
          </h2>
        </div>
        <div className="grid grid-cols-3 md:grid-cols-6 gap-2 md:gap-3">
          {Array.from({ length: 12 }).map((_, i) => {
            const colors = [
              "var(--ink-soft)",
              "var(--slate)",
              "var(--bone-deep)",
              "var(--bone)",
            ];
            return (
              <div
                key={i}
                className="aspect-[3/4] rounded-lg"
                style={{
                  backgroundColor: colors[i % colors.length],
                  opacity: 0.7 + (i % 3) * 0.1,
                  animation: `fade-up 0.6s ${0.04 * i}s both`,
                }}
              />
            );
          })}
        </div>
      </div>
    </section>
  );
}

function FinalCta() {
  return (
    <section
      id="cta"
      className="snap-section min-h-[80vh] flex items-center px-6 md:px-10 py-16 md:py-24 bg-bone"
    >
      <div className="max-w-4xl mx-auto w-full text-center">
        <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-slate mb-4">
          Ready when you are
        </p>
        <h2 className="font-display text-5xl md:text-7xl lg:text-8xl font-medium tracking-tight leading-[1.0]">
          Find your
          <br />
          <span className="text-fresh">race photos.</span>
        </h2>
        <p className="font-sans text-lg md:text-xl text-ink-soft mt-6 max-w-md mx-auto">
          Free to search. Pay only for the ones you want.
        </p>
        <div className="mt-10 flex flex-col md:flex-row items-center justify-center gap-4">
          <Link
            href={ROUTES.EVENTS}
            className="w-full md:w-auto bg-fresh hover:bg-fresh-deep text-bone px-8 py-4 rounded-full font-mono uppercase tracking-[0.2em] text-sm transition-colors"
          >
            Find my photos →
          </Link>
          <Link
            href={ROUTES.PHOTOGRAPHERS}
            className="font-mono uppercase tracking-[0.2em] text-xs text-slate hover:text-ink transition-colors"
          >
            I&apos;m a photographer instead
          </Link>
        </div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="px-6 md:px-10 py-8 pb-24 md:pb-8 flex flex-col md:flex-row items-center justify-between gap-4 border-t border-line bg-bone">
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        QuickPitik &middot; Cebu, Philippines
      </p>
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        © 2026
      </p>
    </footer>
  );
}

function MobileStickyCta() {
  return (
    <div className="md:hidden fixed bottom-0 inset-x-0 px-4 py-3 bg-bone/95 backdrop-blur-md border-t border-line z-40">
      <Link
        href={ROUTES.EVENTS}
        className="block w-full bg-fresh active:bg-fresh-deep text-bone text-center py-3.5 rounded-full font-mono uppercase tracking-[0.2em] text-xs"
      >
        Find my photos →
      </Link>
    </div>
  );
}
