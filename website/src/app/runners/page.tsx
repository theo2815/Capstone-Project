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
    <main className="snap-y-mobile bg-paper text-ink relative">
      <SiteHeader />

      <Hero />

      <StepSection
        n={1}
        title="Pick your event."
        sub="Tap one. That's it."
        screen={<EventListScreen />}
        bg="paper"
      />

      <StepSection
        n={2}
        title="Show us your face."
        sub="Or just type your bib number."
        screen={<ScanScreen />}
        bg="paper-deep"
        flip
      />

      <StepSection
        n={3}
        title="Save them all."
        sub="Yours to keep, share, or print."
        screen={<PhotosScreen />}
        bg="paper"
      />

      <PhotoTeaser />

      <FinalCta />

      <Footer />

      <MobileStickyCta />
    </main>
  );
}

function Kicker({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3">
      <span className="race-stripe" aria-hidden="true">
        <span className="bg-fresh" />
        <span className="bg-fresh-deep" />
        <span className="bg-ink" />
      </span>
      <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
        {children}
      </p>
    </div>
  );
}

function PrimaryCta({
  href,
  children,
}: {
  href: string;
  children: React.ReactNode;
}) {
  return (
    <Link
      href={href}
      className="group inline-flex items-center gap-2 rounded-full bg-fresh px-6 py-3.5 font-display font-bold text-[15px] text-surface transition-colors hover:bg-fresh-deep"
    >
      {children}
      <Arrow />
    </Link>
  );
}

function Arrow() {
  return (
    <svg
      className="size-4 transition-transform duration-300 group-hover:translate-x-1"
      viewBox="0 0 20 20"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M4 10h12m0 0l-5-5m5 5l-5 5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function Hero() {
  return (
    <section className="snap-section md:min-h-[calc(100vh-var(--site-header-h))] flex items-start md:items-center px-6 md:px-10 pt-10 pb-14 md:py-20 bg-paper">
      <div className="max-w-7xl mx-auto w-full grid md:grid-cols-2 gap-12 md:gap-16 items-center">
        <div className="stagger-children">
          <Kicker>For runners</Kicker>
          <h1 className="font-hero text-ink text-5xl md:text-7xl lg:text-8xl mt-5">
            From finish line
            <br />
            <span className="text-fresh">to your phone.</span>
          </h1>
          <p className="font-sans text-lg md:text-xl text-ink-soft max-w-md mt-7 leading-relaxed">
            Three taps. No sorting through thousands of photos.
          </p>
          <div className="mt-10 flex items-center gap-5 flex-wrap">
            <a
              href="#step-1"
              className="group inline-flex items-center gap-2 rounded-full bg-fresh px-6 py-3.5 font-display font-bold text-[15px] text-surface transition-colors hover:bg-fresh-deep"
            >
              See how
              <Arrow />
            </a>
            <a
              href="#cta"
              className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors"
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
        fill="var(--paper)"
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
  bg: "paper" | "paper-deep";
  flip?: boolean;
}

function StepSection({ n, title, sub, screen, bg, flip = false }: StepSectionProps) {
  const bgClass = bg === "paper-deep" ? "bg-paper-deep" : "bg-paper";
  return (
    <section
      id={`step-${n}`}
      className={`snap-section min-h-screen flex items-center px-6 md:px-10 py-16 md:py-20 ${bgClass}`}
    >
      <div className="max-w-7xl mx-auto w-full grid md:grid-cols-2 gap-12 md:gap-20 items-center">
        <div className={flip ? "md:order-2" : ""}>{screen && <DeviceMock>{screen}</DeviceMock>}</div>
        <div className={`space-y-5 ${flip ? "md:order-1" : ""}`}>
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
            Step
          </p>
          <p className="font-hero text-fresh text-8xl md:text-9xl leading-[0.8] tnum">
            0{n}
          </p>
          <h2 className="font-display font-extrabold text-4xl md:text-5xl lg:text-6xl tracking-tight leading-[1.05]">
            {title}
          </h2>
          <p className="font-sans text-lg md:text-xl text-ink-soft max-w-md leading-relaxed">
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
      <div className="h-full w-full rounded-[2rem] bg-paper overflow-hidden relative">
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
        <span className="font-mono uppercase tracking-[0.14em] text-[9px] text-slate">
          Events
        </span>
        <span className="font-mono uppercase tracking-[0.14em] text-[9px] text-slate-soft">
          This week
        </span>
      </div>
      <div className="space-y-2 flex-1">
        {events.map((e, i) => (
          <div
            key={e.name}
            className={`p-3 rounded-xl border transition-all ${
              e.active
                ? "bg-ink text-paper border-ink"
                : "bg-paper-deep/60 text-ink-soft border-line"
            }`}
            style={{ animation: `fade-up 0.5s ${0.1 * i + 0.3}s both`, opacity: 0 }}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="font-display text-sm font-semibold tracking-tight">
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
      <span className="font-mono uppercase tracking-[0.14em] text-[9px] text-slate mb-6">
        Scan your face
      </span>
      <div className="relative size-32 md:size-36 mb-6 rounded-full border-2 border-dashed border-slate flex items-center justify-center">
        <div className="size-20 md:size-24 rounded-full border-[3px] border-fresh breathe" />
      </div>
      <span className="font-mono uppercase tracking-[0.14em] text-[8px] text-slate-soft mb-3">
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
        <span className="font-mono uppercase tracking-[0.14em] text-[9px] text-slate tnum">
          12 photos found
        </span>
        <span className="size-2 rounded-full bg-fresh" />
      </div>
      <div className="grid grid-cols-2 gap-2 flex-1">
        {[0, 1, 2, 3, 4, 5].map((i) => {
          const colors = ["var(--ink-soft)", "var(--slate)", "var(--paper-deep)"];
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
      <button className="mt-4 bg-fresh text-surface py-2.5 rounded-full font-mono uppercase tracking-[0.14em] text-[10px]">
        Save all →
      </button>
    </div>
  );
}

function PhotoTeaser() {
  return (
    <section className="snap-section min-h-screen flex items-center px-6 md:px-10 py-16 md:py-20 bg-paper-deep text-ink">
      <div className="max-w-7xl mx-auto w-full">
        <div className="text-center mb-12 md:mb-16 flex flex-col items-center">
          <div className="race-stripe mb-4" aria-hidden="true">
            <span className="bg-fresh" />
            <span className="bg-fresh-deep" />
            <span className="bg-ink" />
          </div>
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate mb-4">
            Your moments. Captured.
          </p>
          <h2 className="font-hero text-ink text-5xl md:text-7xl lg:text-8xl">
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
              "var(--fresh)",
              "var(--pine)",
            ];
            return (
              <div
                key={i}
                className="aspect-[3/4] rounded-lg"
                style={{
                  backgroundColor: colors[i % colors.length],
                  opacity: 0.85,
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
      className="snap-section min-h-[80vh] flex items-center px-6 md:px-10 py-16 md:py-24 bg-paper"
    >
      <div className="max-w-4xl mx-auto w-full text-center flex flex-col items-center">
        <div className="race-stripe mb-4" aria-hidden="true">
          <span className="bg-fresh" />
          <span className="bg-fresh-deep" />
          <span className="bg-ink" />
        </div>
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate mb-4">
          Ready when you are
        </p>
        <h2 className="font-hero text-ink text-5xl md:text-7xl lg:text-8xl">
          Find your
          <br />
          <span className="text-fresh">race photos.</span>
        </h2>
        <p className="font-sans text-lg md:text-xl text-ink-soft mt-6 max-w-md">
          Free to search. Pay only for the ones you want.
        </p>
        <div className="mt-10 flex flex-col md:flex-row items-center justify-center gap-5">
          <PrimaryCta href={ROUTES.EVENTS}>Find my photos</PrimaryCta>
          <Link
            href={ROUTES.PHOTOGRAPHERS}
            className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors"
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
    <footer className="px-6 md:px-10 py-8 pb-24 md:pb-8 flex flex-col md:flex-row items-center justify-between gap-4 border-t border-line bg-paper">
      <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
        QuickPitik &middot; Cebu, Philippines
      </p>
      <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
        © 2026
      </p>
    </footer>
  );
}

function MobileStickyCta() {
  return (
    <div
      className="md:hidden fixed bottom-0 inset-x-0 px-4 pt-3 bg-paper/95 backdrop-blur-md border-t border-line z-40"
      style={{ paddingBottom: "calc(0.75rem + env(safe-area-inset-bottom))" }}
    >
      <Link
        href={ROUTES.EVENTS}
        className="flex w-full items-center justify-center gap-2 bg-fresh active:bg-fresh-deep text-surface text-center py-3.5 rounded-full font-display font-bold text-[15px]"
      >
        Find my photos
        <Arrow />
      </Link>
    </div>
  );
}
