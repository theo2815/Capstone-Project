import Link from "next/link";
import { BrandLogo } from "@/components/layout/brand-logo";
import { ROUTES } from "@/lib/constants";

export default function SplashPage() {
  return (
    <main className="min-h-screen bg-paper flex flex-col">
      {/* ── Top bar ── */}
      <header className="border-b border-line">
        <div className="max-w-6xl mx-auto w-full px-5 md:px-10 h-16 flex items-center justify-between">
          <Link href={ROUTES.HOME} aria-label="QuickPitik home">
            <BrandLogo className="h-11 w-44" priority />
          </Link>
          <p className="hidden sm:block font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
            Cebu · Philippines
          </p>
        </div>
        <div className="h-1 w-full flex">
          <span className="flex-1 bg-fresh" />
          <span className="flex-1 bg-fresh-deep" />
          <span className="flex-1 bg-pine" />
        </div>
      </header>

      {/* ── Hero ── */}
      <section className="flex-1 flex flex-col justify-center px-5 md:px-10 py-12 md:py-16 max-w-6xl mx-auto w-full">
        <div
          className="mb-10 md:mb-14"
          style={{ animation: "fade-up 0.7s ease-out both" }}
        >
          <div className="flex items-center gap-3 mb-5">
            <span className="race-stripe" aria-hidden="true">
              <span className="bg-fresh" />
              <span className="bg-fresh-deep" />
              <span className="bg-ink" />
            </span>
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
              Race photo marketplace
            </p>
          </div>

          <h1 className="font-hero text-ink text-[15vw] sm:text-7xl md:text-8xl lg:text-9xl">
            Find your
            <br />
            <span className="text-fresh">race photos.</span>
            <br />
            <span className="text-slate-soft">Or sell yours.</span>
          </h1>

          <p className="mt-6 font-sans text-lg md:text-xl text-ink-soft max-w-xl leading-relaxed">
            Marathon shots delivered minutes after the finish line — searchable
            by face or bib in seconds. One home for runners and the
            photographers who shoot them.
          </p>
        </div>

        {/* ── Two-path chooser ── */}
        <div className="grid md:grid-cols-2 gap-4 md:gap-6">
          <RunnerCard />
          <PhotographerCard />
        </div>
      </section>

      <footer className="px-5 md:px-10 py-7 max-w-6xl mx-auto w-full flex items-center justify-between border-t border-line">
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
          QuickPitik · 2026
        </p>
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
          Cebu · PH
        </p>
      </footer>
    </main>
  );
}

/* ───────────────────────────── Chooser cards ───────────────────────────── */

function RunnerCard() {
  return (
    <Link
      href={ROUTES.RUNNERS}
      className="group relative flex flex-col overflow-hidden rounded-[var(--radius-xl)] bg-surface border border-line p-7 md:p-9 min-h-[300px] md:min-h-[420px] shadow-[var(--shadow-card)] transition-all duration-500 hover:-translate-y-1.5 hover:shadow-[var(--shadow-lift)] hover:border-fresh/50"
      style={{ animation: "fade-up 0.7s 0.15s both" }}
    >
      <div className="flex items-start justify-between">
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-fresh">
          For runners
        </p>
        <span className="text-fresh">
          <RunnerIcon />
        </span>
      </div>

      <div className="mt-auto pt-10">
        <p className="font-mono text-[13px] text-slate tnum">Search time</p>
        <p className="font-hero text-fresh text-6xl md:text-7xl mt-1 tnum leading-none">
          5–10<span className="text-2xl md:text-3xl ml-2 align-top">sec</span>
        </p>
        <h2 className="font-display font-extrabold text-3xl md:text-4xl text-ink tracking-tight mt-5">
          Find my photos
        </h2>
        <p className="font-sans text-base text-ink-soft mt-2 max-w-xs">
          Snap a selfie or type your bib. We surface every shot of you.
        </p>
      </div>

      <div className="mt-7">
        <span className="inline-flex items-center gap-2 rounded-full bg-fresh px-5 py-2.5 font-display font-bold text-[15px] text-surface transition-colors group-hover:bg-fresh-deep">
          Find yours
          <Arrow />
        </span>
      </div>
    </Link>
  );
}

function PhotographerCard() {
  return (
    <Link
      href={ROUTES.PHOTOGRAPHERS}
      className="group relative flex flex-col overflow-hidden rounded-[var(--radius-xl)] bg-fresh-tint border border-fresh/25 p-7 md:p-9 min-h-[300px] md:min-h-[420px] shadow-[var(--shadow-card)] transition-all duration-500 hover:-translate-y-1.5 hover:shadow-[var(--shadow-lift)] hover:border-fresh/50"
      style={{ animation: "fade-up 0.7s 0.25s both" }}
    >
      <div className="flex items-start justify-between">
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-fresh-deep">
          For photographers
        </p>
        <span className="text-pine">
          <CameraIcon />
        </span>
      </div>

      <div className="mt-auto pt-10">
        <p className="font-mono text-[13px] text-slate tnum">Sort time</p>
        <p className="font-hero text-6xl md:text-7xl mt-1 tnum leading-none">
          <span className="text-slate-soft">1–2</span>
          <span className="text-xl md:text-2xl ml-2 align-top text-slate-soft">hrs</span>
          <span className="mx-2 text-slate-soft/50">→</span>
          <span className="text-fresh-deep">10s</span>
        </p>
        <h2 className="font-display font-extrabold text-3xl md:text-4xl text-ink tracking-tight mt-5">
          Sell my photos
        </h2>
        <p className="font-sans text-base text-ink-soft mt-2 max-w-xs">
          Upload in real time, cull the blur, get found and paid.
        </p>
      </div>

      <div className="mt-7">
        <span className="inline-flex items-center gap-2 rounded-full bg-fresh-deep px-5 py-2.5 font-display font-bold text-[15px] text-surface transition-colors group-hover:bg-pine">
          Start selling
          <Arrow />
        </span>
      </div>
    </Link>
  );
}

/* ───────────────────────────── Bits ───────────────────────────── */

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

function RunnerIcon() {
  return (
    <svg
      className="size-12 md:size-14"
      viewBox="0 0 80 80"
      fill="none"
      aria-hidden="true"
    >
      <path d="M6 36 L22 36" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      <path d="M10 46 L24 46" stroke="currentColor" strokeWidth="3" strokeLinecap="round" opacity="0.5" />
      <circle cx="50" cy="14" r="6" fill="currentColor" />
      <path d="M50 22 L46 40" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" />
      <path d="M48 26 L62 22" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" />
      <path d="M48 28 L36 32" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" />
      <path d="M46 40 L58 50 L52 64" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <path d="M46 40 L34 56 L26 64" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );
}

function CameraIcon() {
  return (
    <svg
      className="size-12 md:size-14"
      viewBox="0 0 80 80"
      fill="none"
      aria-hidden="true"
    >
      <path d="M26 22 L32 14 L48 14 L54 22 Z" stroke="currentColor" strokeWidth="3" fill="none" strokeLinejoin="round" />
      <rect x="10" y="22" width="60" height="42" rx="5" stroke="currentColor" strokeWidth="3" />
      <circle cx="40" cy="44" r="14" stroke="currentColor" strokeWidth="3" />
      <circle cx="40" cy="44" r="7" className="fill-fresh" />
      <circle cx="40" cy="44" r="2.5" fill="currentColor" />
      <circle cx="60" cy="30" r="2" fill="currentColor" />
    </svg>
  );
}
