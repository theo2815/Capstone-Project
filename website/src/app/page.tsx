import Link from "next/link";
import { ROUTES } from "@/lib/constants";

export default function SplashPage() {
  return (
    <main className="min-h-screen bg-bone flex flex-col">
      <header className="px-6 md:px-10 py-6 md:py-8 flex justify-center">
        <Logo />
      </header>

      <section className="flex-1 flex flex-col justify-center px-6 md:px-10 pb-12 md:pb-16 max-w-6xl mx-auto w-full">
        <div
          className="text-center mb-10 md:mb-16"
          style={{ animation: "fade-up 0.7s ease-out both" }}
        >
          <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-slate mb-4">
            Welcome to QuickPitik
          </p>
          <h1 className="font-display text-4xl md:text-6xl lg:text-7xl font-medium tracking-tight text-ink leading-[1.05]">
            Find your race photos.
            <br />
            <span className="text-slate font-light">Or sell yours.</span>
          </h1>
        </div>

        <div className="grid md:grid-cols-2 gap-4 md:gap-6">
          <ChooserCard
            href={ROUTES.RUNNERS}
            kicker="For Runners"
            title="Find my photos"
            sub="Search by face or bib in seconds."
            icon={<RunnerIcon />}
            delay="0.15s"
          />
          <ChooserCard
            href={ROUTES.PHOTOGRAPHERS}
            kicker="For Photographers"
            title="Sell my photos"
            sub="Upload in real time. List and earn."
            icon={<CameraIcon />}
            delay="0.25s"
          />
        </div>
      </section>

      <footer className="px-6 md:px-10 py-6 md:py-8 flex justify-center">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
          Cebu &middot; Philippines
        </p>
      </footer>
    </main>
  );
}

function Logo() {
  return (
    <Link
      href={ROUTES.HOME}
      className="flex items-center gap-2 text-ink"
      aria-label="QuickPitik home"
    >
      <svg className="size-7" viewBox="0 0 28 28" fill="none" aria-hidden="true">
        <circle cx="14" cy="14" r="13" stroke="currentColor" strokeWidth="1.5" />
        <circle cx="14" cy="14" r="5" className="fill-fresh" />
      </svg>
      <span className="font-display text-lg font-semibold tracking-tight">
        QuickPitik
      </span>
    </Link>
  );
}

interface ChooserCardProps {
  href: string;
  kicker: string;
  title: string;
  sub: string;
  icon: React.ReactNode;
  delay: string;
}

function ChooserCard({
  href,
  kicker,
  title,
  sub,
  icon,
  delay,
}: ChooserCardProps) {
  return (
    <Link
      href={href}
      className="group relative block rounded-3xl bg-bone-deep border border-line hover:border-ink/40 hover:-translate-y-1 transition-all duration-500 p-8 md:p-10 lg:p-12 min-h-[280px] md:min-h-[420px] flex flex-col overflow-hidden"
      style={{ animation: `fade-up 0.7s ${delay} both` }}
    >
      <div className="flex-1 flex flex-col">
        <div className="text-ink mb-8 md:mb-12 breathe inline-block">
          {icon}
        </div>
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] md:text-[11px] text-slate mb-3">
          {kicker}
        </p>
        <h2 className="font-display text-3xl md:text-4xl lg:text-5xl font-medium tracking-tight text-ink leading-[1.05] mb-3">
          {title}
        </h2>
        <p className="font-sans text-base md:text-lg text-ink-soft max-w-sm">
          {sub}
        </p>
      </div>
      <div className="mt-8 md:mt-12 flex items-center justify-between">
        <span className="font-mono uppercase tracking-[0.25em] text-xs text-ink group-hover:text-fresh transition-colors">
          Continue
        </span>
        <ArrowRight />
      </div>
    </Link>
  );
}

function ArrowRight() {
  return (
    <svg
      className="size-5 text-ink group-hover:text-fresh group-hover:translate-x-1 transition-all duration-300"
      viewBox="0 0 20 20"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M4 10h12m0 0l-5-5m5 5l-5 5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function RunnerIcon() {
  return (
    <svg
      className="size-16 md:size-20"
      viewBox="0 0 80 80"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M6 36 L22 36"
        className="stroke-fresh"
        strokeWidth="3"
        strokeLinecap="round"
      />
      <path
        d="M10 46 L24 46"
        className="stroke-fresh"
        strokeWidth="3"
        strokeLinecap="round"
        opacity="0.5"
      />
      <circle cx="50" cy="14" r="6" fill="currentColor" />
      <path
        d="M50 22 L46 40"
        stroke="currentColor"
        strokeWidth="3.5"
        strokeLinecap="round"
      />
      <path
        d="M48 26 L62 22"
        stroke="currentColor"
        strokeWidth="3.5"
        strokeLinecap="round"
      />
      <path
        d="M48 28 L36 32"
        stroke="currentColor"
        strokeWidth="3.5"
        strokeLinecap="round"
      />
      <path
        d="M46 40 L58 50 L52 64"
        stroke="currentColor"
        strokeWidth="3.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      <path
        d="M46 40 L34 56 L26 64"
        stroke="currentColor"
        strokeWidth="3.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}

function CameraIcon() {
  return (
    <svg
      className="size-16 md:size-20"
      viewBox="0 0 80 80"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M26 22 L32 14 L48 14 L54 22 Z"
        stroke="currentColor"
        strokeWidth="3"
        fill="none"
        strokeLinejoin="round"
      />
      <rect
        x="10"
        y="22"
        width="60"
        height="42"
        rx="5"
        stroke="currentColor"
        strokeWidth="3"
      />
      <circle cx="40" cy="44" r="14" stroke="currentColor" strokeWidth="3" />
      <circle cx="40" cy="44" r="7" className="fill-fresh" />
      <circle cx="40" cy="44" r="2.5" fill="currentColor" />
      <circle cx="60" cy="30" r="2" fill="currentColor" />
    </svg>
  );
}
