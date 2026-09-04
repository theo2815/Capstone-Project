import Link from "next/link";
import type { Metadata } from "next";
import { ROUTES } from "@/lib/constants";
import { SiteHeader } from "@/components/layout/site-header";

const REGISTER_AS_PHOTOGRAPHER = `${ROUTES.REGISTER}?role=PHOTOGRAPHER`;

export const metadata: Metadata = {
  title: "QuickPitik — Sell your race photos",
  description:
    "Two ways to sell. Tether your camera and stream live, or upload after the event.",
};

const BATCH_MY_PHOTOS_URL = "https://www.batchmyphotos.com/";

export default function PhotographersPage() {
  return (
    <main className="snap-y-mobile bg-paper text-ink relative">
      <SiteHeader />

      <Hero />

      <ChapterLabel
        kicker="Recommended · Mobile app"
        title="Just shoot."
        sub="Photos sell themselves while you work."
      />

      <StepSection
        id="step-1"
        n={1}
        title="Connect your phone to your camera."
        sub="One cable. Or wireless. Tether takes seconds."
        screen={<ConnectScreen />}
        bg="bone"
      />

      <StepSection
        id="step-2"
        n={2}
        title="Every shot lands in the marketplace."
        sub="No SD cards. No transfer. Photos go live as you press the shutter."
        screen={<StreamScreen />}
        bg="bone-deep"
        flip
      />

      <StepSection
        id="step-3"
        n={3}
        title="Get paid."
        sub="Runners buy. Payouts hit weekly."
        screen={<EarnScreen />}
        bg="bone"
      />

      <Pivot />

      <StepSection
        id="alt-1"
        n={1}
        title="Drop photos into the gallery."
        sub="Upload from your laptop. 500 photos per upload."
        screen={<UploadScreen />}
        bg="bone"
        mockType="laptop"
      />

      <StepSection
        id="alt-2"
        n={2}
        title="Got thousands? Let the desktop sort them."
        sub="BatchMyPhotos splits 10,000 photos into folders of 500 in seconds. Then drag them into the uploader."
        screen={<BatchScreen />}
        bg="bone-deep"
        flip
        mockType="laptop"
        cta={
          <a
            href={BATCH_MY_PHOTOS_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 bg-ink hover:bg-ink-soft text-bone px-6 py-3 rounded-full font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] transition-colors"
          >
            Get the desktop app
            <svg
              className="size-3"
              viewBox="0 0 12 12"
              fill="none"
              aria-hidden="true"
            >
              <path
                d="M3.5 8.5 L8.5 3.5 M5 3.5 L8.5 3.5 L8.5 7"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </a>
        }
      />

      <FinalCta />

      <Footer />

      <MobileStickyCta />
    </main>
  );
}

function Kicker({
  children,
  color = "slate",
}: {
  children: React.ReactNode;
  color?: "slate" | "fresh";
}) {
  return (
    <div className="flex items-center gap-3 justify-center">
      <span className="race-stripe" aria-hidden="true">
        <span className="bg-fresh" />
        <span className="bg-fresh-deep" />
        <span className="bg-ink" />
      </span>
      <p
        className={`font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] ${
          color === "fresh" ? "text-fresh" : "text-slate"
        }`}
      >
        {children}
      </p>
    </div>
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

function Hero() {
  return (
    <section className="snap-section md:min-h-[calc(100vh-var(--site-header-h))] flex items-start md:items-center px-6 md:px-10 pt-10 pb-14 md:py-20 bg-paper">
      <div className="max-w-7xl mx-auto w-full grid md:grid-cols-2 gap-12 md:gap-16 items-center">
        <div className="stagger-children">
          <div className="flex items-center gap-3">
            <span className="race-stripe" aria-hidden="true">
              <span className="bg-fresh" />
              <span className="bg-fresh-deep" />
              <span className="bg-ink" />
            </span>
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
              For photographers
            </p>
          </div>
          <h1 className="font-hero text-ink text-5xl md:text-7xl lg:text-8xl mt-5">
            Stop sorting.
            <br />
            <span className="text-fresh">Start selling.</span>
          </h1>
          <p className="font-sans text-lg md:text-xl text-ink-soft max-w-md mt-7 leading-relaxed">
            Two ways to upload. Both go straight to the marketplace.
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
              Skip to signup
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
      viewBox="0 0 400 300"
      className="w-full h-auto"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M 50 40 Q 150 40 150 110 Q 150 180 250 180 Q 350 180 350 250"
        stroke="var(--ink)"
        strokeWidth="2"
        strokeLinecap="round"
        className="draw-line"
        style={{ "--dash": "700" } as React.CSSProperties}
      />
      <DotNode cx={50} cy={40} label="CAMERA" delay="1.2s" />
      <DotNode cx={150} cy={110} label="CLOUD" delay="1.5s" />
      <DotNode cx={250} cy={180} label="MARKET" delay="1.8s" />
      <DotNode cx={350} cy={250} label="PAID" delay="2.1s" />
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

interface ChapterLabelProps {
  kicker: string;
  title: string;
  sub: string;
}

function ChapterLabel({ kicker, title, sub }: ChapterLabelProps) {
  return (
    <section className="snap-section py-16 md:py-24 px-6 md:px-10 bg-paper border-y border-line">
      <div className="max-w-3xl mx-auto text-center stagger-children flex flex-col items-center">
        <Kicker color="fresh">{kicker}</Kicker>
        <h2 className="font-hero text-ink text-5xl md:text-7xl lg:text-8xl mt-5">
          {title}
        </h2>
        <p className="font-sans text-base md:text-lg text-ink-soft mt-5 max-w-md mx-auto">
          {sub}
        </p>
      </div>
    </section>
  );
}

function Pivot() {
  return (
    <section className="snap-section py-20 md:py-28 px-6 md:px-10 bg-paper-deep border-y border-line">
      <div className="max-w-3xl mx-auto text-center stagger-children flex flex-col items-center">
        <Kicker>No mobile app?</Kicker>
        <h2 className="font-hero text-ink text-5xl md:text-7xl lg:text-8xl mt-5">
          Upload it manually.
        </h2>
        <p className="font-sans text-base md:text-lg text-ink-soft mt-5 max-w-md mx-auto">
          Two more steps. Same marketplace.
        </p>
        <svg
          className="size-6 mx-auto mt-10 text-slate"
          viewBox="0 0 24 24"
          fill="none"
          aria-hidden="true"
        >
          <path
            d="M12 4 L12 20 M5 13 L12 20 L19 13"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    </section>
  );
}

interface StepSectionProps {
  id: string;
  n: number;
  title: string;
  sub: string;
  screen: React.ReactNode;
  bg: "bone" | "bone-deep";
  flip?: boolean;
  cta?: React.ReactNode;
  mockType?: "phone" | "laptop";
}

function StepSection({
  id,
  n,
  title,
  sub,
  screen,
  bg,
  flip = false,
  cta,
  mockType = "phone",
}: StepSectionProps) {
  const bgClass = bg === "bone-deep" ? "bg-paper-deep" : "bg-paper";
  const Mock = mockType === "laptop" ? LaptopMock : DeviceMock;
  return (
    <section
      id={id}
      className={`snap-section min-h-screen flex items-center px-6 md:px-10 py-16 md:py-20 ${bgClass}`}
    >
      <div className="max-w-7xl mx-auto w-full grid md:grid-cols-2 gap-12 md:gap-20 items-center">
        <div className={flip ? "md:order-2" : ""}>
          <Mock>{screen}</Mock>
        </div>
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
          {cta && <div className="pt-2">{cta}</div>}
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

function LaptopMock({ children }: { children: React.ReactNode }) {
  return (
    <div className="mx-auto w-full max-w-[440px] md:max-w-[520px]">
      {/* lid: screen */}
      <div className="aspect-[16/10] rounded-xl bg-ink p-2 shadow-2xl shadow-ink/10">
        <div className="h-full w-full rounded-md bg-bone overflow-hidden relative">
          {children}
        </div>
      </div>
      {/* hinge + base */}
      <div className="relative -mx-[6%] mt-0.5">
        <div className="h-2.5 bg-ink rounded-b-2xl" />
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-20 h-1 bg-slate-soft/30 rounded-b-md" />
      </div>
    </div>
  );
}

function ConnectScreen() {
  return (
    <div className="h-full p-5 flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <span className="font-mono uppercase tracking-[0.14em] text-[9px] text-fresh flex items-center gap-1.5">
          <span className="size-1.5 rounded-full bg-fresh breathe" />
          Tethered
        </span>
        <span className="font-mono text-[10px] text-slate">USB · WiFi</span>
      </div>

      <div className="flex-1 flex items-center justify-center">
        <svg
          viewBox="0 0 240 160"
          className="w-full"
          fill="none"
          aria-hidden="true"
        >
          {/* camera */}
          <rect
            x="14"
            y="58"
            width="78"
            height="58"
            rx="6"
            stroke="var(--ink)"
            strokeWidth="2.5"
            fill="var(--bone-deep)"
          />
          <path
            d="M30 58 L38 46 L60 46 L68 58 Z"
            stroke="var(--ink)"
            strokeWidth="2.5"
            fill="var(--bone-deep)"
            strokeLinejoin="round"
          />
          <circle
            cx="53"
            cy="87"
            r="16"
            stroke="var(--ink)"
            strokeWidth="2.5"
            fill="var(--bone)"
          />
          <circle cx="53" cy="87" r="7" fill="var(--fresh)" />
          <rect
            x="76"
            y="65"
            width="6"
            height="3"
            rx="1"
            fill="var(--ink)"
          />

          {/* connection line (dashed → animated) */}
          <path
            d="M92 87 L148 87"
            stroke="var(--ink)"
            strokeWidth="2"
            strokeDasharray="4 4"
            strokeLinecap="round"
          />

          {/* phone */}
          <rect
            x="148"
            y="34"
            width="78"
            height="106"
            rx="12"
            stroke="var(--ink)"
            strokeWidth="2.5"
            fill="var(--bone-deep)"
          />
          <rect
            x="156"
            y="46"
            width="62"
            height="80"
            rx="3"
            fill="var(--ink)"
          />
          <circle cx="187" cy="133" r="2.5" fill="var(--ink)" />

          {/* phone screen content */}
          <rect x="162" y="52" width="50" height="6" rx="1" fill="var(--fresh)" />
          <rect x="162" y="62" width="34" height="3" rx="1" fill="var(--bone-deep)" opacity="0.5" />
          <rect x="162" y="70" width="22" height="22" rx="2" fill="var(--bone-deep)" opacity="0.6" />
          <rect x="188" y="70" width="22" height="22" rx="2" fill="var(--bone-deep)" opacity="0.4" />
          <rect x="162" y="96" width="22" height="22" rx="2" fill="var(--bone-deep)" opacity="0.5" />
          <rect x="188" y="96" width="22" height="22" rx="2" fill="var(--bone-deep)" opacity="0.7" />
        </svg>
      </div>

      <div className="mt-4 px-3 py-2 rounded-lg bg-fresh/10 border border-fresh/30 flex items-center justify-between">
        <span className="font-mono uppercase tracking-[0.14em] text-[9px] text-fresh">
          Connected
        </span>
        <span className="font-mono text-[10px] text-ink">Canon R5</span>
      </div>
    </div>
  );
}

function StreamScreen() {
  return (
    <div className="h-full p-5 flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <span className="font-mono uppercase tracking-[0.14em] text-[9px] text-fresh flex items-center gap-1.5">
          <span className="size-1.5 rounded-full bg-fresh" />
          Live
        </span>
        <span className="font-mono text-[10px] text-slate tnum">412 ↗</span>
      </div>

      <span className="font-mono uppercase tracking-[0.14em] text-[8px] text-slate mb-2">
        Streaming to marketplace
      </span>

      <div className="grid grid-cols-3 gap-1.5 flex-1 mb-4">
        {Array.from({ length: 9 }).map((_, i) => {
          const colors = ["var(--ink-soft)", "var(--slate)", "var(--bone-deep)"];
          return (
            <div
              key={i}
              className="aspect-square rounded-md relative"
              style={{
                backgroundColor: colors[i % colors.length],
                animation: `fade-up 0.5s ${0.07 * i + 0.2}s both`,
                opacity: 0,
              }}
            >
              {i < 3 && (
                <div
                  className="absolute top-1 right-1 size-3 rounded-full bg-fresh flex items-center justify-center"
                  style={{ animation: `fade-in 0.4s ${0.07 * i + 0.6}s both` }}
                >
                  <svg
                    className="size-2 text-bone"
                    viewBox="0 0 8 8"
                    fill="none"
                    aria-hidden="true"
                  >
                    <path
                      d="M1.5 4 L3.5 6 L6.5 2"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div
        className="px-3 py-2 rounded-lg bg-fresh/10 border border-fresh/30 flex items-center justify-between"
        style={{ animation: "fade-up 0.5s 0.9s both" }}
      >
        <span className="font-mono uppercase tracking-[0.14em] text-[9px] text-fresh">
          On marketplace
        </span>
        <span className="font-mono text-[10px] text-ink tnum">412 listed</span>
      </div>
    </div>
  );
}

function EarnScreen() {
  return (
    <div className="h-full p-5 flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <span className="font-mono uppercase tracking-[0.14em] text-[9px] text-slate">
          This week
        </span>
        <span className="font-mono text-[10px] text-slate-soft tnum">+12 sold</span>
      </div>

      <div className="mb-5">
        <p className="font-mono uppercase tracking-[0.14em] text-[8px] text-slate mb-1">
          Earnings
        </p>
        <p
          className="font-display text-5xl font-semibold tracking-tight text-ink leading-none tnum"
          style={{ animation: "count-up 0.8s 0.4s both" }}
        >
          ₱4,250
        </p>
      </div>

      <div className="space-y-3 flex-1">
        {[
          { d: "Mon", v: 35 },
          { d: "Tue", v: 60 },
          { d: "Wed", v: 28 },
          { d: "Thu", v: 78 },
          { d: "Fri", v: 92 },
          { d: "Sat", v: 100 },
        ].map((b, i) => (
          <div key={b.d} className="flex items-center gap-3">
            <span className="font-mono text-[9px] text-slate w-8 uppercase">
              {b.d}
            </span>
            <div className="flex-1 h-2 bg-bone-deep rounded-full overflow-hidden">
              <div
                className="h-full bg-fresh rounded-full"
                style={{
                  width: `${b.v}%`,
                  animation: `fade-in 0.6s ${0.1 * i + 0.5}s both`,
                }}
              />
            </div>
          </div>
        ))}
      </div>

      <button className="mt-4 bg-ink text-bone py-2.5 rounded-full font-mono uppercase tracking-[0.14em] text-[10px]">
        View payouts →
      </button>
    </div>
  );
}

function UploadScreen() {
  return (
    <div className="h-full flex flex-col">
      {/* browser chrome */}
      <div className="flex items-center gap-1.5 px-3 py-2 bg-bone-deep border-b border-line">
        <div className="size-1.5 rounded-full bg-slate-soft/60" />
        <div className="size-1.5 rounded-full bg-slate-soft/60" />
        <div className="size-1.5 rounded-full bg-slate-soft/60" />
        <div className="ml-3 flex-1 h-3 bg-bone rounded-sm border border-line flex items-center max-w-[70%]">
          <span className="font-mono text-[7px] text-slate-soft px-2 leading-none truncate">
            quickpitik.com/upload
          </span>
        </div>
      </div>

      {/* main */}
      <div className="flex-1 p-3 md:p-4 flex flex-col">
        <div className="flex items-center justify-between mb-2">
          <span className="font-display text-sm font-semibold tracking-tight">
            Upload photos
          </span>
          <span className="font-mono text-[12px] text-fresh tnum">247 / 500</span>
        </div>

        {/* drop zone */}
        <div className="flex-1 rounded-md border-2 border-dashed border-slate-soft flex flex-col items-center justify-center bg-bone-deep/40">
          <svg
            className="size-8 md:size-10 text-ink mb-1.5"
            viewBox="0 0 60 60"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M30 12 L30 38 M30 12 L20 22 M30 12 L40 22"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M12 44 L12 50 L48 50 L48 44"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span className="font-mono uppercase tracking-[0.14em] text-[8px] md:text-[9px] text-slate mb-0.5">
            Drag photos here
          </span>
          <span className="font-mono text-[7px] md:text-[8px] text-slate-soft tracking-wider">
            or click to browse
          </span>
        </div>

        {/* progress */}
        <div className="mt-2.5">
          <div className="flex items-center justify-between mb-1">
            <span className="font-mono uppercase tracking-[0.14em] text-[7px] md:text-[8px] text-slate">
              Per upload limit
            </span>
            <span className="font-mono text-[9px] text-ink tnum">500 max</span>
          </div>
          <div className="h-1 bg-bone-deep rounded-full overflow-hidden">
            <div
              className="h-full bg-fresh rounded-full"
              style={{ width: "49%", animation: "fade-in 1s 0.5s both" }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function BatchScreen() {
  return (
    <div className="h-full flex flex-col">
      {/* app chrome (desktop app, dark) */}
      <div className="flex items-center gap-1.5 px-3 py-1.5 bg-ink border-b border-ink-soft">
        <div className="size-1.5 rounded-full bg-fresh" />
        <div className="size-1.5 rounded-full bg-slate-soft/40" />
        <div className="size-1.5 rounded-full bg-slate-soft/40" />
        <span className="ml-3 font-mono uppercase tracking-[0.14em] text-[7px] text-bone/60">
          BatchMyPhotos
        </span>
        <span className="ml-auto font-mono text-[8px] text-fresh tnum flex items-center gap-1">
          <span className="size-1 rounded-full bg-fresh breathe" />
          1.2s
        </span>
      </div>

      {/* main: source → folders */}
      <div className="flex-1 p-3 grid grid-cols-[1fr_auto_1fr] gap-2 md:gap-3 items-stretch">
        {/* source */}
        <div className="flex flex-col">
          <p className="font-mono uppercase tracking-[0.14em] text-[7px] text-slate mb-1">
            Source
          </p>
          <p
            className="font-display text-lg md:text-xl font-semibold tracking-tight tnum leading-none mb-2"
            style={{ animation: "count-up 0.8s 0.3s both" }}
          >
            10,000
          </p>
          <div className="flex-1 grid grid-cols-5 gap-[3px] content-start">
            {Array.from({ length: 25 }).map((_, i) => {
              const colors = [
                "var(--ink-soft)",
                "var(--slate)",
                "var(--slate-soft)",
              ];
              return (
                <div
                  key={i}
                  className="aspect-square rounded-[2px]"
                  style={{
                    backgroundColor: colors[i % colors.length],
                    animation: `fade-in 0.4s ${0.02 * i}s both`,
                    opacity: 0,
                  }}
                />
              );
            })}
          </div>
        </div>

        {/* arrow */}
        <div className="flex items-center justify-center">
          <svg
            className="size-5 md:size-6 text-fresh"
            viewBox="0 0 24 24"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M4 12 L20 12 M14 6 L20 12 L14 18"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>

        {/* sorted folders */}
        <div className="flex flex-col">
          <p className="font-mono uppercase tracking-[0.14em] text-[7px] text-slate mb-1">
            Sorted
          </p>
          <p
            className="font-display text-lg md:text-xl font-semibold tracking-tight tnum leading-none mb-2"
            style={{ animation: "count-up 0.8s 0.6s both" }}
          >
            20 folders
          </p>
          <div className="flex-1 grid grid-cols-3 gap-1 content-start">
            {Array.from({ length: 9 }).map((_, i) => (
              <div
                key={i}
                className="aspect-square rounded-[3px] border border-line bg-bone-deep flex flex-col items-center justify-center"
                style={{
                  animation: `fade-up 0.4s ${0.05 * i + 0.7}s both`,
                  opacity: 0,
                }}
              >
                <svg
                  className="size-2.5 md:size-3 text-ink"
                  viewBox="0 0 16 16"
                  fill="none"
                  aria-hidden="true"
                >
                  <path
                    d="M2 5 L6 5 L7.5 3.5 L14 3.5 L14 13 L2 13 Z"
                    fill="currentColor"
                    opacity="0.18"
                  />
                  <path
                    d="M2 5 L6 5 L7.5 3.5 L14 3.5 L14 13 L2 13 Z"
                    stroke="currentColor"
                    strokeWidth="1.2"
                    strokeLinejoin="round"
                  />
                </svg>
                <span className="font-mono text-[6px] text-slate tnum tracking-wider mt-0.5">
                  500
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* footer: manual comparison */}
      <div
        className="px-3 py-1.5 bg-bone-deep border-t border-line flex items-center justify-between"
        style={{ animation: "fade-up 0.5s 1.2s both" }}
      >
        <span className="font-mono uppercase tracking-[0.14em] text-[7px] text-slate">
          Manual sort
        </span>
        <span className="font-mono text-[9px] text-slate-soft tnum line-through">
          30–60 min
        </span>
      </div>
    </div>
  );
}

function FinalCta() {
  return (
    <section
      id="cta"
      className="snap-section min-h-[80vh] flex items-center px-6 md:px-10 py-16 md:py-24 bg-paper"
    >
      <div className="max-w-4xl mx-auto w-full text-center flex flex-col items-center">
        <Kicker>Cover your next event</Kicker>
        <h2 className="font-hero text-ink text-5xl md:text-7xl lg:text-8xl mt-5">
          Sell your
          <br />
          <span className="text-fresh">race photos.</span>
        </h2>
        <p className="font-sans text-lg md:text-xl text-ink-soft mt-6 max-w-md mx-auto">
          Free to list. We take a small cut only when a photo sells.
        </p>
        <div className="mt-10 flex flex-col md:flex-row items-center justify-center gap-5">
          <PrimaryCta href={REGISTER_AS_PHOTOGRAPHER}>Start selling</PrimaryCta>
          <Link
            href={ROUTES.RUNNERS}
            className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors"
          >
            I&apos;m a runner instead
          </Link>
        </div>
        <Link
          href="/?intro=1"
          className="mt-6 font-mono uppercase tracking-[0.14em] text-[13px] text-slate-soft hover:text-slate transition-colors"
        >
          Back to the intro
        </Link>
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
        href={REGISTER_AS_PHOTOGRAPHER}
        className="flex w-full items-center justify-center gap-2 bg-fresh active:bg-fresh-deep text-surface text-center py-3.5 rounded-full font-display font-bold text-[15px]"
      >
        Start selling
        <Arrow />
      </Link>
    </div>
  );
}
