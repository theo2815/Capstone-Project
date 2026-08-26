"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useCanUpload, type UploadGateState } from "@/hooks/use-can-upload";
import { usePhotographerEvents } from "@/hooks/use-photographer-data";
import { ROUTES } from "@/lib/constants";
import { SECTION_GUIDE } from "@/lib/section-guide";

const SECTION_GUIDE_STORAGE_KEY = "qp:dashboard:section-guide-collapsed";

// Setup journey shown on /dashboard while a photographer hasn't reached the
// "verified + first upload" milestone. Replaces the data-glance overview with
// a 3-step linear path: settings → submit for review → first upload. A
// preview card at the bottom shows what the dashboard becomes once they're
// live, so first-timers can see the destination.

type StepStatus = "active" | "done" | "locked";

interface SetupStep {
  number: string;
  title: string;
  body: string;
  status: StepStatus;
  hint?: string;
  cta?: { label: string; href: string };
}

const STEP_COUNT = 3;

export function SetupJourney() {
  const gate = useCanUpload();
  const events = usePhotographerEvents();
  // "Uploaded so far" = sum of photoCount across every event_photographer
  // row for this photographer. event_photographer.photo_count is updated by
  // the BE upsert on every successful upload (PhotoUploadService).
  const uploadedCount = (events ?? []).reduce(
    (sum, e) => sum + e.photoCount,
    0,
  );
  const steps = buildSteps(gate, uploadedCount);
  const activeIndex = steps.findIndex((s) => s.status === "active");
  const currentStep = activeIndex >= 0 ? activeIndex + 1 : steps.length;

  return (
    <section className="space-y-12 md:space-y-16">
      <SetupHero current={currentStep} total={STEP_COUNT} />

      <ol
        className="border-y border-line divide-y divide-line"
        aria-label="Setup steps"
      >
        {steps.map((step) => (
          <li key={step.number}>
            <SetupStepRow step={step} />
          </li>
        ))}
      </ol>

      <SectionsAtAGlance />

      <ComingUpPreview />
    </section>
  );
}

function buildSteps(
  gate: UploadGateState,
  uploaded: number,
): readonly SetupStep[] {
  return [
    {
      number: "01",
      title: "Complete your photographer settings",
      body:
        "Cover banner, brand, watermark, public URL, region, social, and payout — runners need these to know whose photos they're seeing.",
      status: gate.kind === "incomplete" ? "active" : "done",
      hint:
        gate.kind === "incomplete"
          ? `We still need your ${gate.missingLabel}.`
          : undefined,
      cta:
        gate.kind === "incomplete"
          ? { label: "Open settings", href: ROUTES.DASHBOARD_SETTINGS }
          : undefined,
    },
    {
      number: "02",
      title: "Submit for verification",
      body:
        "We'll look over your settings and approve within 1–2 business days. Uploads unlock as soon as we're done.",
      status:
        gate.kind === "incomplete"
          ? "locked"
          : gate.kind === "pending"
            ? "active"
            : "done",
      hint:
        gate.kind === "pending"
          ? "Your settings look good — submit them for review when you're ready."
          : undefined,
      cta:
        gate.kind === "pending"
          ? { label: "Open settings", href: ROUTES.DASHBOARD_SETTINGS }
          : undefined,
    },
    {
      number: "03",
      title: "Upload your first frame",
      body:
        "Pick any active event from Cebu's calendar. Photos auto-publish, so runners can find them by selfie or bib in seconds.",
      status:
        gate.kind !== "ok" ? "locked" : uploaded === 0 ? "active" : "done",
      hint:
        gate.kind === "ok" && uploaded === 0
          ? "Verified — open the catalog and pick your first event."
          : undefined,
      cta:
        gate.kind === "ok" && uploaded === 0
          ? { label: "Open upload", href: ROUTES.DASHBOARD_UPLOAD }
          : undefined,
    },
  ];
}

function SetupHero({ current, total }: { current: number; total: number }) {
  return (
    <div>
      <p className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate tnum">
        Welcome · Step {current} of {total}
      </p>
      <h1 className="font-display text-3xl md:text-5xl font-medium tracking-tight text-ink mt-4 leading-[1.05]">
        Let&apos;s get you ready for
        <br className="hidden md:block" /> your first event.
      </h1>
      <p className="font-sans text-base md:text-lg text-ink-soft mt-4 max-w-xl">
        Three short steps before runners can find your photos. We&apos;ll keep
        this page simple until you&apos;re live.
      </p>
      <ProgressDots current={current} total={total} />
    </div>
  );
}

function ProgressDots({ current, total }: { current: number; total: number }) {
  const dots = Array.from({ length: total }, (_, i) => i + 1);
  return (
    <div
      className="mt-8 flex items-center gap-2.5"
      role="progressbar"
      aria-valuenow={current}
      aria-valuemin={1}
      aria-valuemax={total}
      aria-label={`Setup progress, step ${current} of ${total}`}
    >
      {dots.map((n, i) => {
        const isActive = n === current;
        const isDone = n < current;
        const isLast = i === dots.length - 1;
        return (
          <div key={n} className="flex items-center gap-2.5">
            <span
              aria-hidden="true"
              className={
                isActive
                  ? "size-2.5 rounded-full bg-fresh breathe"
                  : isDone
                    ? "size-2.5 rounded-full bg-ink"
                    : "size-2.5 rounded-full border border-line bg-bone"
              }
            />
            {!isLast && (
              <span
                aria-hidden="true"
                className={
                  isDone ? "h-px w-8 bg-ink" : "h-px w-8 bg-line"
                }
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

function SetupStepRow({ step }: { step: SetupStep }) {
  const { number, title, body, status, hint, cta } = step;

  const numberClass =
    status === "active"
      ? "text-ink"
      : status === "done"
        ? "text-slate"
        : "text-slate-soft";
  const titleClass =
    status === "active"
      ? "text-ink"
      : status === "done"
        ? "text-slate"
        : "text-slate-soft";
  const bodyClass =
    status === "locked" ? "text-slate-soft" : "text-ink-soft";
  const hintClass =
    status === "active" ? "text-ink" : "text-slate";

  return (
    <div
      data-status={status}
      className={
        "py-7 md:py-9 px-4 md:px-6 transition-colors " +
        (status === "active" ? "bg-bone-deep/30" : "")
      }
    >
      <div className="flex flex-col md:flex-row md:items-start md:gap-10">
        <div className="flex items-center justify-between md:flex-col md:items-start md:justify-start md:w-28 md:shrink-0 md:gap-3">
          <span
            className={
              "font-mono uppercase tracking-[0.14em] text-[10px] tnum " +
              numberClass
            }
          >
            {number}
          </span>
          <StatusChip status={status} />
        </div>

        <div className="flex-1 mt-4 md:mt-0">
          <h2
            className={
              "font-display text-xl md:text-2xl font-medium tracking-tight " +
              titleClass
            }
          >
            {title}
          </h2>
          <p
            className={
              "font-sans text-sm md:text-base mt-2 max-w-xl " + bodyClass
            }
          >
            {body}
          </p>
          {hint && (
            <p
              className={
                "font-mono uppercase tracking-[0.14em] text-[10px] mt-4 tnum " +
                hintClass
              }
            >
              {hint}
            </p>
          )}
          {cta && (
            <Link
              href={cta.href}
              className="mt-5 inline-flex items-center gap-2 font-display text-base font-bold bg-fresh hover:bg-fresh-deep text-surface py-3 px-6 rounded-full transition-colors"
            >
              {cta.label}
              <span aria-hidden="true">→</span>
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}

function StatusChip({ status }: { status: StepStatus }) {
  if (status === "active") {
    return (
      <span className="font-mono uppercase tracking-[0.14em] text-[10px] text-ink inline-flex items-center gap-1.5">
        <span
          aria-hidden="true"
          className="size-1.5 rounded-full bg-fresh breathe"
        />
        Next up
      </span>
    );
  }
  if (status === "done") {
    return (
      <span className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate inline-flex items-center gap-1.5">
        <span aria-hidden="true">✓</span>
        Done
      </span>
    );
  }
  return (
    <span className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate-soft inline-flex items-center gap-1.5">
      <span
        aria-hidden="true"
        className="size-1.5 rounded-full border border-line bg-bone"
      />
      Locked
    </span>
  );
}

// SECTION_GUIDE moved to @/lib/section-guide so DashboardRail's per-item
// tip toggles can share the same copy. Edit there, not here.

function SectionsAtAGlance() {
  // Default to expanded for the SSR/first-paint pass; pull the stored
  // preference on mount so a returning first-timer who hid it stays hid.
  // SetupJourney itself unmounts once verified + uploaded, so the
  // collapsed state never leaks into the data-mode dashboard.
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    try {
      if (window.localStorage.getItem(SECTION_GUIDE_STORAGE_KEY) === "1") {
        setCollapsed(true);
      }
    } catch {
      // localStorage unavailable (private mode, SSR) — keep the default.
    }
  }, []);

  function toggle() {
    setCollapsed((prev) => {
      const next = !prev;
      try {
        if (next) {
          window.localStorage.setItem(SECTION_GUIDE_STORAGE_KEY, "1");
        } else {
          window.localStorage.removeItem(SECTION_GUIDE_STORAGE_KEY);
        }
      } catch {
        // best-effort persistence; UI state still flips.
      }
      return next;
    });
  }

  const bodyId = "sections-at-a-glance-body";

  return (
    <section aria-labelledby="sections-at-a-glance-title">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <p className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate tnum">
            Your sidebar · Six sections
          </p>
          <h2
            id="sections-at-a-glance-title"
            className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink mt-3"
          >
            What each tab does.
          </h2>
        </div>
        <button
          type="button"
          onClick={toggle}
          aria-expanded={!collapsed}
          aria-controls={bodyId}
          className="shrink-0 mt-1 font-mono uppercase tracking-[0.14em] text-[10px] text-slate hover:text-ink transition-colors inline-flex items-center gap-1.5 rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
        >
          <span>{collapsed ? "Show tip" : "Hide tip"}</span>
          <span aria-hidden="true">{collapsed ? "+" : "−"}</span>
        </button>
      </div>
      {!collapsed && (
        <div id={bodyId}>
          <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
            A quick map of the sidebar so you know where to land once
            you&apos;re live. This panel hides the moment you&apos;re verified
            and uploading.
          </p>
          <ul className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
            {SECTION_GUIDE.map((s) => (
              <li
                key={s.number}
                className="border border-line rounded-2xl p-6 md:p-7 bg-bone"
              >
                <p className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate-soft tnum">
                  {s.number}
                </p>
                <h3 className="font-display text-lg md:text-xl font-medium tracking-tight text-ink mt-3">
                  {s.label}
                </h3>
                <p className="font-sans text-sm text-ink-soft mt-2">{s.body}</p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function ComingUpPreview() {
  return (
    <div className="border border-dashed border-line rounded-2xl p-8 md:p-10">
      <p className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate tnum">
        Coming up · Once you&apos;re live
      </p>
      <h2 className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink mt-3">
        This page becomes your
        <br className="hidden md:block" /> earnings dashboard.
      </h2>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Your first sale shows up here with a 12-week sparkline, weekly payouts,
        active-event pipeline, and upcoming coverage — every number you&apos;ll
        check before each race day.
      </p>
      <ul className="mt-6 grid grid-cols-3 gap-4 md:gap-6">
        <PreviewStat label="Lifetime kept" placeholder="₱—" />
        <PreviewStat label="Photos live" placeholder="—" />
        <PreviewStat label="Last sale" placeholder="—" />
      </ul>
    </div>
  );
}

function PreviewStat({
  label,
  placeholder,
}: {
  label: string;
  placeholder: string;
}) {
  return (
    <li>
      <p className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate-soft tnum">
        {label}
      </p>
      <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-slate-soft tnum mt-2">
        {placeholder}
      </p>
    </li>
  );
}
