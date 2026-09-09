"use client";

import Link from "next/link";
import type { UploadGateState } from "@/hooks/use-can-upload";
import { BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import { Kicker } from "@/components/ui/kicker";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";

// Shared friendly notice for a photographer who can't act yet. Used by the
// dashboard VerificationBanner (top strip) and the RequirePhotographerReady
// railguard (blocked-feature panel), so the "you're not ready" experience is
// authored once. Reads the backend-driven gate — never rendered for "ok".
export function PhotographerNotReadyNotice({
  gate,
}: {
  gate: Exclude<UploadGateState, { kind: "ok" }>;
}) {
  if (gate.kind === "suspended") {
    return (
      <div className="border border-ink rounded-2xl px-5 py-5 bg-bone-deep/60">
        <Kicker as="p">Account suspended</Kicker>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2">
          Uploads are paused while we review your account.
        </p>
        {gate.reason && (
          <p className="font-sans text-sm text-ink-soft mt-2 max-w-md">
            Reason: {gate.reason}
          </p>
        )}
        <p className="font-sans text-sm text-slate mt-2 max-w-md">
          Reach out to support to appeal — your public profile and uploads stay
          paused until the suspension is lifted.
        </p>
        <a
          href="mailto:support@quickpitik.com"
          className="mt-5 inline-flex w-fit items-center gap-2 font-sans text-base font-medium border border-ink text-ink hover:bg-ink hover:text-surface py-3 px-6 rounded-full transition-colors"
        >
          Contact support
          <span aria-hidden="true">→</span>
        </a>
      </div>
    );
  }

  if (gate.kind === "pending") {
    return (
      <div className="border border-line rounded-2xl px-5 py-5 bg-bone-deep/40">
        <Kicker as="p">Awaiting review</Kicker>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2">
          We&apos;re looking over your settings.
        </p>
        <p className="font-sans text-sm text-slate mt-2 max-w-md">
          Reviews take 1–2 business days. You&apos;ll be able to use the
          dashboard as soon as we&apos;re done — we&apos;ll let you know.
        </p>
      </div>
    );
  }

  // incomplete
  return (
    <div className="border border-line rounded-2xl px-5 py-5 bg-bone-deep/40">
      <Kicker as="p">Finish your settings</Kicker>
      <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2">
        Your photographer profile isn&apos;t ready yet.
      </p>
      <p className="font-sans text-sm text-slate mt-2 max-w-md">
        Complete your required settings before using the dashboard.
        {gate.missing.length > 0 && " Still needed:"}
      </p>
      {gate.missing.length > 0 && (
        <ul className="mt-3 space-y-1.5">
          {gate.missing.map((item) => (
            <li
              key={item}
              className="font-sans text-sm text-ink flex items-center gap-2.5"
            >
              <span
                aria-hidden="true"
                className="size-1.5 rounded-full bg-slate shrink-0"
              />
              <span className="capitalize">{item}</span>
            </li>
          ))}
        </ul>
      )}
      <Link
        href={ROUTES.DASHBOARD_SETTINGS}
        className={cn(BTN_PRIMARY, BTN_SIZE.md, "mt-5 inline-flex w-fit items-center gap-2")}
      >
        Open settings
        <span aria-hidden="true">→</span>
      </Link>
    </div>
  );
}
