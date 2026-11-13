"use client";

import Link from "next/link";
import type { UploadGateState } from "@/hooks/use-can-upload";
import { BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import { Kicker } from "@/components/ui/kicker";
import { ROUTES, SUPPORT_EMAIL } from "@/lib/constants";
import { cn } from "@/lib/utils";

const SELLER_DETAILS_HREF = `${ROUTES.DASHBOARD_SETTINGS}#seller-details`;

// The identity gate (AA-I15). Approved states never reach it (they pass both
// upload sets); the map falls back to the manual-review copy for them.
type IdentityCopy = { kicker: string; headline: string; line: string; cta: string };
const IDENTITY_COPY: Record<"missing" | "manual_review" | "rejected", IdentityCopy> = {
  missing: {
    kicker: "Before you upload",
    headline: "Seller details needed",
    line: "Add your ID and take a live selfie before you upload.",
    cta: "Add your seller details",
  },
  manual_review: {
    kicker: "Identity check",
    headline: "An admin is checking your details",
    line: "Uploads open when they approve.",
    cta: "See your seller-details status",
  },
  rejected: {
    kicker: "Seller details",
    headline: "Your seller details were not accepted.",
    line: "New uploads are paused until you send them again.",
    cta: "Send them again",
  },
};

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
          href={`mailto:${SUPPORT_EMAIL}`}
          className="mt-5 inline-flex w-fit items-center gap-2 font-sans text-base font-medium border border-ink text-ink hover:bg-ink hover:text-surface py-3 px-6 rounded-full transition-colors"
        >
          Contact support
          <span aria-hidden="true">→</span>
        </a>
      </div>
    );
  }

  if (gate.kind === "pending") {
    // The profile goes live with the identity check, unless an admin holds it (R-10).
    return (
      <div className="border border-line rounded-2xl px-5 py-5 bg-bone-deep/40">
        <Kicker as="p">Awaiting identity check</Kicker>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2">
          {gate.awaitingAdmin
            ? "An admin is reviewing your profile."
            : "Your studio goes live when your identity check passes."}
        </p>
        <Link
          href={SELLER_DETAILS_HREF}
          className="font-sans text-sm text-ink underline decoration-line-strong underline-offset-4 hover:decoration-fresh mt-3 inline-block"
        >
          See your seller-details status
        </Link>
      </div>
    );
  }

  if (gate.kind === "identity") {
    const copy: IdentityCopy =
      IDENTITY_COPY[gate.reviewState as keyof typeof IDENTITY_COPY] ?? IDENTITY_COPY.manual_review;
    return (
      <div className="border border-line rounded-2xl px-5 py-5 bg-bone-deep/40">
        <Kicker as="p">{copy.kicker}</Kicker>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2">
          {copy.headline}
        </p>
        <p className="font-sans text-sm text-slate mt-2 max-w-md">{copy.line}</p>
        <Link
          href={SELLER_DETAILS_HREF}
          className={cn(BTN_PRIMARY, BTN_SIZE.md, "mt-5 inline-flex w-fit items-center gap-2")}
        >
          {copy.cta}
          <span aria-hidden="true">→</span>
        </Link>
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
