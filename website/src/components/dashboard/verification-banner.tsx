"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCanUpload } from "@/hooks/use-can-upload";
import { ROUTES } from "@/lib/constants";

interface VerificationBannerProps {
  /** When true, even an "ok" gate shows a thin "verified" confirmation strip. */
  showOnApproved?: boolean;
}

// Banner shown across photographer dashboard pages that depend on upload
// permission. Reflects the gate state from useCanUpload:
//   suspended  → "Account suspended" + admin's reason + Contact support
//   incomplete → "Finish your settings" + which field is missing + Settings link
//   pending    → "Awaiting review" + reassurance copy
//   ok         → optional thin "Verified" strip (off by default)
//
// Suppressed on /dashboard/settings: that page renders an inline
// VerificationStatusPanel at the bottom which covers the same state +
// actions ("Submit for review" / "Edit settings"). Two banners would
// just consume vertical space and confuse the eye.
export function VerificationBanner({
  showOnApproved = false,
}: VerificationBannerProps) {
  const gate = useCanUpload();
  const pathname = usePathname();

  if (pathname === ROUTES.DASHBOARD_SETTINGS) return null;
  if (gate.kind === "ok" && !showOnApproved) return null;

  if (gate.kind === "suspended") {
    return (
      <div className="border border-ink rounded-2xl px-5 py-5 bg-bone-deep/60 mb-8">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-ink">
          Account suspended
        </p>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2">
          Uploads are paused while we review your account.
        </p>
        {gate.reason && (
          <p className="font-sans text-sm text-ink-soft mt-2 max-w-md">
            Reason: {gate.reason}
          </p>
        )}
        <p className="font-sans text-sm text-slate mt-2 max-w-md">
          Reach out to support to appeal — your public profile and uploads
          stay paused until the suspension is lifted.
        </p>
        <a
          href="mailto:support@quickpitik.com"
          className="mt-5 inline-flex items-center gap-2 font-sans text-base font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-3 px-6 rounded-full transition-colors"
        >
          Contact support
          <span aria-hidden="true">→</span>
        </a>
      </div>
    );
  }

  if (gate.kind === "ok") {
    return (
      <div className="border border-line rounded-2xl px-5 py-4 bg-bone-deep/30 flex items-center gap-3 mb-8">
        <span
          aria-hidden="true"
          className="size-1.5 rounded-full bg-fresh shrink-0"
        />
        <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate">
          Verified · uploads enabled
        </p>
      </div>
    );
  }

  if (gate.kind === "pending") {
    return (
      <div className="border border-line rounded-2xl px-5 py-4 bg-bone-deep/40 mb-8">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
          Awaiting review
        </p>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2">
          We&apos;re looking over your settings.
        </p>
        <p className="font-sans text-sm text-slate mt-2 max-w-md">
          Reviews take 1–2 business days. You&apos;ll be able to upload as soon
          as we&apos;re done. We&apos;ll let you know.
        </p>
      </div>
    );
  }

  // incomplete
  return (
    <div className="border border-line rounded-2xl px-5 py-5 bg-bone-deep/40 mb-8">
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
        Finish your settings
      </p>
      <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2">
        Add your {gate.missingLabel} before you can upload.
      </p>
      <p className="font-sans text-sm text-slate mt-2 max-w-md">
        Cover banner, brand, watermark, and public URL are all required so
        runners know whose photos they&apos;re looking at.
      </p>
      <Link
        href={ROUTES.DASHBOARD_SETTINGS}
        className="mt-5 inline-flex items-center gap-2 font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-6 rounded-full transition-colors"
      >
        Open settings
        <span aria-hidden="true">→</span>
      </Link>
    </div>
  );
}
