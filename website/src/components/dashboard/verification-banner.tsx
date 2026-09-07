"use client";

import { usePathname } from "next/navigation";
import { PhotographerNotReadyNotice } from "@/components/dashboard/photographer-not-ready-notice";
import { useCanUpload } from "@/hooks/use-can-upload";
import { ROUTES } from "@/lib/constants";

interface VerificationBannerProps {
  /** When true, even an "ok" gate shows a thin "verified" confirmation strip. */
  showOnApproved?: boolean;
}

// Banner shown across photographer dashboard pages that depend on being ready.
// Reflects the gate from useCanUpload (backend-driven); the incomplete /
// pending / suspended body is the shared PhotographerNotReadyNotice — the same
// notice the RequirePhotographerReady railguard renders, so the messaging is
// authored once and can't drift.
//
// Suppressed on /dashboard/settings: that page renders an inline
// VerificationStatusPanel covering the same state + actions ("Submit for
// review" / "Edit settings"). Two banners would just consume vertical space.
export function VerificationBanner({
  showOnApproved = false,
}: VerificationBannerProps) {
  const gate = useCanUpload();
  const pathname = usePathname();

  if (pathname === ROUTES.DASHBOARD_SETTINGS) return null;

  if (gate.kind === "ok") {
    if (!showOnApproved) return null;
    return (
      <div className="border border-line rounded-2xl px-5 py-4 bg-bone-deep/30 flex items-center gap-3 mb-8">
        <span
          aria-hidden="true"
          className="size-1.5 rounded-full bg-fresh shrink-0"
        />
        <p className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate">
          Verified · uploads enabled
        </p>
      </div>
    );
  }

  return (
    <div className="mb-8">
      <PhotographerNotReadyNotice gate={gate} />
    </div>
  );
}
