"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { PhotographerNotReadyNotice } from "@/components/dashboard/photographer-not-ready-notice";
import { BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import { Kicker } from "@/components/ui/kicker";
import { useCanUpload } from "@/hooks/use-can-upload";
import { useSellerDetails } from "@/hooks/use-seller-details";
import { formatGateDate, identityNoticeDate } from "@/lib/api-seller-details";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";

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
  const noticeDate = identityNoticeDate(useSellerDetails().query.data);
  const pathname = usePathname();

  if (pathname === ROUTES.DASHBOARD_SETTINGS) return null;

  if (gate.kind === "ok") {
    // The identity-gate notice (AA-I22, AA-0.8 wording): uploads still work today, but not after the date.
    if (noticeDate) {
      const date = formatGateDate(noticeDate);
      return (
        <div className="mb-8 border border-line rounded-2xl px-5 py-5 bg-bone-deep/40">
          <Kicker as="p">From {date}</Kicker>
          <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2">
            A quick live selfie before you upload
          </p>
          <p className="font-sans text-sm text-slate mt-2 max-w-xl">
            From {date}, you need a passed identity check to upload new photos. It takes about a minute: open
            Settings → Seller details, type your ID number and take a short live selfie. We compare it with your ID
            photo. Rather not? Choose &ldquo;Have an admin check instead&rdquo;, and an admin will look at your
            details. Until {date}, uploads work as they do now.
          </p>
          <Link
            href={`${ROUTES.DASHBOARD_SETTINGS}#seller-details`}
            className={cn(BTN_PRIMARY, BTN_SIZE.md, "mt-5 inline-flex w-fit items-center gap-2")}
          >
            Do it now
            <span aria-hidden="true">→</span>
          </Link>
        </div>
      );
    }
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
