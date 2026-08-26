"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";

// Mobile-only nudge for photographers. The dashboard flow (bulk upload,
// mosaic review, BatchMyPhotos pairing) is desktop-shaped, so /dashboard/*
// on a phone is partial-by-design. One quiet dismissable line — slate-soft
// dot, NOT fresh, to preserve the one-fresh-per-viewport rule on routes
// where <DashboardActionGrid> already owns the fresh accent.
//
// Renders as a flush horizontal bar (no rounded card) so it can sit pinned
// under <SiteHeader> alongside <DashboardMobileStrip>. The shared sticky
// wrapper in <DashboardShell> handles `md:hidden` and `sticky
// top-[var(--site-header-h)]`
// — this component is responsible only for its own bar styling and dismiss
// behavior. Returning null on dismiss lets the wrapper shrink so the strip
// pins flush under the header.
//
// Dismissal is **per-page-visit only** (no localStorage). Closing the nudge
// hides it only on the current route; navigating to any other /dashboard/*
// page (or returning to a previously-dismissed one) re-shows it. The
// pathname-keyed effect is what resets dismissal — without it, a single
// dismiss would silence the nudge session-wide because <DashboardShell>
// lives in the layout and the component does NOT unmount on nav.
export function DesktopNudge() {
  const pathname = usePathname();
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    setDismissed(false);
  }, [pathname]);

  if (dismissed) return null;

  return (
    <div className="bg-bone/90 backdrop-blur-md border-b border-line">
      <div className="max-w-7xl mx-auto w-full px-6 py-2.5 flex items-center gap-3">
        <span
          aria-hidden="true"
          className="size-1.5 rounded-full bg-slate-soft shrink-0"
        />
        <p className="flex-1 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate leading-relaxed">
          For the best experience, please open this on a desktop.
        </p>
        <button
          type="button"
          onClick={() => setDismissed(true)}
          aria-label="Dismiss desktop nudge"
          className="shrink-0 -m-1 p-1 text-slate hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
        >
          <span
            aria-hidden="true"
            className="font-mono text-[14px] leading-none"
          >
            ×
          </span>
        </button>
      </div>
    </div>
  );
}
