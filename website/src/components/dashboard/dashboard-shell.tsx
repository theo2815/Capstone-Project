"use client";

import type { ReactNode } from "react";
import { usePathname } from "next/navigation";
import { ROUTES } from "@/lib/constants";
import { SiteHeader } from "@/components/layout/site-header";
import { DashboardRail, DASHBOARD_ROUTES } from "@/components/dashboard/dashboard-rail";
import { DashboardBottomNav } from "@/components/dashboard/dashboard-bottom-nav";
import { VerificationBanner } from "@/components/dashboard/verification-banner";
import { LegalLinks } from "@/components/legal/legal-links";
import { usePhotographerVerificationSync } from "@/lib/photographer-verification-sync";

// Persistent shell for every /dashboard/* page. SiteHeader, sticky rail with
// route nav, content column. The page below the layout renders children only
// — no per-page chrome. Mirrors /profile / /orders / /account container
// width and grid for visual consistency. Footer removed 2026-05-06 PM so the
// rail stays visible at the bottom of the viewport instead of getting pushed
// up by the footer band.
//
// Verification sync + banner are mounted once here so suspended state shows
// up across overview / upload / events / earnings / billing / settings
// without each page re-mounting them. Approved photographers see nothing
// (banner returns null); suspended / incomplete / pending see the matching
// banner. The sync hook polls /me/photographer/verification on mount, focus,
// and every 30s while pending — single mount keeps duplicate polling away.
//
// Photographer section nav is split by breakpoint: the vertical <DashboardRail>
// on desktop (hidden below md), and the floating <DashboardBottomNav> pill on
// mobile (hidden from md up).
export function DashboardShell({ children }: { children: ReactNode }) {
  usePhotographerVerificationSync();
  // Section pages opened straight into a slab or a row of buttons, with no
  // title of their own; the Overview has its welcome heading. Exact paths only:
  // nested pages (an event, a new event) carry their own headers.
  const pathname = usePathname();
  const section = DASHBOARD_ROUTES.find((r) => r.href === pathname && r.href !== ROUTES.DASHBOARD);
  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col scroll-smooth">
      <SiteHeader />
      <div className="flex-1 max-w-7xl mx-auto w-full px-6 md:px-10 flex flex-col">
        <div className="md:grid md:grid-cols-[15rem_1fr] md:gap-12 lg:gap-20 flex-1">
          <DashboardRail />
          <div className="stagger-children min-w-0 pt-6 md:pt-10 pb-28 md:pb-20 md:border-l md:border-line md:-ml-6 lg:-ml-10 md:pl-6 lg:pl-10">
            <VerificationBanner />
            {section && (
              <header className="pb-4 md:pb-6">
                <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
                  Dashboard
                </p>
                <h2 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight text-ink leading-[1.05] mt-4">
                  {section.label}.
                </h2>
              </header>
            )}
            {children}
            {/* The rail (and its legal links) is desktop-only. */}
            <div className="md:hidden mt-12">
              <LegalLinks />
            </div>
          </div>
        </div>
      </div>
      <DashboardBottomNav />
    </main>
  );
}
