"use client";

import type { ReactNode } from "react";
import { SiteHeader } from "@/components/layout/site-header";
import { DashboardRail } from "@/components/dashboard/dashboard-rail";
import { DashboardMobileStrip } from "@/components/dashboard/dashboard-mobile-strip";
import { DesktopNudge } from "@/components/dashboard/desktop-nudge";
import { VerificationBanner } from "@/components/dashboard/verification-banner";
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
// Mobile-only sticky stack under <SiteHeader>: a single wrapper at
// `top-[3.75rem]` (matching the /events/[slug]?browse=1 "Find your photos"
// pattern) holds <DesktopNudge> on top + <DashboardMobileStrip> below. Both
// pin together as the user scrolls every /dashboard/* route. When the user
// dismisses the nudge, it returns null and the wrapper shrinks so the strip
// pins flush under the header — no gap. Putting both in one sticky parent
// avoids two siblings racing at the same `top` value (they'd overlap).
export function DashboardShell({ children }: { children: ReactNode }) {
  usePhotographerVerificationSync();

  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col scroll-smooth">
      <SiteHeader />
      <div className="md:hidden sticky top-[3.75rem] z-20">
        <DesktopNudge />
        <DashboardMobileStrip />
      </div>
      <div className="flex-1 max-w-7xl mx-auto w-full px-6 md:px-10 flex flex-col">
        <div className="md:grid md:grid-cols-[15rem_1fr] md:gap-12 lg:gap-20 flex-1">
          <DashboardRail />
          <div className="stagger-children min-w-0 pt-6 md:pt-10 pb-8 md:pb-20 md:border-l md:border-line md:-ml-6 lg:-ml-10 md:pl-6 lg:pl-10">
            <VerificationBanner />
            {children}
          </div>
        </div>
      </div>
    </main>
  );
}
