import type { ReactNode } from "react";
import { SiteHeader } from "@/components/layout/site-header";
import { AdminRail } from "@/components/admin/admin-rail";
import { AdminMobileStrip } from "@/components/admin/admin-mobile-strip";
import { AdminShellKeyboard } from "@/components/admin/admin-shell-keyboard";

// Persistent shell for every /admin/* page. Diverges from <DashboardShell>
// on layout width: admin goes full-bleed (no max-w-7xl) because queue +
// table + photographer-list surfaces benefit from the extra real estate.
// The header goes full-bleed too, so the logo lines up with the rail on wide
// screens (a centered brand bar sat ~90px right of the rail at 1440px).
export function AdminShell({ children }: { children: ReactNode }) {
  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col scroll-smooth">
      <SiteHeader fullWidth />
      <div className="md:hidden sticky top-[var(--site-header-h)] z-20">
        <AdminMobileStrip />
      </div>
      <div className="flex-1 w-full px-6 md:px-10 flex flex-col">
        <div className="md:grid md:grid-cols-[15rem_1fr] md:gap-12 lg:gap-20 flex-1">
          <AdminRail />
          {/* id="main" is the skip link's target; the scroll margin keeps the sticky header and strip off the title */}
          <div id="main" className="stagger-children min-w-0 scroll-mt-40 md:scroll-mt-24 pt-6 md:pt-10 pb-8 md:pb-20 md:border-l md:border-line md:-ml-6 lg:-ml-10 md:pl-6 lg:pl-10">
            {children}
          </div>
        </div>
      </div>
      <AdminShellKeyboard />
    </main>
  );
}
