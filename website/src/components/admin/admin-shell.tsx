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
// Tablets (below lg) get the phone strip: beside the rail, a 768 px screen left about 400 px for the page.
export function AdminShell({ children }: { children: ReactNode }) {
  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col scroll-smooth">
      <SiteHeader fullWidth />
      <div className="lg:hidden sticky top-[var(--site-header-h)] z-20">
        <AdminMobileStrip />
      </div>
      <div className="flex-1 w-full px-6 lg:px-10 flex flex-col">
        <div className="lg:grid lg:grid-cols-[15rem_1fr] lg:gap-20 flex-1">
          <AdminRail />
          {/* id="main" is the skip link's target; the scroll margin keeps the sticky header and strip off the title */}
          <div id="main" className="stagger-children min-w-0 scroll-mt-40 lg:scroll-mt-24 pt-6 lg:pt-10 pb-8 lg:pb-20 lg:border-l lg:border-line lg:-ml-10 lg:pl-10">
            {children}
          </div>
        </div>
      </div>
      <AdminShellKeyboard />
    </main>
  );
}
