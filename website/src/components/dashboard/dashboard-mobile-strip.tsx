"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCanUpload } from "@/hooks/use-can-upload";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";

interface DashboardRoute {
  readonly href: string;
  readonly label: string;
  readonly nestedPrefix?: string;
}

const DASHBOARD_ROUTES: ReadonlyArray<DashboardRoute> = [
  { href: ROUTES.DASHBOARD, label: "Overview" },
  { href: ROUTES.DASHBOARD_UPLOAD, label: "Upload photos" },
  {
    href: ROUTES.DASHBOARD_EVENTS,
    label: "Events",
    nestedPrefix: ROUTES.DASHBOARD_EVENTS,
  },
  { href: ROUTES.DASHBOARD_EARNINGS, label: "Earnings" },
  { href: ROUTES.DASHBOARD_BILLING, label: "Billing" },
  { href: ROUTES.DASHBOARD_SETTINGS, label: "Settings" },
];

// Mobile-only chip strip for /dashboard/*. Visual treatment mirrors the
// `/events/[slug]?browse=1` "Find your photos" bar — `bg-bone/90
// backdrop-blur-md border-b border-line` — but the **sticky positioning
// lives on the parent wrapper in <DashboardShell>** so this component
// stacks together with <DesktopNudge> in a single sticky block under the
// SiteHeader (z-30). Doing the sticky here would race with the nudge's
// own sticky — they'd both pin at `top-[3.75rem]` and overlap. The shared
// wrapper also gives us "nudge dismissed → strip pins flush" for free
// because the wrapper's height collapses when the nudge returns null.
//
// Renders nothing ≥ md (the wrapper carries `md:hidden`); desktop uses the
// vertical rail nav.
export function DashboardMobileStrip() {
  const pathname = usePathname();
  const gate = useCanUpload();

  return (
    <div className="bg-bone/90 backdrop-blur-md border-b border-line">
      <nav aria-label="Dashboard sections" className="max-w-7xl mx-auto">
        <ul className="flex gap-2 overflow-x-auto px-6 py-3 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {DASHBOARD_ROUTES.map((route) => {
            const isActive = isRouteActive(pathname, route);
            const showSettingsBadge =
              route.href === ROUTES.DASHBOARD_SETTINGS && gate.kind !== "ok";
            return (
              <li key={route.href} className="shrink-0">
                <Link
                  href={route.href}
                  aria-current={isActive ? "page" : undefined}
                  className={cn(
                    "inline-flex items-center gap-2 px-3 py-1.5 rounded-full border font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] transition-colors",
                    isActive
                      ? "border-ink text-ink bg-ink/5"
                      : "border-line text-slate hover:text-ink",
                  )}
                >
                  <span
                    aria-hidden="true"
                    className={cn(
                      "size-1.5 rounded-full",
                      isActive ? "bg-fresh" : "bg-line",
                    )}
                  />
                  {route.label}
                  {showSettingsBadge && (
                    <span
                      aria-label="Settings need attention"
                      className="size-1.5 rounded-full bg-fresh breathe"
                    />
                  )}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </div>
  );
}

function isRouteActive(
  pathname: string | null,
  route: DashboardRoute,
): boolean {
  if (!pathname) return false;
  if (route.href === ROUTES.DASHBOARD) {
    return pathname === route.href;
  }
  if (route.nestedPrefix) {
    return (
      pathname === route.href || pathname.startsWith(`${route.nestedPrefix}/`)
    );
  }
  return pathname === route.href || pathname.startsWith(`${route.href}/`);
}
