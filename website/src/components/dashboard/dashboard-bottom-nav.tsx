"use client";

import type { FC } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCanUpload } from "@/hooks/use-can-upload";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";

// Mobile-only bottom navigation for /dashboard/* — the photographer's primary
// section switcher on a phone, replacing the old top chip strip. Mirrors the
// Kotlin mobile app's floating-pill bottom bar (5 tabs, Home is the start tab):
// unselected tabs show an icon only (slate); the selected tab expands into an
// ink pill with icon + label. Only one label is ever on screen, so labels fit
// 375px without the mono-uppercase width-inflation wrap trap (vault
// ui-pitfalls 2026-05-06) — hence sentence-case sans labels, which also match
// the mobile app's labels.
//
// Billing has no tab here by design — the mobile app folds billing into the
// money flow, so it is reached from a link on the Earnings page (5-tab parity,
// user's call 2026-08-26).
//
// Renders nothing >= md (md:hidden); desktop uses the vertical <DashboardRail>.
// The selection fill is ink, not fresh, to preserve one-fresh-per-viewport; the
// Settings badge is a small fresh dot (exempt from that rule).

interface BottomTab {
  readonly href: string;
  readonly label: string;
  /** Sub-routes that should also light this tab active. */
  readonly nestedPrefix?: string;
  readonly icon: FC;
}

const TABS: ReadonlyArray<BottomTab> = [
  { href: ROUTES.DASHBOARD, label: "Home", icon: HomeIcon },
  { href: ROUTES.DASHBOARD_UPLOAD, label: "Upload", icon: UploadIcon },
  {
    href: ROUTES.DASHBOARD_EVENTS,
    label: "Events",
    nestedPrefix: ROUTES.DASHBOARD_EVENTS,
    icon: EventsIcon,
  },
  { href: ROUTES.DASHBOARD_EARNINGS, label: "Earnings", icon: EarningsIcon },
  { href: ROUTES.DASHBOARD_SETTINGS, label: "Settings", icon: SettingsIcon },
];

export function DashboardBottomNav() {
  const pathname = usePathname();
  const gate = useCanUpload();

  return (
    <nav
      aria-label="Dashboard sections"
      className="md:hidden fixed inset-x-0 bottom-5 z-30 flex justify-center px-6 pb-[env(safe-area-inset-bottom)]"
    >
      <ul className="flex items-center gap-1 rounded-full border border-line bg-bone/95 backdrop-blur-md px-2 py-2 shadow-[0_18px_50px_-12px_rgba(17,17,17,0.18)]">
        {TABS.map((tab) => {
          const isActive = isTabActive(pathname, tab);
          const showBadge =
            tab.href === ROUTES.DASHBOARD_SETTINGS && gate.kind !== "ok";
          const Icon = tab.icon;
          return (
            <li key={tab.href}>
              <Link
                href={tab.href}
                aria-current={isActive ? "page" : undefined}
                aria-label={tab.label}
                className={cn(
                  "relative flex h-11 items-center justify-center rounded-full transition-colors",
                  isActive
                    ? "gap-2 bg-ink px-4 text-surface"
                    : "w-11 text-slate hover:text-ink",
                )}
              >
                <span className="relative grid place-items-center">
                  <Icon />
                  {showBadge && (
                    <span
                      aria-label="Settings need attention"
                      className="absolute -right-1 -top-1 size-2 rounded-full bg-fresh breathe"
                    />
                  )}
                </span>
                {isActive && (
                  <span className="font-sans text-[13px] font-semibold leading-none">
                    {tab.label}
                  </span>
                )}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

function isTabActive(pathname: string | null, tab: BottomTab): boolean {
  if (!pathname) return false;
  // Overview is active only on the exact /dashboard path; otherwise its tab
  // would never go off — every dashboard route starts with /dashboard.
  if (tab.href === ROUTES.DASHBOARD) return pathname === tab.href;
  if (tab.nestedPrefix) {
    return pathname === tab.href || pathname.startsWith(`${tab.nestedPrefix}/`);
  }
  return pathname === tab.href || pathname.startsWith(`${tab.href}/`);
}

// Inline SVG icons matching the app's line style (20px, 1.75 stroke,
// currentColor, round joins). Local to this surface — the only place that
// needs them.
const ICON_PROPS = {
  viewBox: "0 0 24 24",
  className: "size-5",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.75,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
} as const;

function HomeIcon() {
  return (
    <svg {...ICON_PROPS}>
      <path d="M4 11 L12 4 L20 11" />
      <path d="M6.5 9.5 V19.5 H17.5 V9.5" />
      <path d="M10 19.5 V14 H14 V19.5" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg {...ICON_PROPS}>
      <path d="M12 4 V14" />
      <path d="M8 8 L12 4 L16 8" />
      <path d="M5 15 V18 A1 1 0 0 0 6 19 H18 A1 1 0 0 0 19 18 V15" />
    </svg>
  );
}

function EventsIcon() {
  return (
    <svg {...ICON_PROPS}>
      <path d="M9 6 H20" />
      <path d="M9 12 H20" />
      <path d="M9 18 H20" />
      <path d="M4.5 6 H4.51" />
      <path d="M4.5 12 H4.51" />
      <path d="M4.5 18 H4.51" />
    </svg>
  );
}

function EarningsIcon() {
  return (
    <svg {...ICON_PROPS}>
      <path d="M4 20 H20" />
      <path d="M7.5 20 V15" />
      <path d="M12 20 V11" />
      <path d="M16.5 20 V6" />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg {...ICON_PROPS}>
      <path d="M4 7 H13" />
      <path d="M17 7 H20" />
      <circle cx="15" cy="7" r="2" />
      <path d="M4 17 H7" />
      <path d="M11 17 H20" />
      <circle cx="9" cy="17" r="2" />
    </svg>
  );
}
