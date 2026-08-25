"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FocusEvent,
} from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { moreLinksForRole } from "@/components/profile-shell";
import { RailTipPortal } from "@/components/ui/rail-tip-portal";
import { useAuth } from "@/hooks/use-auth";
import { useCanUpload } from "@/hooks/use-can-upload";
import { ROUTES } from "@/lib/constants";
import { formatMemberSince } from "@/lib/format";
import {
  getSectionGuideByHref,
  type SectionGuideEntry,
} from "@/lib/section-guide";
import { cn } from "@/lib/utils";

const RAIL_TIPS_STORAGE_KEY = "qp:dashboard:rail-tips-disabled";

interface DashboardRoute {
  readonly href: string;
  readonly label: string;
  /** Sub-routes that should also light this entry as active. */
  readonly nestedPrefix?: string;
}

const DASHBOARD_ROUTES: ReadonlyArray<DashboardRoute> = [
  { href: ROUTES.DASHBOARD, label: "Overview" },
  {
    href: ROUTES.DASHBOARD_UPLOAD,
    label: "Upload photos",
  },
  {
    href: ROUTES.DASHBOARD_EVENTS,
    label: "Events",
    nestedPrefix: ROUTES.DASHBOARD_EVENTS,
  },
  { href: ROUTES.DASHBOARD_EARNINGS, label: "Earnings" },
  { href: ROUTES.DASHBOARD_BILLING, label: "Billing" },
  { href: ROUTES.DASHBOARD_SETTINGS, label: "Settings" },
];

// Persistent rail for /dashboard/* — same skeleton as IdentityRail but the
// jump-to anchors become route Links and active state comes from usePathname
// rather than an IntersectionObserver. Settings entry carries a small fresh
// dot when verification isn't approved (excluded from one-fresh rule).
//
// Each row carries a contextual tip surfaced via hover/focus on desktop and
// a touch fallback "?" button on coarse-pointer devices. The whole tip
// system can be silenced with the master "Hide tips" toggle at the top of
// the Sections list (preference persists in localStorage).
export function DashboardRail() {
  const router = useRouter();
  const pathname = usePathname();
  const { user, logout } = useAuth();
  const gate = useCanUpload();
  const [tipsDisabled, setTipsDisabled] = useState(false);

  useEffect(() => {
    try {
      if (window.localStorage.getItem(RAIL_TIPS_STORAGE_KEY) === "1") {
        setTipsDisabled(true);
      }
    } catch {
      // localStorage unavailable — keep default (tips on).
    }
  }, []);

  function toggleTipsDisabled() {
    setTipsDisabled((prev) => {
      const next = !prev;
      try {
        if (next) {
          window.localStorage.setItem(RAIL_TIPS_STORAGE_KEY, "1");
        } else {
          window.localStorage.removeItem(RAIL_TIPS_STORAGE_KEY);
        }
      } catch {
        // best-effort persistence; UI state still flips.
      }
      return next;
    });
  }

  if (!user) return null;

  const memberSince = formatMemberSince(user.createdAt);
  // Pass DASHBOARD as currentPath so /dashboard is filtered from MORE; we're
  // inside the dashboard tree on every /dashboard/* route.
  const moreLinks = moreLinksForRole(user.role, ROUTES.DASHBOARD);

  function handleSignOut() {
    logout();
    router.replace(ROUTES.HOME);
  }

  return (
    <aside className="md:sticky md:top-20 md:self-start pt-6 md:pt-10 pb-6 md:pb-8 md:max-h-[calc(100vh-5rem)] md:overflow-y-auto border-b md:border-0 border-line">
      <div className="flex items-start gap-5 md:block">
        <AvatarDisc name={user.name} size="md" />
        <div className="flex-1 min-w-0 md:mt-7">
          <p className="font-mono uppercase tracking-[0.3em] text-[12px] text-slate">
            Photographer · Cebu
            <span className="text-slate-soft"> · </span>
            <span className="tnum">Since {memberSince}</span>
          </p>
          <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
            Dashboard.
          </h1>
          <p className="font-sans text-sm text-slate mt-2 max-w-xs">
            Earnings, events, uploads, and brand.
          </p>
        </div>
      </div>

      <nav aria-label="Dashboard sections" className="hidden md:block mt-10">
        <div className="flex items-baseline justify-between gap-3">
          <p className="font-mono uppercase tracking-[0.3em] text-[12px] text-slate-soft">
            Sections
          </p>
          <button
            type="button"
            onClick={toggleTipsDisabled}
            aria-pressed={tipsDisabled}
            className="font-mono uppercase tracking-[0.25em] text-[12px] text-slate hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
          >
            {tipsDisabled ? "Show tips" : "Hide tips"}
          </button>
        </div>
        <ul className="mt-4 space-y-3">
          {DASHBOARD_ROUTES.map((route) => {
            const isActive = isRouteActive(pathname, route);
            const showSettingsBadge =
              route.href === ROUTES.DASHBOARD_SETTINGS && gate.kind !== "ok";
            const guide = getSectionGuideByHref(route.href);
            return (
              <RailRow
                key={route.href}
                route={route}
                isActive={isActive}
                showSettingsBadge={showSettingsBadge}
                guide={guide}
                tipsDisabled={tipsDisabled}
              />
            );
          })}
        </ul>
      </nav>

      <div className="mt-8 md:mt-10">
        <p className="font-mono uppercase tracking-[0.3em] text-[12px] text-slate-soft">
          More
        </p>
        <ul className="mt-4 space-y-3">
          {moreLinks.map((link) => (
            <li key={link.href}>
              <Link
                href={link.href}
                className="font-display text-base text-slate hover:text-ink transition-colors"
              >
                {link.label}
              </Link>
            </li>
          ))}
          <li>
            <button
              type="button"
              onClick={handleSignOut}
              className="font-display text-base text-slate hover:text-ink transition-colors"
            >
              Sign out
            </button>
          </li>
        </ul>
      </div>
    </aside>
  );
}

interface RailRowProps {
  route: DashboardRoute;
  isActive: boolean;
  showSettingsBadge: boolean;
  guide: SectionGuideEntry | undefined;
  tipsDisabled: boolean;
}

function RailRow({
  route,
  isActive,
  showSettingsBadge,
  guide,
  tipsDisabled,
}: RailRowProps) {
  const rowRef = useRef<HTMLLIElement>(null);
  // labelRef anchors the popover to the actual label text so the tip sits
  // immediately past the wordmark instead of jumping to the row's full
  // 15rem width and floating in the gutter.
  const labelRef = useRef<HTMLSpanElement>(null);
  const [hovered, setHovered] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Master toggle override: clear any pinned state the moment tips are
  // silenced so a tap-pinned tooltip on touch can't survive the switch.
  useEffect(() => {
    if (tipsDisabled) {
      setPinned(false);
      setHovered(false);
    }
  }, [tipsDisabled]);

  const handleMouseEnter = useCallback(() => {
    if (!tipsDisabled) setHovered(true);
  }, [tipsDisabled]);

  const handleMouseLeave = useCallback(() => {
    setHovered(false);
  }, []);

  const handleFocus = useCallback(() => {
    if (!tipsDisabled) setHovered(true);
  }, [tipsDisabled]);

  const handleBlur = useCallback((e: FocusEvent<HTMLLIElement>) => {
    // Only clear if focus genuinely left the row (e.g. tabbed away).
    if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
      setHovered(false);
    }
  }, []);

  const handleTouchToggle = useCallback(() => {
    if (tipsDisabled) return;
    setPinned((p) => !p);
  }, [tipsDisabled]);

  const showTip = !tipsDisabled && !!guide && (hovered || pinned);
  const tipId = `dashboard-rail-tip-${route.href.replace(/\//g, "-")}`;

  return (
    <li
      ref={rowRef}
      className="relative"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onFocus={handleFocus}
      onBlur={handleBlur}
    >
      <div className="flex items-center gap-2">
        <Link
          href={route.href}
          aria-current={isActive ? "page" : undefined}
          aria-describedby={showTip ? tipId : undefined}
          className={cn(
            "group flex flex-1 items-center gap-3 font-display text-base transition-colors min-w-0",
            isActive ? "text-ink" : "text-slate hover:text-ink",
          )}
        >
          <span
            aria-hidden="true"
            className={cn(
              "size-1.5 rounded-full transition-colors shrink-0",
              isActive ? "bg-fresh" : "bg-line group-hover:bg-slate",
            )}
          />
          <span ref={labelRef}>{route.label}</span>
          {showSettingsBadge && (
            <span
              aria-label="Settings need attention"
              className="ml-auto size-1.5 rounded-full bg-fresh breathe shrink-0"
            />
          )}
        </Link>
        {!tipsDisabled && guide && (
          <button
            type="button"
            onClick={handleTouchToggle}
            aria-expanded={pinned}
            aria-controls={tipId}
            aria-label={
              pinned
                ? `Hide tip for ${route.label}`
                : `Show tip for ${route.label}`
            }
            className="shrink-0 size-5 rounded-full border border-line text-slate hover:text-ink hover:border-ink transition-colors flex items-center justify-center font-mono text-[12px] leading-none focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone [@media(hover:hover)]:hidden"
          >
            <span aria-hidden="true">{pinned ? "−" : "?"}</span>
          </button>
        )}
      </div>
      {mounted && showTip && guide && labelRef.current && (
        <RailTipPortal anchor={labelRef.current} guide={guide} tipId={tipId} />
      )}
    </li>
  );
}

function isRouteActive(
  pathname: string | null,
  route: DashboardRoute,
): boolean {
  if (!pathname) return false;
  if (route.href === ROUTES.DASHBOARD) {
    // Overview is active only on the exact /dashboard path. Otherwise its
    // dot would never go off — every dashboard page starts with /dashboard.
    return pathname === route.href;
  }
  if (route.nestedPrefix) {
    return (
      pathname === route.href ||
      pathname.startsWith(`${route.nestedPrefix}/`)
    );
  }
  return pathname === route.href || pathname.startsWith(`${route.href}/`);
}
