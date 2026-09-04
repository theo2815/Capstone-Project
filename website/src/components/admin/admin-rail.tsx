"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FocusEvent,
} from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Kicker } from "@/components/ui/kicker";
import { RailTipPortal } from "@/components/ui/rail-tip-portal";
import { ROUTES, ADMIN_FLAGS_ENABLED } from "@/lib/constants";
import { getAdminSectionGuideByHref } from "@/lib/admin-section-guide";
import type { SectionGuideEntry } from "@/lib/section-guide";
import {
  useAdminAttentionTarget,
  useAdminQueueCounts,
} from "@/lib/admin-queues";
import { cn } from "@/lib/utils";

const RAIL_TIPS_STORAGE_KEY = "qp:admin:rail-tips-disabled";

type AdminRouteGroup = "queues" | "catalog" | "reports";

interface AdminRoute {
  readonly href: string;
  readonly label: string;
  readonly number: string;
  readonly group: AdminRouteGroup;
  readonly nestedPrefix?: string;
  // Extra paths that should also activate this row (e.g. legacy
  // /admin/verifications activating the Inbox row when accessed directly).
  readonly aliasPrefixes?: readonly string[];
}

// Phase 1 admin redesign — Inbox-led order, grouped 2026-05-08 into the
// editorial-data sidebar pattern. QUEUES (5) carry live counts and the
// breathing attention dot; CATALOG (2) and REPORTS (1) are router-only.
// Numbers mirror admin-section-guide.ts so the rail and tooltip headings
// stay in sync.
const ADMIN_ROUTES_ALL: ReadonlyArray<AdminRoute> = [
  {
    href: ROUTES.ADMIN_INBOX,
    label: "Inbox",
    number: "01",
    group: "queues",
    nestedPrefix: ROUTES.ADMIN_INBOX,
    aliasPrefixes: [ROUTES.ADMIN_VERIFICATIONS],
  },
  {
    href: ROUTES.ADMIN_DISPUTES,
    label: "Disputes",
    number: "02",
    group: "queues",
    nestedPrefix: ROUTES.ADMIN_DISPUTES,
  },
  {
    href: ROUTES.ADMIN_FLAGS,
    label: "Flags",
    number: "03",
    group: "queues",
    nestedPrefix: ROUTES.ADMIN_FLAGS,
  },
  {
    href: ROUTES.ADMIN_PAYOUTS,
    label: "Payouts",
    number: "04",
    group: "queues",
  },
  {
    href: ROUTES.ADMIN_EVENT_REQUESTS,
    label: "Event requests",
    number: "05",
    group: "queues",
  },
  {
    href: ROUTES.ADMIN_EVENTS,
    label: "Events",
    number: "06",
    group: "catalog",
    nestedPrefix: ROUTES.ADMIN_EVENTS,
  },
  {
    href: ROUTES.ADMIN_PHOTOGRAPHERS,
    label: "Photographers",
    number: "07",
    group: "catalog",
    nestedPrefix: ROUTES.ADMIN_PHOTOGRAPHERS,
  },
  {
    href: ROUTES.ADMIN_OVERVIEW,
    label: "Overview",
    number: "08",
    group: "reports",
  },
  {
    href: ROUTES.ADMIN_SALES,
    label: "Sales",
    number: "09",
    group: "reports",
  },
];

// Filter out the Flags row when the v1 feature gate is off. Numbers stay
// as authored (03 reserved for Flags) so reviving needs zero number churn
// — accept the small visual gap (01 → 02 → 04…) as the honest signal.
const ADMIN_ROUTES: ReadonlyArray<AdminRoute> = ADMIN_FLAGS_ENABLED
  ? ADMIN_ROUTES_ALL
  : ADMIN_ROUTES_ALL.filter((r) => r.href !== ROUTES.ADMIN_FLAGS);

// Persistent rail for /admin/*. Editorial-data sidebar — three semantic
// groups (queues, catalog, reports) with mono numerals and live counts on
// queue rows. Sign-out lives in the SiteHeader's UserMenu (shared with all
// roles); profile/account routes don't apply to admin context.
//
//   - Active row: 1px fresh vertical bar on the left edge + fresh number
//     (both tiny indicators per Quiet Studio rule, count as one active
//     treatment).
//   - Attention dot driven by useAdminAttentionTarget() — at most ONE
//     non-active queue row breathes fresh.
//   - "QUEUES · all clear" tone="active" when total = 0; otherwise the
//     line shows the open count in slate.
//   - Storage key namespaced "qp:admin:rail-tips-disabled" so admin and
//     photographer toggles don't collide.
export function AdminRail() {
  const pathname = usePathname();
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

  const attention = useAdminAttentionTarget();
  const counts = useAdminQueueCounts();
  const allClear = counts.total === 0;

  function getCountForHref(href: string): number | null {
    switch (href) {
      case ROUTES.ADMIN_INBOX:
        return counts.inbox;
      case ROUTES.ADMIN_DISPUTES:
        return counts.disputes;
      case ROUTES.ADMIN_FLAGS:
        return counts.flags;
      case ROUTES.ADMIN_PAYOUTS:
        return counts.payouts;
      case ROUTES.ADMIN_EVENT_REQUESTS:
        return counts.events;
      default:
        return null;
    }
  }

  const queueRoutes = ADMIN_ROUTES.filter((r) => r.group === "queues");
  const catalogRoutes = ADMIN_ROUTES.filter((r) => r.group === "catalog");
  const reportRoutes = ADMIN_ROUTES.filter((r) => r.group === "reports");

  return (
    <aside className="hidden md:block md:sticky md:top-20 md:self-start pt-6 md:pt-10 pb-6 md:pb-8 md:max-h-[calc(100vh-5rem)] md:overflow-y-auto">
      <Kicker as="p" size="sm">
        Admin
      </Kicker>

      <nav aria-label="Admin sections" className="mt-8 space-y-10">
        <div>
          <Kicker
            as="p"
            tone={allClear ? "active" : "default"}
            size="sm"
            tnum
          >
            {allClear ? "Queues · all clear" : `Queues · ${counts.total} open`}
          </Kicker>
          <ul className="mt-4 space-y-3">
            {queueRoutes.map((route) => (
              <RailRow
                key={route.href}
                route={route}
                isActive={isRouteActive(pathname, route)}
                isAttention={
                  attention !== null && attention.href === route.href
                }
                count={getCountForHref(route.href)}
                guide={getAdminSectionGuideByHref(route.href)}
                tipsDisabled={tipsDisabled}
              />
            ))}
          </ul>
        </div>

        <div>
          <Kicker as="p" size="sm">
            Catalog
          </Kicker>
          <ul className="mt-4 space-y-3">
            {catalogRoutes.map((route) => (
              <RailRow
                key={route.href}
                route={route}
                isActive={isRouteActive(pathname, route)}
                isAttention={false}
                count={null}
                guide={getAdminSectionGuideByHref(route.href)}
                tipsDisabled={tipsDisabled}
              />
            ))}
          </ul>
        </div>

        <div>
          <Kicker as="p" size="sm">
            Reports
          </Kicker>
          <ul className="mt-4 space-y-3">
            {reportRoutes.map((route) => (
              <RailRow
                key={route.href}
                route={route}
                isActive={isRouteActive(pathname, route)}
                isAttention={false}
                count={null}
                guide={getAdminSectionGuideByHref(route.href)}
                tipsDisabled={tipsDisabled}
              />
            ))}
          </ul>
        </div>
      </nav>

      <div className="mt-12 pt-6 border-t border-line">
        <Kicker
          as="button"
          size="sm"
          type="button"
          onClick={toggleTipsDisabled}
          aria-pressed={tipsDisabled}
          className="hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
        >
          {tipsDisabled ? "Show tips" : "Hide tips"}
        </Kicker>
      </div>
    </aside>
  );
}

interface RailRowProps {
  route: AdminRoute;
  isActive: boolean;
  isAttention: boolean;
  count: number | null;
  guide: SectionGuideEntry | undefined;
  tipsDisabled: boolean;
}

function RailRow({
  route,
  isActive,
  isAttention,
  count,
  guide,
  tipsDisabled,
}: RailRowProps) {
  const labelRef = useRef<HTMLSpanElement>(null);
  const [hovered, setHovered] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

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
    if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
      setHovered(false);
    }
  }, []);

  const handleTouchToggle = useCallback(() => {
    if (tipsDisabled) return;
    setPinned((p) => !p);
  }, [tipsDisabled]);

  const showTip = !tipsDisabled && !!guide && (hovered || pinned);
  const tipId = `admin-rail-tip-${route.href.replace(/\//g, "-")}`;
  const showCount = count !== null && count > 0;

  return (
    <li
      className="relative"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onFocus={handleFocus}
      onBlur={handleBlur}
    >
      {isActive && (
        <span
          aria-hidden="true"
          className="absolute -left-3 top-1/2 -translate-y-1/2 h-4 w-px bg-fresh"
        />
      )}
      <div className="flex items-center gap-2">
        <Link
          href={route.href}
          aria-current={isActive ? "page" : undefined}
          aria-describedby={showTip ? tipId : undefined}
          className={cn(
            "group flex flex-1 items-baseline gap-3 transition-colors min-w-0",
            isActive ? "text-ink" : "text-slate hover:text-ink",
          )}
        >
          <span
            aria-hidden="true"
            className={cn(
              "font-mono tnum text-[14px] min-[400px]:text-[15px] md:text-[13px] shrink-0 transition-colors",
              isActive
                ? "text-fresh"
                : "text-slate-soft group-hover:text-slate",
            )}
          >
            {route.number}
          </span>
          <span ref={labelRef} className="font-display text-base">
            {route.label}
          </span>
          <span className="ml-auto flex items-baseline gap-2 shrink-0">
            {isAttention && !isActive && (
              <span
                aria-label="Queue needs attention"
                className="size-1.5 rounded-full bg-fresh breathe"
              />
            )}
            {showCount && (
              <span
                className={cn(
                  "font-mono tnum text-[14px] min-[400px]:text-[15px] md:text-[13px]",
                  isActive ? "text-ink" : "text-slate",
                )}
                aria-label={`${count} open`}
              >
                {count}
              </span>
            )}
          </span>
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
  route: AdminRoute,
): boolean {
  if (!pathname) return false;
  if (route.aliasPrefixes) {
    for (const prefix of route.aliasPrefixes) {
      if (pathname === prefix || pathname.startsWith(`${prefix}/`)) {
        return true;
      }
    }
  }
  if (route.nestedPrefix) {
    return (
      pathname === route.href || pathname.startsWith(`${route.nestedPrefix}/`)
    );
  }
  return pathname === route.href || pathname.startsWith(`${route.href}/`);
}
