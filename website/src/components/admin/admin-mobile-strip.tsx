"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ROUTES, ADMIN_FLAGS_ENABLED } from "@/lib/constants";
import { getAdminAttentionTarget } from "@/lib/admin-queues";
import { cn } from "@/lib/utils";

interface AdminRoute {
  readonly href: string;
  readonly label: string;
  readonly nestedPrefix?: string;
  readonly aliasPrefixes?: readonly string[];
}

// Mirror of <AdminRail>'s ADMIN_ROUTES. Inbox-led order, Overview last.
const ADMIN_ROUTES_ALL: ReadonlyArray<AdminRoute> = [
  {
    href: ROUTES.ADMIN_INBOX,
    label: "Inbox",
    nestedPrefix: ROUTES.ADMIN_INBOX,
    aliasPrefixes: [ROUTES.ADMIN_VERIFICATIONS],
  },
  {
    href: ROUTES.ADMIN_DISPUTES,
    label: "Disputes",
    nestedPrefix: ROUTES.ADMIN_DISPUTES,
  },
  {
    href: ROUTES.ADMIN_FLAGS,
    label: "Flags",
    nestedPrefix: ROUTES.ADMIN_FLAGS,
  },
  { href: ROUTES.ADMIN_PAYOUTS, label: "Payouts" },
  {
    href: ROUTES.ADMIN_EVENTS,
    label: "Events",
    nestedPrefix: ROUTES.ADMIN_EVENTS,
  },
  {
    href: ROUTES.ADMIN_PHOTOGRAPHERS,
    label: "Photographers",
    nestedPrefix: ROUTES.ADMIN_PHOTOGRAPHERS,
  },
  { href: ROUTES.ADMIN_OVERVIEW, label: "Overview" },
];

const ADMIN_ROUTES: ReadonlyArray<AdminRoute> = ADMIN_FLAGS_ENABLED
  ? ADMIN_ROUTES_ALL
  : ADMIN_ROUTES_ALL.filter((r) => r.href !== ROUTES.ADMIN_FLAGS);

// Mobile-only chip strip for /admin/*. Mirrors `<DashboardMobileStrip>`:
// sticky wrapper lives in <AdminShell> so the strip pins under SiteHeader
// alongside <DesktopNudge>. Renders nothing ≥ md (parent has `md:hidden`).
export function AdminMobileStrip() {
  const pathname = usePathname();
  const attention = getAdminAttentionTarget();

  return (
    <div className="bg-bone/90 backdrop-blur-md border-b border-line">
      <nav aria-label="Admin sections" className="max-w-7xl mx-auto">
        <ul className="flex gap-2 overflow-x-auto px-6 py-3 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {ADMIN_ROUTES.map((route) => {
            const isActive = isRouteActive(pathname, route);
            const isAttention =
              attention !== null && attention.href === route.href;
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
                  {isAttention && !isActive && (
                    <span
                      aria-label={`${attention.label} need attention`}
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
