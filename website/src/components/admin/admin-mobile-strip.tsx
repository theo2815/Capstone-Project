"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ADMIN_ROUTES,
  isRouteActive,
  useAdminRouteCounts,
} from "@/components/admin/admin-rail";
import { Kicker, KICKER_SIZE_CLASS } from "@/components/ui/kicker";
import { queuePath, useAdminAttentionTarget } from "@/lib/admin-queues";
import { useAdminPaletteStore } from "@/store/admin-palette-store";
import { cn } from "@/lib/utils";

// Mobile-only chip strip for /admin/*. The sticky wrapper lives in
// <AdminShell> so the strip pins under SiteHeader. Renders nothing ≥ lg
// (parent has `lg:hidden`): tablets get the strip too. Order: Search (opens the palette), then
// each rail group's name before its rows, with the open total after "Queues" as on the rail.
export function AdminMobileStrip() {
  const pathname = usePathname();
  const attention = useAdminAttentionTarget();
  const { queueTotal, getCountForHref } = useAdminRouteCounts();
  const queueRoutes = ADMIN_ROUTES.filter((r) => r.group === "queues");
  const otherGroups = [["catalog", "Catalog"], ["reports", "Reports"], ["account", "Account"]] as const;

  return (
    <div className="bg-bone/90 backdrop-blur-md border-b border-line">
      <nav aria-label="Admin sections" className="max-w-7xl mx-auto">
        <ul className="flex items-center gap-2 overflow-x-auto px-6 py-3 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden [mask-image:linear-gradient(to_right,black_85%,transparent)] lg:[mask-image:none] after:content-[''] after:w-[15%] after:shrink-0 lg:after:hidden">
          <li className="shrink-0">
            <button
              type="button"
              onClick={() => useAdminPaletteStore.getState().setOpen(true)}
              className={`inline-flex items-center gap-2 px-3 min-h-[44px] rounded-full border font-mono uppercase ${KICKER_SIZE_CLASS.sm} transition-colors border-line text-slate hover:text-ink`}
            >
              Search
            </button>
          </li>
          <li className="shrink-0">
            <Kicker tone="soft">Queues</Kicker>
          </li>
          {queueTotal !== null && (
            <li className="shrink-0 pr-1">
              <Kicker tnum>{queueTotal === 0 ? "All clear" : `${queueTotal} waiting`}</Kicker>
            </li>
          )}
          {[
            ...queueRoutes,
            ...otherGroups.flatMap(([group, name]) => [name, ...ADMIN_ROUTES.filter((r) => r.group === group)]),
          ].map((route) => {
            if (typeof route === "string") {
              return (
                <li key={route} className="shrink-0 pl-2">
                  <Kicker tone="soft">{route}</Kicker>
                </li>
              );
            }
            const isActive = isRouteActive(pathname, route);
            const isAttention =
              attention !== null && queuePath(attention.href) === route.href;
            const count = getCountForHref(route.href);
            return (
              <li key={route.href} className="shrink-0">
                <Link
                  href={route.href}
                  aria-current={isActive ? "page" : undefined}
                  className={cn(
                    `inline-flex items-center gap-2 px-3 min-h-[44px] rounded-full border font-mono uppercase ${KICKER_SIZE_CLASS.sm} transition-colors`,
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
                  {count !== null && count > 0 && (
                    <span className="tnum" aria-label={`${count} waiting`}>
                      {count}
                    </span>
                  )}
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
