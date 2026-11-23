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
// <AdminShell> so the strip pins under SiteHeader. Renders nothing ≥ md
// (parent has `md:hidden`). Order: Search (opens the palette), the open
// total, the queue rows, a divider, then the other rows.
export function AdminMobileStrip() {
  const pathname = usePathname();
  const attention = useAdminAttentionTarget();
  const { queueTotal, getCountForHref } = useAdminRouteCounts();
  const queueRoutes = ADMIN_ROUTES.filter((r) => r.group === "queues");
  const otherRoutes = ADMIN_ROUTES.filter((r) => r.group !== "queues");

  return (
    <div className="bg-bone/90 backdrop-blur-md border-b border-line">
      <nav aria-label="Admin sections" className="max-w-7xl mx-auto">
        <ul className="flex items-center gap-2 overflow-x-auto px-6 py-3 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden [mask-image:linear-gradient(to_right,black_85%,transparent)] md:[mask-image:none] after:content-[''] after:w-[15%] after:shrink-0 md:after:hidden">
          <li className="shrink-0">
            <button
              type="button"
              onClick={() => useAdminPaletteStore.getState().setOpen(true)}
              className={`inline-flex items-center gap-2 px-3 min-h-[44px] rounded-full border font-mono uppercase ${KICKER_SIZE_CLASS.sm} transition-colors border-line text-slate hover:text-ink`}
            >
              Search
            </button>
          </li>
          {queueTotal !== null && (
            <li className="shrink-0 pr-1">
              <Kicker tnum>{queueTotal === 0 ? "All clear" : `${queueTotal} open`}</Kicker>
            </li>
          )}
          {[...queueRoutes, null, ...otherRoutes].map((route) => {
            if (route === null) {
              return <li key="divider" data-divider aria-hidden="true" className="shrink-0 w-px h-6 bg-line" />;
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
                    <span className="tnum" aria-label={`${count} open`}>
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
