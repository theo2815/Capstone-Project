"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ADMIN_ROUTES,
  isRouteActive,
  useAdminRouteCounts,
} from "@/components/admin/admin-rail";
import { useAdminAttentionTarget } from "@/lib/admin-queues";
import { cn } from "@/lib/utils";

// Mobile-only chip strip for /admin/*. The sticky wrapper lives in
// <AdminShell> so the strip pins under SiteHeader. Renders nothing ≥ md
// (parent has `md:hidden`).
export function AdminMobileStrip() {
  const pathname = usePathname();
  const attention = useAdminAttentionTarget();
  const { getCountForHref } = useAdminRouteCounts();

  return (
    <div className="bg-bone/90 backdrop-blur-md border-b border-line">
      <nav aria-label="Admin sections" className="max-w-7xl mx-auto">
        <ul className="flex gap-2 overflow-x-auto px-6 py-3 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden [mask-image:linear-gradient(to_right,black_85%,transparent)] md:[mask-image:none]">
          {ADMIN_ROUTES.map((route) => {
            const isActive = isRouteActive(pathname, route);
            const isAttention =
              attention !== null && attention.href === route.href;
            const count = getCountForHref(route.href);
            return (
              <li key={route.href} className="shrink-0">
                <Link
                  href={route.href}
                  aria-current={isActive ? "page" : undefined}
                  className={cn(
                    "inline-flex items-center gap-2 px-3 py-1.5 rounded-full border font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] transition-colors",
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
