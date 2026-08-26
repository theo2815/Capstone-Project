"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/hooks/use-auth";
import { UserMenu } from "@/components/layout/user-menu";
import { NotificationBell } from "@/components/layout/notification-bell";
import { RunnerNotificationBell } from "@/components/layout/runner-notification-bell";
import { useViewModeStore } from "@/store/view-mode-store";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";

type RightLink = { label: string; href: string };

interface SiteHeaderProps {
  rightLink?: RightLink | null;
}

const DEFAULT_RIGHT_LINK: RightLink = { label: "Sign in", href: ROUTES.LOGIN };

export function SiteHeader({
  rightLink = DEFAULT_RIGHT_LINK,
}: SiteHeaderProps) {
  const pathname = usePathname();
  const { user, isAuthenticated, isLoading } = useAuth();
  const viewMode = useViewModeStore((s) => s.viewMode);
  // Photographer-only chrome: a photographer in photographer mode shouldn't see
  // the runner-facing Events browse link. Guests, runners, admins, and a
  // photographer who switched to runner view keep it. (user is null until
  // AuthHydrator populates it, so guests + first paint show the link.)
  const hidePublicNav =
    isAuthenticated &&
    user?.role === "PHOTOGRAPHER" &&
    viewMode === "photographer";
  // Prefix match so /events/[slug] (where users spend the most time) still
  // shows the "you are here" state.
  const isEventsActive =
    pathname === ROUTES.EVENTS || pathname.startsWith(ROUTES.EVENTS + "/");

  return (
    // Height is pinned to --site-header-h rather than falling out of the
    // padding, because every sticky bar on the site offsets against it. Left
    // to `py-4` the header measured 69px signed-in but ~57px signed-out (the
    // size-9 bells/avatar are the tallest child, and guests don't get them),
    // so no single offset could sit flush in both states.
    <header className="sticky top-0 z-30 h-[var(--site-header-h)] bg-bone/85 backdrop-blur-md border-b border-line">
      <div className="px-6 md:px-10 h-full flex items-center justify-between max-w-7xl mx-auto">
        <Link
          href={ROUTES.HOME}
          className="flex items-center gap-2 text-ink transition-opacity hover:opacity-75"
          aria-label="Back to QuickPitik home"
        >
          <svg
            className="size-6"
            viewBox="0 0 28 28"
            fill="none"
            aria-hidden="true"
          >
            <circle
              cx="14"
              cy="14"
              r="13"
              stroke="currentColor"
              strokeWidth="1.5"
            />
            <circle cx="14" cy="14" r="5" className="fill-fresh" />
          </svg>
          <span className="font-display text-base font-extrabold tracking-tight">
            QuickPitik
          </span>
        </Link>
        <nav className="flex items-center gap-6 md:gap-8">
          {!hidePublicNav && (
            <Link
              href={ROUTES.EVENTS}
              aria-current={isEventsActive ? "page" : undefined}
              className={cn(
                "relative inline-flex items-center gap-2 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] transition-colors",
                isEventsActive
                  ? "text-ink"
                  : "nav-link text-slate hover:text-ink",
              )}
            >
              {isEventsActive && (
                <span
                  className="size-1.5 rounded-full bg-fresh"
                  aria-hidden="true"
                />
              )}
              <span>Events</span>
              {isEventsActive && (
                <span
                  className="pointer-events-none absolute -bottom-4 left-0 right-0 h-px bg-fresh"
                  aria-hidden="true"
                />
              )}
            </Link>
          )}
          {isLoading ? (
            <div
              className="size-9 rounded-full bg-line/40 animate-pulse"
              aria-hidden="true"
            />
          ) : isAuthenticated && user ? (
            <>
              <NotificationBell />
              <RunnerNotificationBell />
              <UserMenu user={user} />
            </>
          ) : rightLink ? (
            <Link
              href={rightLink.href}
              className="nav-link font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink hover:text-fresh transition-colors"
            >
              {rightLink.label}
            </Link>
          ) : null}
        </nav>
      </div>
    </header>
  );
}
