"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/hooks/use-auth";
import { UserMenu } from "@/components/layout/user-menu";
import { BrandLogo } from "@/components/layout/brand-logo";
import { NotificationBell } from "@/components/layout/notification-bell";
import { RunnerNotificationBell } from "@/components/layout/runner-notification-bell";
import { useViewModeStore } from "@/store/view-mode-store";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";

type RightLink = { label: string; href: string };

interface SiteHeaderProps {
  rightLink?: RightLink | null;
  /** Span the full width instead of max-w-7xl, so the logo lines up with a full-bleed shell's rail. */
  fullWidth?: boolean;
}

const DEFAULT_RIGHT_LINK: RightLink = { label: "Sign in", href: ROUTES.LOGIN };

export function SiteHeader({
  rightLink = DEFAULT_RIGHT_LINK,
  fullWidth = false,
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
  // Signing in from an event page comes back to that event instead of the
  // role home, so a runner doesn't have to find their race again.
  const onEventPage = pathname.startsWith(ROUTES.EVENTS + "/");
  const rightHref =
    rightLink && rightLink.href === ROUTES.LOGIN && onEventPage
      ? `${ROUTES.LOGIN}?redirect=${encodeURIComponent(pathname)}`
      : rightLink?.href;

  return (
    // Height is pinned to --site-header-h rather than falling out of the
    // padding, because every sticky bar on the site offsets against it. Left
    // to `py-4` the header measured 69px signed-in but ~57px signed-out (the
    // bells/avatar are the tallest child, and guests don't get them),
    // so no single offset could sit flush in both states. Every control in it is
    // a 44 px target (FU-23), which still fits inside the pinned height.
    <header className="sticky top-0 z-30 h-[var(--site-header-h)] bg-bone/85 backdrop-blur-md border-b border-line">
      <div className={`px-6 md:px-10 h-full flex items-center justify-between ${fullWidth ? "" : "max-w-7xl mx-auto"}`}>
        <Link
          href={ROUTES.RUNNERS}
          className="flex min-h-11 items-center text-ink transition-opacity hover:opacity-75"
          aria-label="Back to QuickPitik home"
        >
          <BrandLogo className="md:h-11 md:w-44" priority />
        </Link>
        {/* Tighter below 400 px so "Sign in" (which no longer wraps) fits narrow phones,
            and so Events, a 44 px bell and the 44 px avatar still fit beside the logo. */}
        <nav className="flex items-center gap-2 min-[400px]:gap-6 md:gap-8">
          {!hidePublicNav && (
            <Link
              href={ROUTES.EVENTS}
              aria-current={isEventsActive ? "page" : undefined}
              className={cn(
                "relative inline-flex min-h-11 items-center gap-2 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] transition-colors",
                isEventsActive
                  ? "text-ink"
                  : "text-slate hover:text-ink",
              )}
            >
              {/* The dot is hidden below 400 px so Events, a bell and the avatar fit at 360 px; the underline still marks the page. */}
              {isEventsActive && (
                <span
                  className="hidden min-[400px]:block size-1.5 rounded-full bg-fresh"
                  aria-hidden="true"
                />
              )}
              {/* nav-link sits on the text, not the 44 px link, so its hover underline stays under the word. */}
              <span className={isEventsActive ? undefined : "nav-link"}>Events</span>
              {isEventsActive && (
                <span
                  className="pointer-events-none absolute -bottom-1 left-0 right-0 h-px bg-fresh"
                  aria-hidden="true"
                />
              )}
            </Link>
          )}
          {isLoading ? (
            <div
              className="size-11 rounded-full bg-line/40 animate-pulse"
              aria-hidden="true"
            />
          ) : isAuthenticated && user ? (
            <>
              <NotificationBell />
              <RunnerNotificationBell />
              <UserMenu user={user} />
            </>
          ) : rightLink && rightHref ? (
            <Link
              href={rightHref}
              className="inline-flex min-h-11 items-center font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink hover:text-fresh transition-colors whitespace-nowrap"
            >
              <span className="nav-link">{rightLink.label}</span>
            </Link>
          ) : null}
        </nav>
      </div>
    </header>
  );
}
