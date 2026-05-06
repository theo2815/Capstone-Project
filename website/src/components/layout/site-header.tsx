"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/hooks/use-auth";
import { UserMenu } from "@/components/layout/user-menu";
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
  const isEventsActive = pathname === ROUTES.EVENTS;

  return (
    <header className="sticky top-0 z-30 bg-bone/85 backdrop-blur-md border-b border-line">
      <div className="px-6 md:px-10 py-4 flex items-center justify-between max-w-7xl mx-auto">
        <Link
          href={ROUTES.HOME}
          className="flex items-center gap-2 text-ink"
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
          <span className="font-display text-base font-semibold tracking-tight">
            QuickPitik
          </span>
        </Link>
        <nav className="flex items-center gap-6 md:gap-8">
          <Link
            href={ROUTES.EVENTS}
            aria-current={isEventsActive ? "page" : undefined}
            className={cn(
              "relative inline-flex items-center gap-2 font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] transition-colors",
              isEventsActive ? "text-ink" : "text-slate hover:text-ink",
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
          {isLoading ? (
            <div
              className="size-9 rounded-full bg-line/40 animate-pulse"
              aria-hidden="true"
            />
          ) : isAuthenticated && user ? (
            <UserMenu user={user} />
          ) : rightLink ? (
            <Link
              href={rightLink.href}
              className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink hover:text-fresh transition-colors"
            >
              {rightLink.label}
            </Link>
          ) : null}
        </nav>
      </div>
    </header>
  );
}
