"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import type { ReactNode } from "react";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { useAuth } from "@/hooks/use-auth";
import { useActiveSection } from "@/hooks/use-active-section";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { Role, User } from "@/types/user";

export interface JumpSection {
  readonly id: string;
  readonly label: string;
}

interface IdentityRailProps {
  user: User;
  /**
   * Mono uppercase kicker line. Profile pages pass identity context
   * ("Runner · Cebu · Since MAY 2026"); page-led pages pass a path
   * ("Profile · Receipts · Since MAY 2026").
   */
  kicker: ReactNode;
  /**
   * Display headline. Profile pages pass the user's name; page-led pages
   * pass a page title (e.g., "Orders.", "Dashboard."). Plain text or JSX
   * — wrap a span in `text-fresh` to fresh-accent the title.
   */
  headline: ReactNode;
  /**
   * Headline size — `name` is softer for identity-led pages (/profile),
   * `title` is bigger for page-led pages (/account, /orders, /dashboard).
   * Default `title`.
   */
  headlineSize?: "name" | "title";
  /** Optional sub-paragraph under the headline (email or description). */
  subline?: ReactNode;
  /** Sections rendered in the JUMP TO list. Pass [] to hide the nav. */
  jumpSections: ReadonlyArray<JumpSection>;
  /** Current path — used to filter the MORE list so we don't link back to self. */
  currentPath: string;
}

const HEADLINE_SIZE: Record<NonNullable<IdentityRailProps["headlineSize"]>, string> = {
  name: "text-2xl md:text-3xl",
  title: "text-3xl md:text-4xl",
};

export function IdentityRail({
  user,
  kicker,
  headline,
  headlineSize = "title",
  subline,
  jumpSections,
  currentPath,
}: IdentityRailProps) {
  const router = useRouter();
  const { logout } = useAuth();
  const activeId = useActiveSection(jumpSections.map((s) => s.id));
  const moreLinks = moreLinksForRole(user.role, currentPath);

  function handleSignOut() {
    logout();
    router.replace(ROUTES.HOME);
  }

  return (
    <aside className="md:sticky md:top-20 md:self-start pt-6 md:pt-10 pb-6 md:pb-8 md:max-h-[calc(100vh-5rem)] md:overflow-y-auto border-b md:border-0 border-line">
      <div className="flex items-start gap-5 md:block">
        <AvatarDisc name={user.name} size="md" />
        <div className="flex-1 min-w-0 md:mt-7">
          <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate">
            {kicker}
          </p>
          <h1
            className={cn(
              "font-display font-extrabold tracking-tight leading-[1.05] text-ink mt-3",
              HEADLINE_SIZE[headlineSize],
            )}
          >
            {headline}
          </h1>
          {subline && (
            <div className="font-sans text-sm text-slate mt-2 break-words">
              {subline}
            </div>
          )}
        </div>
      </div>

      {jumpSections.length > 0 && (
        <nav aria-label="Jump to section" className="hidden md:block mt-10">
          <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
            Jump to
          </p>
          <ul className="mt-4 space-y-3">
            {jumpSections.map((section) => {
              const isActive = activeId === section.id;
              return (
                <li key={section.id}>
                  <a
                    href={`#${section.id}`}
                    aria-current={isActive ? "true" : undefined}
                    className={cn(
                      "group flex items-center gap-3 font-display text-base transition-colors",
                      isActive ? "text-ink" : "text-slate hover:text-ink",
                    )}
                  >
                    <span
                      aria-hidden="true"
                      className={cn(
                        "size-1.5 rounded-full transition-colors",
                        isActive
                          ? "bg-fresh"
                          : "bg-line group-hover:bg-slate",
                      )}
                    />
                    <span>{section.label}</span>
                  </a>
                </li>
              );
            })}
          </ul>
        </nav>
      )}

      <div className="mt-8 md:mt-10">
        <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
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

// Role-aware MORE list. Filters out the current page so the rail doesn't
// link back to itself. Order matters: Profile and Account come first since
// every role sees them; role-specific entries (Orders for runners, Dashboard
// for photographers, Admin for admins) trail.
export function moreLinksForRole(
  role: Role,
  current: string,
): ReadonlyArray<{ label: string; href: string }> {
  const all: Array<{ label: string; href: string }> = [
    { label: "Profile", href: ROUTES.PROFILE },
    { label: "Account", href: ROUTES.ACCOUNT },
  ];
  if (role === "RUNNER") all.push({ label: "Orders", href: ROUTES.ORDERS });
  if (role === "PHOTOGRAPHER")
    all.push({ label: "Dashboard", href: ROUTES.DASHBOARD });
  if (role === "ADMIN") all.push({ label: "Admin", href: ROUTES.ADMIN });
  return all.filter((l) => l.href !== current);
}
