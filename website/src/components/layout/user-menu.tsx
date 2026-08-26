"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/hooks/use-auth";
import { useConfirmation } from "@/hooks/use-confirmation";
import { useUserMediaStore } from "@/store/user-media-store";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { Role, User } from "@/types/user";

interface UserMenuProps {
  user: User;
}

interface MenuLink {
  label: string;
  href: string;
}

function getInitials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function linksForRole(role: Role): MenuLink[] {
  switch (role) {
    case "ADMIN":
      return [
        { label: "Admin", href: ROUTES.ADMIN },
        { label: "Profile", href: ROUTES.PROFILE },
        { label: "Account", href: ROUTES.ACCOUNT },
      ];
    case "PHOTOGRAPHER":
      return [
        { label: "Dashboard", href: ROUTES.DASHBOARD },
        { label: "Profile", href: ROUTES.PROFILE },
        { label: "Account", href: ROUTES.ACCOUNT },
      ];
    case "RUNNER":
      return [
        { label: "Orders", href: ROUTES.ORDERS },
        { label: "Profile", href: ROUTES.PROFILE },
        { label: "Account", href: ROUTES.ACCOUNT },
      ];
  }
}

function pad2(n: number): string {
  return n.toString().padStart(2, "0");
}

export function UserMenu({ user }: UserMenuProps) {
  const router = useRouter();
  const { logout } = useAuth();
  const { confirm } = useConfirmation();
  const avatar = useUserMediaStore((s) => s.avatar);
  const [isOpen, setIsOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!isOpen) return;

    function onPointerDown(e: MouseEvent) {
      if (
        wrapRef.current &&
        !wrapRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        setIsOpen(false);
        triggerRef.current?.focus();
      }
    }

    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [isOpen]);

  async function handleSignOut() {
    // Confirmation gate is mobile-only — desktop sign-out stays a single click
    // (intentional dropdown action, low miss-tap risk). Tailwind's md breakpoint
    // is 768px, so anything narrower gets the modal.
    const isMobile =
      typeof window !== "undefined" &&
      window.matchMedia("(max-width: 767px)").matches;

    if (isMobile) {
      setIsOpen(false);
      const ok = await confirm({
        title: "Sign out?",
        message: "You'll need to sign in again to access your account.",
        confirmLabel: "Sign out",
      });
      if (!ok) return;
    } else {
      setIsOpen(false);
    }

    logout();
    // Sign-out always lands on /login. The pre-2026-05-20 path was the
    // splash chooser (/), but that's confusing for a user who just signed
    // out — they almost always either want to sign back in or pick a
    // different account. Landing on /login matches that intent directly.
    router.replace(ROUTES.LOGIN);
  }

  const initials = getInitials(user.name);
  const links = linksForRole(user.role);

  return (
    <div className="relative" ref={wrapRef}>
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={isOpen}
        aria-label={`Account menu for ${user.name}`}
        onClick={() => setIsOpen((o) => !o)}
        className={cn(
          "size-9 rounded-full overflow-hidden",
          !avatar && "bg-fresh text-surface hover:bg-fresh-deep",
          avatar && "bg-bone-deep",
          "font-mono uppercase tracking-[0.05em] text-[11px] font-semibold",
          "flex items-center justify-center",
          "ring-2 ring-transparent ring-offset-2 ring-offset-bone",
          "transition-[background-color,box-shadow] duration-200",
          "hover:ring-line",
          "focus-visible:outline-none focus-visible:ring-fresh",
          isOpen && "ring-line",
        )}
      >
        {avatar ? (
          /* eslint-disable-next-line @next/next/no-img-element -- data-URL avatar mock; backend will return signed S3 URLs. */
          <img
            src={avatar.dataUrl}
            alt=""
            className="size-full object-cover"
            draggable={false}
          />
        ) : (
          initials
        )}
      </button>

      {isOpen && (
        <div
          role="menu"
          aria-label="Account"
          className={cn(
            "absolute right-0 top-full mt-3 z-40",
            "w-[300px] origin-top-right",
            "bg-bone border border-line rounded-2xl overflow-hidden",
            "shadow-[0_18px_50px_-12px_rgba(17,17,17,0.14)]",
            "animate-[user-menu-in_180ms_cubic-bezier(0.16,1,0.3,1)_both]",
          )}
        >
          <div className="px-6 pt-6 pb-5">
            <p className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate-soft">
              Signed in
            </p>
            <p className="font-display text-2xl text-ink leading-[1.1] tracking-tight mt-3 truncate">
              {user.name}
            </p>
            <div className="flex items-center gap-2 mt-3">
              <span
                className="size-1.5 rounded-full bg-fresh"
                aria-hidden="true"
              />
              <span className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate">
                {user.role}
              </span>
            </div>
            <p className="font-mono text-[11px] text-slate-soft mt-4 truncate">
              {user.email}
            </p>
          </div>

          <ul className="border-t border-line">
            {links.map((link, idx) => (
              <li key={link.href}>
                <Link
                  href={link.href}
                  role="menuitem"
                  onClick={() => setIsOpen(false)}
                  className="group flex items-center gap-5 px-6 py-3.5 hover:bg-bone-deep transition-colors"
                >
                  <span
                    aria-hidden="true"
                    className="font-mono text-[10px] tracking-[0.15em] text-slate-soft tnum group-hover:text-fresh transition-colors"
                  >
                    {pad2(idx + 1)}
                  </span>
                  <span className="flex-1 font-display text-base text-ink">
                    {link.label}
                  </span>
                </Link>
              </li>
            ))}
          </ul>

          <div className="border-t border-line">
            <button
              type="button"
              role="menuitem"
              onClick={handleSignOut}
              className="group w-full flex items-center justify-between px-6 py-3.5 font-mono uppercase tracking-[0.14em] text-[10px] text-slate hover:text-ink hover:bg-bone-deep transition-colors"
            >
              <span>Sign out</span>
              <svg
                aria-hidden="true"
                viewBox="0 0 16 16"
                fill="none"
                className="size-4 text-slate-soft group-hover:text-ink transition-colors"
              >
                <path
                  d="M9.5 2H3.5C2.94772 2 2.5 2.44772 2.5 3V13C2.5 13.5523 2.94772 14 3.5 14H9.5"
                  stroke="currentColor"
                  strokeWidth="1.25"
                  strokeLinecap="round"
                />
                <path
                  d="M11 5L14 8L11 11"
                  stroke="currentColor"
                  strokeWidth="1.25"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M14 8H6.5"
                  stroke="currentColor"
                  strokeWidth="1.25"
                  strokeLinecap="round"
                />
              </svg>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
