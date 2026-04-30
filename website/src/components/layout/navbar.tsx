"use client";

import Link from "next/link";
import { useAuth } from "@/hooks/use-auth";
import { useCart } from "@/hooks/use-cart";
import { ROUTES } from "@/lib/constants";
import { ShoppingCart, User, Menu } from "lucide-react";
import { useState, useEffect } from "react";
import { MobileNav } from "./mobile-nav";
import { cn } from "@/lib/utils";

export function Navbar() {
  const { user, isAuthenticated, logout } = useAuth();
  const { itemCount } = useCart();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 32);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={cn(
        "fixed top-0 left-0 right-0 z-40 transition-all duration-300",
        scrolled
          ? "border-b border-paper/[0.08] bg-ink/90 backdrop-blur-xl shadow-lg shadow-black/20"
          : "bg-transparent"
      )}
    >
      <div className="mx-auto flex h-16 max-w-[1320px] items-center justify-between px-6 lg:px-12">
        {/* Logo — wordmark with signal-dot */}
        <Link href={ROUTES.HOME} className="group flex items-center gap-2.5">
          <span
            className="h-2 w-2 rounded-full bg-signal"
            style={{ animation: "rec-pulse 1.4s ease-in-out infinite" }}
          />
          <span className="font-display text-[22px] font-black uppercase tracking-[0.04em] text-paper">
            Quick&nbsp;Pitik
          </span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden items-center gap-8 md:flex">
          <Link
            href={ROUTES.EVENTS}
            className="nav-link text-[13px] font-medium uppercase tracking-widest text-cool-gray hover:text-white transition-colors duration-200"
          >
            Events
          </Link>
          <Link
            href="/#how-it-works"
            className="nav-link text-[13px] font-medium uppercase tracking-widest text-cool-gray hover:text-white transition-colors duration-200"
          >
            How It Works
          </Link>

          {isAuthenticated ? (
            <div className="flex items-center gap-5 pl-4 border-l border-white/10">
              {user?.role === "PHOTOGRAPHER" && (
                <Link
                  href={ROUTES.DASHBOARD}
                  className="nav-link text-[13px] font-medium uppercase tracking-widest text-cool-gray hover:text-white transition-colors duration-200"
                >
                  Dashboard
                </Link>
              )}
              {user?.role === "ADMIN" && (
                <Link
                  href={ROUTES.ADMIN}
                  className="nav-link text-[13px] font-medium uppercase tracking-widest text-cool-gray hover:text-white transition-colors duration-200"
                >
                  Admin
                </Link>
              )}
              <Link href={ROUTES.CART} className="relative group">
                <ShoppingCart className="h-[18px] w-[18px] text-cool-gray group-hover:text-white transition-colors duration-200" />
                {itemCount > 0 && (
                  <span className="absolute -right-2 -top-2 flex h-4 w-4 items-center justify-center rounded-full bg-signal text-[10px] font-bold text-ink tnum">
                    {itemCount}
                  </span>
                )}
              </Link>
              <Link href={ROUTES.PROFILE} className="group">
                <User className="h-[18px] w-[18px] text-cool-gray group-hover:text-white transition-colors duration-200" />
              </Link>
              <button
                onClick={logout}
                className="text-[13px] font-medium text-cool-gray-dark hover:text-white transition-colors duration-200"
              >
                Logout
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-4 pl-4 border-l border-white/10">
              <Link
                href={ROUTES.LOGIN}
                className="text-[13px] font-medium text-cool-gray hover:text-white transition-colors duration-200"
              >
                Log In
              </Link>
              <Link
                href={ROUTES.REGISTER}
                className="bg-signal px-5 py-2.5 font-mono text-[11px] font-semibold uppercase tracking-[0.2em] text-ink hover:bg-signal-bright transition-colors duration-200"
              >
                Sign Up
              </Link>
            </div>
          )}
        </nav>

        {/* Mobile hamburger */}
        <button
          className="md:hidden"
          onClick={() => setMobileOpen(true)}
          aria-label="Open menu"
        >
          <Menu className="h-5 w-5 text-white" />
        </button>
      </div>

      <MobileNav isOpen={mobileOpen} onClose={() => setMobileOpen(false)} />
    </header>
  );
}
