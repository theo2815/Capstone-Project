"use client";

import Link from "next/link";
import { useAuth } from "@/hooks/use-auth";
import { useCart } from "@/hooks/use-cart";
import { ROUTES } from "@/lib/constants";
import { ShoppingCart, User, Menu, Camera } from "lucide-react";
import { useState } from "react";
import { MobileNav } from "./mobile-nav";

export function Navbar() {
  const { user, isAuthenticated, logout } = useAuth();
  const { itemCount } = useCart();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 border-b border-border bg-surface/95 backdrop-blur-sm">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Logo */}
        <Link href={ROUTES.HOME} className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
            <Camera className="h-4 w-4 text-white" />
          </div>
          <span className="text-lg font-bold text-charcoal">
            Quick Pitik
          </span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden items-center gap-8 md:flex">
          <Link
            href={ROUTES.EVENTS}
            className="text-sm font-medium text-charcoal-light hover:text-primary transition-colors"
          >
            Events
          </Link>
          <Link
            href="/#how-it-works"
            className="text-sm font-medium text-charcoal-light hover:text-primary transition-colors"
          >
            How It Works
          </Link>

          {isAuthenticated ? (
            <div className="flex items-center gap-5">
              {user?.role === "PHOTOGRAPHER" && (
                <Link
                  href={ROUTES.DASHBOARD}
                  className="text-sm font-medium text-charcoal-light hover:text-primary transition-colors"
                >
                  Dashboard
                </Link>
              )}
              {user?.role === "ADMIN" && (
                <Link
                  href={ROUTES.ADMIN}
                  className="text-sm font-medium text-charcoal-light hover:text-primary transition-colors"
                >
                  Admin
                </Link>
              )}
              <Link href={ROUTES.CART} className="relative">
                <ShoppingCart className="h-5 w-5 text-charcoal-light hover:text-primary transition-colors" />
                {itemCount > 0 && (
                  <span className="absolute -right-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full bg-primary text-xs text-white">
                    {itemCount}
                  </span>
                )}
              </Link>
              <Link href={ROUTES.PROFILE}>
                <User className="h-5 w-5 text-charcoal-light hover:text-primary transition-colors" />
              </Link>
              <button
                onClick={logout}
                className="text-sm font-medium text-charcoal-light hover:text-primary transition-colors"
              >
                Logout
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <Link
                href={ROUTES.LOGIN}
                className="text-sm font-medium text-charcoal-light hover:text-primary transition-colors"
              >
                Log In
              </Link>
              <Link
                href={ROUTES.REGISTER}
                className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover transition-colors"
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
        >
          <Menu className="h-6 w-6 text-charcoal" />
        </button>
      </div>

      <MobileNav isOpen={mobileOpen} onClose={() => setMobileOpen(false)} />
    </header>
  );
}
