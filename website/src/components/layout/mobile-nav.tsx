"use client";

import Link from "next/link";
import { useAuth } from "@/hooks/use-auth";
import { ROUTES } from "@/lib/constants";
import { X, Camera } from "lucide-react";

interface MobileNavProps {
  isOpen: boolean;
  onClose: () => void;
}

export function MobileNav({ isOpen, onClose }: MobileNavProps) {
  const { user, isAuthenticated, logout } = useAuth();

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 md:hidden">
      <div className="fixed inset-0 bg-charcoal/50" onClick={onClose} aria-hidden />
      <div className="fixed inset-y-0 right-0 w-full max-w-xs bg-surface p-6 shadow-xl">
        <div className="flex items-center justify-between">
          <Link href={ROUTES.HOME} onClick={onClose} className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
              <Camera className="h-4 w-4 text-white" />
            </div>
            <span className="text-lg font-bold text-charcoal">Quick Pitik</span>
          </Link>
          <button onClick={onClose}>
            <X className="h-6 w-6 text-charcoal" />
          </button>
        </div>

        <nav className="mt-8 flex flex-col gap-4">
          <Link
            href={ROUTES.EVENTS}
            onClick={onClose}
            className="text-base font-medium text-charcoal hover:text-primary transition-colors"
          >
            Events
          </Link>
          <Link
            href="/#how-it-works"
            onClick={onClose}
            className="text-base font-medium text-charcoal hover:text-primary transition-colors"
          >
            How It Works
          </Link>

          <div className="my-2 border-t border-border" />

          {isAuthenticated ? (
            <>
              {user?.role === "PHOTOGRAPHER" && (
                <Link
                  href={ROUTES.DASHBOARD}
                  onClick={onClose}
                  className="text-base font-medium text-charcoal hover:text-primary transition-colors"
                >
                  Dashboard
                </Link>
              )}
              {user?.role === "ADMIN" && (
                <Link
                  href={ROUTES.ADMIN}
                  onClick={onClose}
                  className="text-base font-medium text-charcoal hover:text-primary transition-colors"
                >
                  Admin
                </Link>
              )}
              <Link
                href={ROUTES.CART}
                onClick={onClose}
                className="text-base font-medium text-charcoal hover:text-primary transition-colors"
              >
                Cart
              </Link>
              <Link
                href={ROUTES.ORDERS}
                onClick={onClose}
                className="text-base font-medium text-charcoal hover:text-primary transition-colors"
              >
                Orders
              </Link>
              <Link
                href={ROUTES.PROFILE}
                onClick={onClose}
                className="text-base font-medium text-charcoal hover:text-primary transition-colors"
              >
                Profile
              </Link>
              <button
                onClick={() => {
                  logout();
                  onClose();
                }}
                className="text-left text-base font-medium text-charcoal hover:text-primary transition-colors"
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <Link
                href={ROUTES.LOGIN}
                onClick={onClose}
                className="text-base font-medium text-charcoal hover:text-primary transition-colors"
              >
                Log In
              </Link>
              <Link
                href={ROUTES.REGISTER}
                onClick={onClose}
                className="mt-2 block rounded-lg bg-primary px-4 py-2.5 text-center text-base font-medium text-white hover:bg-primary-hover transition-colors"
              >
                Sign Up
              </Link>
            </>
          )}
        </nav>
      </div>
    </div>
  );
}
