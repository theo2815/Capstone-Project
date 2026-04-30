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
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} aria-hidden />
      <div className="fixed inset-y-0 right-0 w-full max-w-sm bg-charcoal-deep p-6 shadow-2xl animate-slide-in-right">
        <div className="flex items-center justify-between">
          <Link href={ROUTES.HOME} onClick={onClose} className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
              <Camera className="h-4 w-4 text-white" />
            </div>
            <span className="text-lg font-bold text-white">Quick Pitik</span>
          </Link>
          <button onClick={onClose} aria-label="Close menu">
            <X className="h-5 w-5 text-cool-gray hover:text-white transition-colors" />
          </button>
        </div>

        <nav className="mt-10 flex flex-col gap-1 stagger-children">
          <Link
            href={ROUTES.EVENTS}
            onClick={onClose}
            className="rounded-lg px-3 py-3 text-[15px] font-medium text-cool-gray hover:bg-white/5 hover:text-white transition-colors"
          >
            Events
          </Link>
          <Link
            href="/#how-it-works"
            onClick={onClose}
            className="rounded-lg px-3 py-3 text-[15px] font-medium text-cool-gray hover:bg-white/5 hover:text-white transition-colors"
          >
            How It Works
          </Link>

          <div className="my-3 border-t border-white/[0.06]" />

          {isAuthenticated ? (
            <>
              {user?.role === "PHOTOGRAPHER" && (
                <Link
                  href={ROUTES.DASHBOARD}
                  onClick={onClose}
                  className="rounded-lg px-3 py-3 text-[15px] font-medium text-cool-gray hover:bg-white/5 hover:text-white transition-colors"
                >
                  Dashboard
                </Link>
              )}
              {user?.role === "ADMIN" && (
                <Link
                  href={ROUTES.ADMIN}
                  onClick={onClose}
                  className="rounded-lg px-3 py-3 text-[15px] font-medium text-cool-gray hover:bg-white/5 hover:text-white transition-colors"
                >
                  Admin
                </Link>
              )}
              <Link
                href={ROUTES.CART}
                onClick={onClose}
                className="rounded-lg px-3 py-3 text-[15px] font-medium text-cool-gray hover:bg-white/5 hover:text-white transition-colors"
              >
                Cart
              </Link>
              <Link
                href={ROUTES.ORDERS}
                onClick={onClose}
                className="rounded-lg px-3 py-3 text-[15px] font-medium text-cool-gray hover:bg-white/5 hover:text-white transition-colors"
              >
                Orders
              </Link>
              <Link
                href={ROUTES.PROFILE}
                onClick={onClose}
                className="rounded-lg px-3 py-3 text-[15px] font-medium text-cool-gray hover:bg-white/5 hover:text-white transition-colors"
              >
                Profile
              </Link>
              <button
                onClick={() => {
                  logout();
                  onClose();
                }}
                className="rounded-lg px-3 py-3 text-left text-[15px] font-medium text-cool-gray-dark hover:bg-white/5 hover:text-white transition-colors"
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <Link
                href={ROUTES.LOGIN}
                onClick={onClose}
                className="rounded-lg px-3 py-3 text-[15px] font-medium text-cool-gray hover:bg-white/5 hover:text-white transition-colors"
              >
                Log In
              </Link>
              <Link
                href={ROUTES.REGISTER}
                onClick={onClose}
                className="mt-3 block rounded-lg bg-primary px-4 py-3 text-center text-[15px] font-semibold text-white hover:bg-teal-light transition-colors"
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
