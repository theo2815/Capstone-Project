import Link from "next/link";
import { ROUTES } from "@/lib/constants";
import { Camera } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-white/[0.06] bg-charcoal-deep">
      <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-10 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <Link href={ROUTES.HOME} className="group inline-flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary transition-transform duration-200 group-hover:scale-105">
                <Camera className="h-4 w-4 text-white" />
              </div>
              <span className="text-lg font-bold text-white">Quick Pitik</span>
            </Link>
            <p className="mt-4 max-w-xs text-sm leading-relaxed text-cool-gray/50">
              AI-powered marathon photography platform. Find your race photos
              in seconds using face recognition and bib number search.
            </p>
          </div>

          <div>
            <h3 className="text-xs font-semibold uppercase tracking-[0.15em] text-cool-gray-dark">
              Events
            </h3>
            <ul className="mt-4 space-y-3">
              <li>
                <Link
                  href={ROUTES.EVENTS}
                  className="text-sm text-cool-gray/60 hover:text-white transition-colors duration-200"
                >
                  Browse Events
                </Link>
              </li>
              <li>
                <Link
                  href="/#how-it-works"
                  className="text-sm text-cool-gray/60 hover:text-white transition-colors duration-200"
                >
                  How It Works
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xs font-semibold uppercase tracking-[0.15em] text-cool-gray-dark">
              Account
            </h3>
            <ul className="mt-4 space-y-3">
              <li>
                <Link
                  href={ROUTES.LOGIN}
                  className="text-sm text-cool-gray/60 hover:text-white transition-colors duration-200"
                >
                  Log In
                </Link>
              </li>
              <li>
                <Link
                  href={ROUTES.REGISTER}
                  className="text-sm text-cool-gray/60 hover:text-white transition-colors duration-200"
                >
                  Sign Up
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xs font-semibold uppercase tracking-[0.15em] text-cool-gray-dark">
              Support
            </h3>
            <ul className="mt-4 space-y-3">
              <li>
                <span className="text-sm text-cool-gray/60">
                  support@quickpitik.ph
                </span>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-12 border-t border-white/[0.06] pt-8 flex flex-col items-center gap-2 sm:flex-row sm:justify-between">
          <p className="text-xs text-cool-gray-dark">
            &copy; {new Date().getFullYear()} Quick Pitik. All rights reserved.
          </p>
          <p className="text-xs text-cool-gray-dark/50">
            Cebu, Philippines
          </p>
        </div>
      </div>
    </footer>
  );
}
