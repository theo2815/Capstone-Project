import Link from "next/link";
import { ROUTES } from "@/lib/constants";
import { Camera } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-border bg-charcoal">
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <Link href={ROUTES.HOME} className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
                <Camera className="h-4 w-4 text-white" />
              </div>
              <span className="text-lg font-bold text-white">Quick Pitik</span>
            </Link>
            <p className="mt-3 text-sm text-cool-gray">
              AI-powered marathon photography platform. Find your race photos
              in seconds.
            </p>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-white">Events</h3>
            <ul className="mt-3 space-y-2">
              <li>
                <Link
                  href={ROUTES.EVENTS}
                  className="text-sm text-cool-gray hover:text-white transition-colors"
                >
                  Browse Events
                </Link>
              </li>
              <li>
                <Link
                  href="/#how-it-works"
                  className="text-sm text-cool-gray hover:text-white transition-colors"
                >
                  How It Works
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-white">Account</h3>
            <ul className="mt-3 space-y-2">
              <li>
                <Link
                  href={ROUTES.LOGIN}
                  className="text-sm text-cool-gray hover:text-white transition-colors"
                >
                  Log In
                </Link>
              </li>
              <li>
                <Link
                  href={ROUTES.REGISTER}
                  className="text-sm text-cool-gray hover:text-white transition-colors"
                >
                  Sign Up
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-white">Support</h3>
            <ul className="mt-3 space-y-2">
              <li>
                <span className="text-sm text-cool-gray">
                  support@quickpitik.ph
                </span>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-10 border-t border-charcoal-light pt-8 text-center">
          <p className="text-sm text-cool-gray-dark">
            &copy; {new Date().getFullYear()} Quick Pitik. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}
