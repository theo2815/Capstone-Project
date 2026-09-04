"use client";

import { useEffect } from "react";
import Link from "next/link";
import { BrandLogo } from "@/components/layout/brand-logo";
import { cn } from "@/lib/utils";
import { BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ROUTES } from "@/lib/constants";

// Root error boundary: an uncaught throw outside the dashboard/upload
// <ErrorBoundary> mounts (a server fetch on /events, anything under /admin)
// used to surface Next's unbranded digest screen. Same shape as not-found.
export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[app] unhandled error", error);
  }, [error]);

  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center px-4">
      <BrandLogo className="mb-8 h-11 w-44" />
      <h1 className="font-hero text-6xl text-line-strong">Oops</h1>
      <h2 className="mt-4 font-display text-xl font-bold text-ink">
        Something went wrong
      </h2>
      <p className="mt-2 text-slate">
        Try again, or head back to the start line.
      </p>
      <div className="mt-6 flex gap-3">
        <button
          type="button"
          onClick={reset}
          className={cn(BTN_PRIMARY, BTN_SIZE.md)}
        >
          Try again
        </button>
        <Link href={ROUTES.HOME} className={cn(BTN_SECONDARY, BTN_SIZE.md)}>
          Back to Home
        </Link>
      </div>
    </div>
  );
}
