import { Suspense } from "react";
import type { Metadata } from "next";
import { AuthShell } from "@/components/auth/auth-shell";
import { ConfirmEmailChangeForm } from "@/components/auth/confirm-email-change-form";
import { Kicker } from "@/components/ui/kicker";
import { ROUTES } from "@/lib/constants";

// `(auth)` is a route group, so this resolves to /confirm-email-change — the
// exact path backend EmailService builds into the confirmation mail. Renaming
// the folder breaks every link already sitting in someone's inbox.
export const metadata: Metadata = {
  title: "Confirm your new email | QuickPitik",
};

export default function ConfirmEmailChangePage() {
  return (
    <AuthShell rightLink={{ label: "Sign in", href: ROUTES.LOGIN }}>
      <Suspense fallback={<ConfirmEmailChangeSkeleton />}>
        <ConfirmEmailChangeForm />
      </Suspense>
    </AuthShell>
  );
}

function ConfirmEmailChangeSkeleton() {
  return (
    <div className="space-y-7" aria-hidden="true">
      <Kicker as="p" size="md">
        Email change
      </Kicker>
      <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
        Confirm your
        <br />
        <span className="text-fresh">new address.</span>
      </h1>
      <div className="h-12 border-b border-line" />
    </div>
  );
}
