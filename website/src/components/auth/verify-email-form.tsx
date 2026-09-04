"use client";

import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { ROUTES } from "@/lib/constants";
import { api, ApiError } from "@/lib/api";
import { roleHome } from "@/lib/redirect";
import { useAuthStore } from "@/store/auth-store";
import { FieldError } from "@/components/ui/field-error";
import { Kicker } from "@/components/ui/kicker";

type Status = "confirming" | "done" | "failed";

export function VerifyEmailForm() {
  const searchParams = useSearchParams();
  const token = searchParams?.get("token") ?? "";
  // Read at render, not captured on confirm: <AuthHydrator> fills the store
  // from /auth/me in parallel with our POST, so a late hydration must be able
  // to correct the CTA rather than freeze it on the signed-out wording.
  const user = useAuthStore((s) => s.user);

  const [status, setStatus] = useState<Status>("confirming");
  const [submitError, setSubmitError] = useState<string | null>(null);

  // The token is one-shot and StrictMode double-invokes effects: the second
  // call would fail INVALID_VERIFICATION_TOKEN and paint an error over a
  // verification that actually succeeded. A ref, not state — a same-tick
  // second run reads the same stale state value.
  const fired = useRef(false);

  useEffect(() => {
    if (!token || fired.current) return;
    fired.current = true;

    // Confirmed on mount rather than behind a button, unlike the sibling
    // change-email flow. That one signs you out of every device, so it owes
    // the user a deliberate click; this revokes nothing and changes no
    // sign-in identity (EmailVerificationService.confirm keeps refresh tokens
    // on purpose). Opening the mail was already the confirming gesture.
    api
      .post<unknown>("/auth/verify-email", { token })
      .then(() => setStatus("done"))
      .catch((err) => {
        const code = err instanceof ApiError ? err.errors[0]?.code : undefined;
        setSubmitError(
          code === "INVALID_VERIFICATION_TOKEN"
            ? "This link has expired or was already used."
            : "We couldn't reach the server to confirm this link.",
        );
        setStatus("failed");
      });
  }, [token]);

  // Signed-in visitors arrive here straight from registering in this same
  // browser, so bouncing them to /login would be a step backwards.
  const continueHref = user ? roleHome(user.role) : ROUTES.LOGIN;
  const continueLabel = user ? "Continue" : "Sign in";

  // Derived, not stored — same reason as ConfirmEmailChangeForm: a client-side
  // nav that drops `?token=` must be re-detected.
  if (!token) {
    return (
      <div className="stagger-children space-y-7">
        <Kicker as="p" size="md">
          Email confirmation
          <span className="ml-2 text-error">&middot; Invalid link</span>
        </Kicker>

        <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
          Link is
          <br />
          <span className="text-fresh">missing.</span>
        </h1>

        <p className="font-sans text-base text-ink-soft">
          This page needs the token from your confirmation email. Open the link
          we sent you rather than typing the address in.
        </p>

        <Link
          href={continueHref}
          className="block w-full text-center bg-fresh hover:bg-fresh-deep text-surface py-4 rounded-full font-display font-bold text-[15px] transition-colors"
        >
          {continueLabel} &rarr;
        </Link>
      </div>
    );
  }

  if (status === "done") {
    return (
      <div className="stagger-children space-y-7">
        <Kicker as="p" size="md">
          Email confirmation
          <span className="ml-2 text-fresh">&middot; Done</span>
        </Kicker>

        <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
          Address
          <br />
          <span className="text-fresh">confirmed.</span>
        </h1>

        <p className="font-sans text-base text-ink-soft">
          We know we can reach you here. Nothing else to do &mdash; your account
          works exactly as it did before.
        </p>

        <Link
          href={continueHref}
          className="block w-full text-center bg-fresh hover:bg-fresh-deep text-surface py-4 rounded-full font-display font-bold text-[15px] transition-colors"
        >
          {continueLabel} &rarr;
        </Link>
      </div>
    );
  }

  if (status === "failed") {
    return (
      <div className="stagger-children space-y-7">
        <Kicker as="p" size="md">
          Email confirmation
          <span className="ml-2 text-error">&middot; Failed</span>
        </Kicker>

        <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
          Link didn&rsquo;t
          <br />
          <span className="text-fresh">work.</span>
        </h1>

        <FieldError message={submitError} id="verify-email-error" />

        {/* Deliberately promises no "request a new link" affordance: the
            website has no consumer for POST /auth/resend-verification, so
            pointing at account settings would send the user hunting for a
            button that isn't there. Confirming is advisory — nothing on the
            account is gated on it — so the honest line is that this costs
            them nothing. */}
        <p className="font-sans text-base text-ink-soft">
          Nothing is blocked by this. Confirming just tells us the address is
          reachable, so you can carry on as normal.
        </p>

        <Link
          href={continueHref}
          className="block w-full text-center bg-fresh hover:bg-fresh-deep text-surface py-4 rounded-full font-display font-bold text-[15px] transition-colors"
        >
          {continueLabel} &rarr;
        </Link>
      </div>
    );
  }

  return (
    <div className="stagger-children space-y-7">
      <Kicker as="p" size="md">
        Email confirmation
      </Kicker>

      <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
        Confirming your
        <br />
        <span className="text-fresh">address&hellip;</span>
      </h1>

      <p className="font-sans text-base text-ink-soft">
        One moment &mdash; we&rsquo;re marking this address as reachable.
      </p>
    </div>
  );
}
