"use client";

import { useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { ROUTES } from "@/lib/constants";
import { ApiError } from "@/lib/api";
import { clearTokens } from "@/lib/auth";
import { resetUserScopedStores } from "@/lib/auth-reset";
import { useAuthStore } from "@/store/auth-store";
import { confirmEmailChange } from "@/lib/api-account";
import { FieldError } from "@/components/ui/field-error";
import { Kicker } from "@/components/ui/kicker";

type Status = "idle" | "confirming" | "done";

export function ConfirmEmailChangeForm() {
  const searchParams = useSearchParams();
  const token = searchParams?.get("token") ?? "";
  const clearUser = useAuthStore((s) => s.logout);

  const [status, setStatus] = useState<Status>("idle");
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Deliberately a button rather than a confirm-on-mount effect. The token is
  // one-shot, and an effect would be double-invoked under StrictMode — the
  // second call would fail with INVALID_EMAIL_CHANGE_TOKEN and show an error
  // on top of a change that actually succeeded. A click is also the honest
  // shape for something that signs you out of every device.
  async function handleConfirm() {
    setSubmitError(null);
    setStatus("confirming");
    try {
      await confirmEmailChange(token);
      // The backend revoked every refresh token, so this session is already
      // dead — the access token just hasn't expired yet. Drop it now rather
      // than let the user browse for ≤15 min and get bounced mid-action.
      // Same trio as useAuth.logout, minus the /auth/logout POST: there is no
      // refresh token left to revoke.
      clearTokens();
      resetUserScopedStores();
      clearUser();
      setStatus("done");
    } catch (err) {
      setStatus("idle");
      const code = err instanceof ApiError ? err.errors[0]?.code : undefined;
      setSubmitError(
        code === "INVALID_EMAIL_CHANGE_TOKEN"
          ? "This link has expired or was already used. Request the change again from your account settings."
          : code === "EMAIL_TAKEN"
            ? "That address was registered to another account before you confirmed. Try a different one."
            : "Could not confirm the change. Request a new link from your account settings.",
      );
    }
  }

  // Derived, not stored — same reason as ResetPasswordForm: a client-side nav
  // that drops `?token=` must be re-detected. Excluded from `done` so the
  // success screen survives its own token being spent.
  if (status !== "done" && !token) {
    return (
      <div className="stagger-children space-y-7">
        <Kicker as="p" size="md">
          Email change
          <span className="ml-2 text-error">&middot; Invalid link</span>
        </Kicker>

        <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
          Link is
          <br />
          <span className="text-fresh">missing.</span>
        </h1>

        <p className="font-sans text-base text-ink-soft">
          This page needs the confirmation token from your email. Open the link
          we sent to your new address, or request the change again.
        </p>

        <Link
          href={ROUTES.ACCOUNT}
          className="block w-full text-center bg-fresh hover:bg-fresh-deep text-bone py-4 rounded-full font-mono uppercase tracking-[0.2em] text-sm transition-colors"
        >
          Go to account &rarr;
        </Link>

        <div className="border-t border-line pt-6">
          <p className="text-center">
            <Kicker
              as={Link}
              href={ROUTES.LOGIN}
              className="hover:text-ink transition-colors"
            >
              &larr; Back to sign in
            </Kicker>
          </p>
        </div>
      </div>
    );
  }

  if (status === "done") {
    return (
      <div className="stagger-children space-y-7">
        <Kicker as="p" size="md">
          Email change
          <span className="ml-2 text-fresh">&middot; Done</span>
        </Kicker>

        <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
          Address
          <br />
          <span className="text-fresh">updated.</span>
        </h1>

        <p className="font-sans text-base text-ink-soft">
          This is your sign-in email from now on. We signed you out everywhere
          &mdash; sign back in with the new address.
        </p>

        <Link
          href={ROUTES.LOGIN}
          className="block w-full text-center bg-fresh hover:bg-fresh-deep text-bone py-4 rounded-full font-mono uppercase tracking-[0.2em] text-sm transition-colors"
        >
          Sign in &rarr;
        </Link>
      </div>
    );
  }

  return (
    <div className="stagger-children space-y-7">
      <Kicker as="p" size="md">
        Email change
      </Kicker>

      <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
        Confirm your
        <br />
        <span className="text-fresh">new address.</span>
      </h1>

      <p className="font-sans text-base text-ink-soft">
        You asked to sign in with this address instead. Confirming switches it
        over and signs you out on every device.
      </p>

      <FieldError message={submitError} id="confirm-email-change-error" />

      <button
        type="button"
        onClick={handleConfirm}
        disabled={status === "confirming"}
        className="w-full bg-fresh hover:bg-fresh-deep active:bg-fresh-deep text-bone py-4 rounded-full font-mono uppercase tracking-[0.2em] text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {status === "confirming" ? "Confirming…" : "Confirm new email →"}
      </button>

      <div className="border-t border-line pt-6">
        <p className="text-center">
          <Kicker
            as={Link}
            href={ROUTES.ACCOUNT}
            className="hover:text-ink transition-colors"
          >
            &larr; Not you? Go to account
          </Kicker>
        </p>
      </div>
    </div>
  );
}
