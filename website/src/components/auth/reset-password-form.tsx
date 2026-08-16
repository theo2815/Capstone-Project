"use client";

import { useState, type FormEvent } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { ROUTES } from "@/lib/constants";
import { api } from "@/lib/api";
import { splitApiFieldErrors, validateNewPassword } from "@/lib/auth-validation";
import { FieldError } from "@/components/ui/field-error";

type Status = "request" | "sent";

// Backend field name → the local state key that has an input to render under.
// `token` is deliberately absent: it comes from the URL and has no input, so a
// token failure belongs in the submit-level message.
const BE_FIELDS = { newPassword: "password" } as const;

export function ResetPasswordForm() {
  const searchParams = useSearchParams();
  const token = searchParams?.get("token") ?? "";

  const [status, setStatus] = useState<Status>("request");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [confirmError, setConfirmError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const passErr = validateNewPassword(password);
    const confirmErr = !confirmPassword
      ? "Please confirm your new password."
      : password !== confirmPassword
        ? "Passwords don't match."
        : null;
    setPasswordError(passErr);
    setConfirmError(confirmErr);
    setSubmitError(null);
    if (passErr || confirmErr) return;

    setIsLoading(true);
    try {
      await api.post("/auth/reset-password", {
        token,
        newPassword: password,
      });
      setStatus("sent");
    } catch (err) {
      const { fields, message } = splitApiFieldErrors(err, BE_FIELDS);
      const handled = message !== null || Object.keys(fields).length > 0;
      setPasswordError(fields.password ?? null);
      setSubmitError(
        handled
          ? message
          : "Could not reset your password. The link may have expired — request a new one.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  // Derived, not stored. `useSearchParams()` re-renders on URL change, so a
  // client-side nav that drops `?token=` is re-detected — the old `useState`
  // initializer latched the answer at first mount and never revisited it.
  // Excluded from `sent` so the success screen survives its own token being
  // spent.
  if (status !== "sent" && !token) {
    return (
      <div className="stagger-children space-y-7">
        <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-slate">
          Reset access
          <span className="ml-2 text-error">&middot; Invalid link</span>
        </p>

        <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
          Link is
          <br />
          <span className="text-fresh">missing.</span>
        </h1>

        <p className="font-sans text-base text-ink-soft">
          This page needs a reset token. Request a fresh reset link and follow
          it from your email.
        </p>

        <Link
          href={ROUTES.FORGOT_PASSWORD}
          className="block w-full text-center bg-fresh hover:bg-fresh-deep text-bone py-4 rounded-full font-mono uppercase tracking-[0.2em] text-sm transition-colors"
        >
          Request new link →
        </Link>

        <div className="border-t border-line pt-6">
          <p className="text-center">
            <Link
              href={ROUTES.LOGIN}
              className="font-mono uppercase tracking-[0.2em] text-[10px] text-slate hover:text-ink transition-colors"
            >
              ← Back to sign in
            </Link>
          </p>
        </div>
      </div>
    );
  }

  if (status === "sent") {
    return (
      <div className="stagger-children space-y-7">
        <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-slate">
          Reset access
          <span className="ml-2 text-fresh">&middot; Done</span>
        </p>

        <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
          Password
          <br />
          <span className="text-fresh">reset.</span>
        </h1>

        <p className="font-sans text-base text-ink-soft">
          Sign in with your new password to continue to your photos.
        </p>

        <Link
          href={ROUTES.LOGIN}
          className="block w-full text-center bg-fresh hover:bg-fresh-deep text-bone py-4 rounded-full font-mono uppercase tracking-[0.2em] text-sm transition-colors"
        >
          Sign in →
        </Link>
      </div>
    );
  }

  return (
    <form
      onSubmit={handleSubmit}
      noValidate
      className="stagger-children space-y-7"
    >
      <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-slate">
        Reset access
      </p>

      <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
        Set a new
        <br />
        <span className="text-fresh">password.</span>
      </h1>

      <p className="font-sans text-base text-ink-soft">
        Pick a strong one. You&apos;ll use it next time you sign in.
      </p>

      <div className="space-y-6 pt-2">
        <div className="flex flex-col gap-2">
          <label
            htmlFor="password"
            className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate"
          >
            New password
            <span className="ml-2 text-slate-soft normal-case tracking-normal">
              min. 8 characters
            </span>
          </label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(e) => {
              setPassword(e.target.value);
              if (passwordError) setPasswordError(null);
              if (submitError) setSubmitError(null);
            }}
            placeholder="••••••••"
            autoComplete="new-password"
            aria-invalid={!!passwordError}
            aria-describedby={passwordError ? "password-error" : undefined}
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors"
          />
          <FieldError
            message={passwordError}
            id="password-error"
            density="tight"
          />
        </div>

        <div className="flex flex-col gap-2">
          <label
            htmlFor="confirm-password"
            className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate"
          >
            Confirm new password
          </label>
          <input
            id="confirm-password"
            type="password"
            value={confirmPassword}
            onChange={(e) => {
              setConfirmPassword(e.target.value);
              if (confirmError) setConfirmError(null);
              if (submitError) setSubmitError(null);
            }}
            placeholder="••••••••"
            autoComplete="new-password"
            aria-invalid={!!confirmError}
            aria-describedby={confirmError ? "confirm-error" : undefined}
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors"
          />
          <FieldError
            message={confirmError}
            id="confirm-error"
            density="tight"
          />
        </div>
      </div>

      <FieldError message={submitError} id="reset-submit-error" />

      <button
        type="submit"
        disabled={isLoading}
        className="w-full bg-fresh hover:bg-fresh-deep active:bg-fresh-deep text-bone py-4 rounded-full font-mono uppercase tracking-[0.2em] text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? "Saving…" : "Reset password →"}
      </button>

      <div className="border-t border-line pt-6">
        <p className="text-center">
          <Link
            href={ROUTES.LOGIN}
            className="font-mono uppercase tracking-[0.2em] text-[10px] text-slate hover:text-ink transition-colors"
          >
            ← Back to sign in
          </Link>
        </p>
      </div>
    </form>
  );
}
