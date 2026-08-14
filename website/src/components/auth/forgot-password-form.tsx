"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import { ROUTES } from "@/lib/constants";
import { api } from "@/lib/api";
import { splitApiFieldErrors, validateEmail } from "@/lib/auth-validation";
import { FieldError } from "@/components/ui/field-error";

type Status = "request" | "sent";

// Backend field name → the local state key that has an input to render under.
const BE_FIELDS = { email: "email" } as const;

export function ForgotPasswordForm() {
  const [status, setStatus] = useState<Status>("request");
  const [email, setEmail] = useState("");
  const [submittedEmail, setSubmittedEmail] = useState("");
  const [emailError, setEmailError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const fieldError = validateEmail(email);
    setEmailError(fieldError);
    setSubmitError(null);
    if (fieldError) return;

    setIsLoading(true);
    try {
      await api.post("/auth/forgot-password", { email: email.trim() });
      setSubmittedEmail(email.trim());
      setStatus("sent");
    } catch (err) {
      const { fields, message } = splitApiFieldErrors(err, BE_FIELDS);
      const handled = message !== null || Object.keys(fields).length > 0;
      setEmailError(fields.email ?? null);
      setSubmitError(
        handled ? message : "Could not send reset link. Please try again.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  function handleReset() {
    setEmail("");
    setSubmittedEmail("");
    setStatus("request");
    setEmailError(null);
    setSubmitError(null);
  }

  if (status === "sent") {
    return (
      <div className="stagger-children space-y-7">
        <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-slate">
          Reset access
          <span className="ml-2 text-fresh">&middot; Sent</span>
        </p>

        <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
          Check your
          <br />
          <span className="text-fresh">inbox.</span>
        </h1>

        <p className="font-sans text-base text-ink-soft">
          We sent a reset link to:
        </p>

        <div className="flex items-center gap-3 border border-line rounded-2xl px-4 py-3.5">
          <EnvelopeGlyph />
          <p className="font-mono text-sm text-ink break-all">
            {submittedEmail}
          </p>
        </div>

        <p className="font-sans text-sm text-slate">
          Didn&apos;t arrive? Check spam, or try a different address.
        </p>

        <button
          type="button"
          onClick={handleReset}
          className="w-full border border-line hover:border-ink text-ink py-4 rounded-full font-mono uppercase tracking-[0.2em] text-sm transition-colors"
        >
          Try another email →
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
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} noValidate className="stagger-children space-y-7">
      <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-slate">
        Reset access
      </p>

      <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
        Forgot
        <br />
        <span className="text-fresh">your password?</span>
      </h1>

      <p className="font-sans text-base text-ink-soft">
        We&apos;ll send a reset link to your inbox.
      </p>

      <div className="space-y-6 pt-2">
        <div className="flex flex-col gap-2">
          <label
            htmlFor="email"
            className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate"
          >
            Email
          </label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(e) => {
              setEmail(e.target.value);
              if (emailError) setEmailError(null);
              if (submitError) setSubmitError(null);
            }}
            placeholder="you@example.com"
            autoComplete="email"
            aria-invalid={!!emailError}
            aria-describedby={emailError ? "email-error" : undefined}
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors"
          />
          <FieldError message={emailError} id="email-error" density="tight" />
        </div>
      </div>

      <FieldError message={submitError} id="forgot-submit-error" />

      <button
        type="submit"
        disabled={isLoading}
        className="w-full bg-fresh hover:bg-fresh-deep active:bg-fresh-deep text-bone py-4 rounded-full font-mono uppercase tracking-[0.2em] text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? "Sending…" : "Send reset link →"}
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

function EnvelopeGlyph() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      fill="none"
      className="size-5 shrink-0 text-fresh"
    >
      <rect
        x="3"
        y="6"
        width="18"
        height="13"
        rx="1.5"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <path
        d="M3.5 7L12 13L20.5 7"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M16.5 3.5L19 6"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}
