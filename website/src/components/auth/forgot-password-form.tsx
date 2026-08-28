"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import { ROUTES } from "@/lib/constants";
import { api } from "@/lib/api";
import {
  splitApiFieldErrors,
  validateEmail,
  validateNewPassword,
  validateResetCode,
} from "@/lib/auth-validation";
import { FieldError } from "@/components/ui/field-error";

// The whole OTP reset flow lives on this one page: request a code, verify it,
// set the new password. Step idiom follows checkout-modal. The continuation
// token from /auth/verify-reset-otp stays in component state only — it is a
// short-lived credential and must never reach localStorage or the URL.
type Step = "request" | "code" | "password" | "done";

const RESEND_COOLDOWN_SECONDS = 60;

// Backend field name → the local state key that has an input to render under,
// per step. Verify's `email` is deliberately absent from CODE_FIELDS — the
// code step has no email input, so an email failure belongs in the
// submit-level message.
const REQUEST_FIELDS = { email: "email" } as const;
const CODE_FIELDS = { code: "code" } as const;
const PASSWORD_FIELDS = { newPassword: "password" } as const;

export function ForgotPasswordForm() {
  const [step, setStep] = useState<Step>("request");
  const [email, setEmail] = useState("");
  const [submittedEmail, setSubmittedEmail] = useState("");
  const [code, setCode] = useState("");
  const [resetToken, setResetToken] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [emailError, setEmailError] = useState<string | null>(null);
  const [codeError, setCodeError] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [confirmError, setConfirmError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [cooldown, setCooldown] = useState(0);
  // Not `isLoading` — setIsLoading lands on the next render, so a burst of
  // submit events before a paint reads it as false every time. See login-form.
  const submitting = useRef(false);

  useEffect(() => {
    if (cooldown === 0) return;
    const t = setTimeout(() => setCooldown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [cooldown]);

  async function handleRequest(e: FormEvent) {
    e.preventDefault();
    if (submitting.current) return;
    const fieldError = validateEmail(email);
    setEmailError(fieldError);
    setSubmitError(null);
    if (fieldError) return;

    submitting.current = true;
    setIsLoading(true);
    try {
      await api.post("/auth/forgot-password", { email: email.trim() });
      setSubmittedEmail(email.trim());
      setStep("code");
      setCooldown(RESEND_COOLDOWN_SECONDS);
    } catch (err) {
      const { fields, message } = splitApiFieldErrors(err, REQUEST_FIELDS);
      const handled = message !== null || Object.keys(fields).length > 0;
      setEmailError(fields.email ?? null);
      setSubmitError(
        handled ? message : "Could not send a reset code. Please try again.",
      );
    } finally {
      submitting.current = false;
      setIsLoading(false);
    }
  }

  async function handleResend() {
    if (submitting.current || cooldown > 0) return;
    submitting.current = true;
    setIsLoading(true);
    setCode("");
    setCodeError(null);
    setSubmitError(null);
    try {
      await api.post("/auth/forgot-password", { email: submittedEmail });
      setCooldown(RESEND_COOLDOWN_SECONDS);
    } catch (err) {
      const { message } = splitApiFieldErrors(err, {});
      setSubmitError(
        message ?? "Could not resend the code. Please try again.",
      );
    } finally {
      submitting.current = false;
      setIsLoading(false);
    }
  }

  async function handleVerify(e: FormEvent) {
    e.preventDefault();
    if (submitting.current) return;
    const fieldError = validateResetCode(code);
    setCodeError(fieldError);
    setSubmitError(null);
    if (fieldError) return;

    submitting.current = true;
    setIsLoading(true);
    try {
      const res = await api.post<{ resetToken: string }>(
        "/auth/verify-reset-otp",
        { email: submittedEmail, code },
      );
      setResetToken(res.resetToken);
      setCode("");
      setStep("password");
    } catch (err) {
      const { fields, message } = splitApiFieldErrors(err, CODE_FIELDS);
      const handled = message !== null || Object.keys(fields).length > 0;
      setCodeError(fields.code ?? null);
      setSubmitError(
        handled
          ? message
          : "That code didn't work. It may have expired — resend a new one.",
      );
    } finally {
      submitting.current = false;
      setIsLoading(false);
    }
  }

  async function handleSetPassword(e: FormEvent) {
    e.preventDefault();
    if (submitting.current) return;
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

    submitting.current = true;
    setIsLoading(true);
    try {
      await api.post("/auth/reset-password", {
        token: resetToken,
        newPassword: password,
      });
      setResetToken("");
      setStep("done");
    } catch (err) {
      const { fields, message } = splitApiFieldErrors(err, PASSWORD_FIELDS);
      const handled = message !== null || Object.keys(fields).length > 0;
      setPasswordError(fields.password ?? null);
      setSubmitError(
        handled
          ? message
          : "Your verification expired — start over and request a new code.",
      );
    } finally {
      submitting.current = false;
      setIsLoading(false);
    }
  }

  function handleStartOver() {
    setStep("request");
    setEmail("");
    setSubmittedEmail("");
    setCode("");
    setResetToken("");
    setPassword("");
    setConfirmPassword("");
    setEmailError(null);
    setCodeError(null);
    setPasswordError(null);
    setConfirmError(null);
    setSubmitError(null);
    setCooldown(0);
  }

  if (step === "done") {
    return (
      <div className="stagger-children space-y-7">
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
          Reset access
          <span className="ml-2 text-fresh">&middot; Done</span>
        </p>

        <h1 className="font-hero text-5xl md:text-6xl">
          Password
          <br />
          <span className="text-fresh">reset.</span>
        </h1>

        <p className="font-sans text-base text-ink-soft">
          Sign in with your new password to continue to your photos. Other
          devices have been signed out.
        </p>

        <Link
          href={ROUTES.LOGIN}
          className="block w-full text-center bg-fresh hover:bg-fresh-deep text-surface py-4 rounded-full font-display font-bold text-[15px] transition-colors"
        >
          Sign in →
        </Link>
      </div>
    );
  }

  if (step === "password") {
    return (
      <form
        onSubmit={handleSetPassword}
        noValidate
        className="stagger-children space-y-7"
      >
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
          Reset access
          <span className="ml-2 text-fresh">&middot; Verified</span>
        </p>

        <h1 className="font-hero text-5xl md:text-6xl">
          Set a new
          <br />
          <span className="text-fresh">password.</span>
        </h1>

        <p className="font-sans text-base text-ink-soft">
          Pick a strong one. You&apos;ll use it next time you sign in.
        </p>

        <div className="space-y-6 pt-2">
          <div className="flex flex-col gap-2">
            <label htmlFor="password" className="kicker">
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
            <label htmlFor="confirm-password" className="kicker">
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
          className="w-full bg-fresh hover:bg-fresh-deep active:bg-fresh-deep text-surface py-4 rounded-full font-display font-bold text-[15px] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? "Saving…" : "Reset password →"}
        </button>

        <div className="border-t border-line pt-6">
          <p className="text-center">
            <button
              type="button"
              onClick={handleStartOver}
              className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors"
            >
              ← Start over
            </button>
          </p>
        </div>
      </form>
    );
  }

  if (step === "code") {
    return (
      <form
        onSubmit={handleVerify}
        noValidate
        className="stagger-children space-y-7"
      >
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
          Reset access
          <span className="ml-2 text-fresh">&middot; Sent</span>
        </p>

        <h1 className="font-hero text-5xl md:text-6xl">
          Enter your
          <br />
          <span className="text-fresh">code.</span>
        </h1>

        <p className="font-sans text-base text-ink-soft">
          If that address is registered, a 6-digit code is on its way. It
          expires in 10 minutes.
        </p>

        <div className="flex items-center gap-3 border border-line rounded-2xl px-4 py-3.5">
          <EnvelopeGlyph />
          <p className="font-mono text-sm text-ink break-all">
            {submittedEmail}
          </p>
        </div>

        <div className="flex flex-col gap-2">
          <label htmlFor="code" className="kicker">
            Code
          </label>
          <input
            id="code"
            type="text"
            inputMode="numeric"
            autoComplete="one-time-code"
            maxLength={6}
            value={code}
            onChange={(e) => {
              setCode(e.target.value.replace(/\D/g, "").slice(0, 6));
              if (codeError) setCodeError(null);
              if (submitError) setSubmitError(null);
            }}
            placeholder="000000"
            aria-invalid={!!codeError}
            aria-describedby={codeError ? "code-error" : undefined}
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 font-mono text-2xl tracking-[0.35em] tnum text-ink placeholder:text-slate-soft transition-colors"
          />
          <FieldError message={codeError} id="code-error" density="tight" />
        </div>

        <FieldError message={submitError} id="verify-submit-error" />

        <button
          type="submit"
          disabled={isLoading}
          className="w-full bg-fresh hover:bg-fresh-deep active:bg-fresh-deep text-surface py-4 rounded-full font-display font-bold text-[15px] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? "Verifying…" : "Verify code →"}
        </button>

        <button
          type="button"
          onClick={handleResend}
          disabled={cooldown > 0 || isLoading}
          className="w-full border border-line hover:border-ink text-ink py-4 rounded-full font-display font-semibold text-[15px] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {cooldown > 0 ? (
            <span className="tnum">Resend code in {cooldown}s</span>
          ) : (
            "Resend code"
          )}
        </button>

        <div className="border-t border-line pt-6">
          <p className="text-center">
            <button
              type="button"
              onClick={handleStartOver}
              className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors"
            >
              ← Use a different email
            </button>
          </p>
        </div>
      </form>
    );
  }

  return (
    <form
      onSubmit={handleRequest}
      noValidate
      className="stagger-children space-y-7"
    >
      <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
        Reset access
      </p>

      <h1 className="font-hero text-5xl md:text-6xl">
        Forgot
        <br />
        <span className="text-fresh">your password?</span>
      </h1>

      <p className="font-sans text-base text-ink-soft">
        We&apos;ll email you a 6-digit code.
      </p>

      <div className="space-y-6 pt-2">
        <div className="flex flex-col gap-2">
          <label htmlFor="email" className="kicker">
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
        className="w-full bg-fresh hover:bg-fresh-deep active:bg-fresh-deep text-surface py-4 rounded-full font-display font-bold text-[15px] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? "Sending…" : "Send code →"}
      </button>

      <div className="border-t border-line pt-6">
        <p className="text-center">
          <Link
            href={ROUTES.LOGIN}
            className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors"
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
