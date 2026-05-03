"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import { ROUTES } from "@/lib/constants";

type Status = "request" | "sent";

export function ForgotPasswordForm() {
  const [status, setStatus] = useState<Status>("request");
  const [email, setEmail] = useState("");
  const [submittedEmail, setSubmittedEmail] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    // TODO: api.post('/auth/forgot-password', { email }) once backend is wired
    setSubmittedEmail(email);
    setStatus("sent");
  }

  function handleReset() {
    setEmail("");
    setSubmittedEmail("");
    setStatus("request");
  }

  if (status === "sent") {
    return (
      <div className="stagger-children space-y-7">
        <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-slate">
          Check your inbox
        </p>

        <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
          <span className="text-fresh">Sent.</span>
        </h1>

        <p className="font-sans text-base text-ink-soft">
          We sent a reset link to:
        </p>

        <p className="border border-line rounded-md px-4 py-3 font-mono text-sm text-ink break-all">
          {submittedEmail}
        </p>

        <p className="font-sans text-sm text-slate">
          Didn&apos;t get it? Check spam, or try a different address.
        </p>

        <button
          type="button"
          onClick={handleReset}
          className="w-full border border-line hover:border-ink text-ink py-4 rounded-full font-mono uppercase tracking-[0.2em] text-sm transition-colors"
        >
          Try another email →
        </button>

        <div className="border-t border-line pt-6">
          <p className="text-center font-mono uppercase tracking-[0.2em] text-[10px] text-slate">
            <Link
              href={ROUTES.LOGIN}
              className="text-ink hover:text-fresh transition-colors"
            >
              ← Back to sign in
            </Link>
          </p>
        </div>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="stagger-children space-y-7">
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
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            autoComplete="email"
            required
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors"
          />
        </div>
      </div>

      <button
        type="submit"
        className="w-full bg-fresh hover:bg-fresh-deep active:bg-fresh-deep text-bone py-4 rounded-full font-mono uppercase tracking-[0.2em] text-sm transition-colors"
      >
        Send reset link →
      </button>

      <div className="border-t border-line pt-6">
        <p className="text-center font-mono uppercase tracking-[0.2em] text-[10px] text-slate">
          <Link
            href={ROUTES.LOGIN}
            className="text-ink hover:text-fresh transition-colors"
          >
            ← Back to sign in
          </Link>
        </p>
      </div>
    </form>
  );
}
