"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/hooks/use-auth";
import { ROUTES } from "@/lib/constants";
import { ApiError } from "@/lib/api";
import { useRedirectTarget } from "@/lib/redirect";
import { AuthDivider, GoogleButton } from "@/components/auth/google-button";

export function LoginForm() {
  const router = useRouter();
  const redirectTo = useRedirectTarget(ROUTES.EVENTS);
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      await login({ email, password });
      router.push(redirectTo);
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Login failed. Please try again.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="stagger-children space-y-7">
      <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-slate">
        Log in
      </p>

      <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
        Welcome
        <br />
        <span className="text-fresh">back.</span>
      </h1>

      <p className="font-sans text-base text-ink-soft">
        Continue to your photos.
      </p>

      <div className="space-y-5 pt-2">
        <GoogleButton disabled={isLoading} />
        <AuthDivider label="or with email" />
      </div>

      <div className="space-y-6 pt-2">
        <FieldBlock>
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
            aria-invalid={!!error}
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors"
          />
        </FieldBlock>

        <FieldBlock>
          <label
            htmlFor="password"
            className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate"
          >
            Password
          </label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Your password"
            autoComplete="current-password"
            required
            aria-invalid={!!error}
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors"
          />
        </FieldBlock>
      </div>

      {error && (
        <p
          role="alert"
          className="font-mono uppercase tracking-[0.15em] text-[11px] text-error"
        >
          {error}
        </p>
      )}

      <div className="space-y-4 pt-2">
        <button
          type="submit"
          disabled={isLoading}
          className="w-full bg-fresh hover:bg-fresh-deep active:bg-fresh-deep text-bone py-4 rounded-full font-mono uppercase tracking-[0.2em] text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? "Logging in…" : "Log in →"}
        </button>

        <div className="text-center">
          <Link
            href={ROUTES.FORGOT_PASSWORD}
            className="font-mono uppercase tracking-[0.2em] text-[10px] text-slate hover:text-ink transition-colors"
          >
            Forgot password?
          </Link>
        </div>
      </div>

      <div className="border-t border-line pt-6">
        <p className="text-center font-mono uppercase tracking-[0.2em] text-[10px] text-slate">
          New here?{" "}
          <Link
            href={
              redirectTo === ROUTES.EVENTS
                ? ROUTES.REGISTER
                : `${ROUTES.REGISTER}?redirect=${encodeURIComponent(redirectTo)}`
            }
            className="text-ink hover:text-fresh transition-colors"
          >
            Create account →
          </Link>
        </p>
      </div>
    </form>
  );
}

function FieldBlock({ children }: { children: React.ReactNode }) {
  return <div className="flex flex-col gap-2">{children}</div>;
}
