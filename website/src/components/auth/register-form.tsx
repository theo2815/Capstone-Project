"use client";

import { useState, type FormEvent } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/hooks/use-auth";
import { ROUTES } from "@/lib/constants";
import { ApiError } from "@/lib/api";
import { isSafeRedirect } from "@/lib/redirect";
import { cn } from "@/lib/utils";
import type { Role } from "@/types/user";
import { AuthDivider, GoogleButton } from "@/components/auth/google-button";

interface RoleOption {
  value: Role;
  label: string;
  sub: string;
}

const ROLE_OPTIONS: ReadonlyArray<RoleOption> = [
  { value: "RUNNER", label: "I run", sub: "Find your photos." },
  { value: "PHOTOGRAPHER", label: "I shoot", sub: "Sell your photos." },
];

function resolveInitialRole(param: string | null): Role {
  return param === "PHOTOGRAPHER" ? "PHOTOGRAPHER" : "RUNNER";
}

function pad2(n: number): string {
  return n.toString().padStart(2, "0");
}

export function RegisterForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { register } = useAuth();

  const rawRedirect = searchParams.get("redirect");
  const redirectTo = isSafeRedirect(rawRedirect) ? rawRedirect : ROUTES.EVENTS;

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<Role>(() =>
    resolveInitialRole(searchParams.get("role")),
  );
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      await register({ name, email, password, role });
      router.push(redirectTo);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Registration failed. Please try again.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="stagger-children space-y-7">
      <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-slate">
        Create account
      </p>

      <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
        Join
        <br />
        <span className="text-fresh">QuickPitik.</span>
      </h1>

      <div className="space-y-5">
        <GoogleButton disabled={isLoading} />
        <AuthDivider label="or with email" />
      </div>

      <div
        role="group"
        aria-label="Account type"
        className="grid grid-cols-2 gap-3"
      >
        {ROLE_OPTIONS.map((option, idx) => {
          const active = role === option.value;
          return (
            <button
              key={option.value}
              type="button"
              aria-pressed={active}
              onClick={() => setRole(option.value)}
              className={cn(
                "group text-left p-4 rounded-2xl border transition-colors duration-200",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
                active
                  ? "border-ink bg-bone-deep"
                  : "border-line bg-bone hover:border-slate-soft",
              )}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-[10px] tracking-[0.15em] text-slate-soft tnum">
                  {pad2(idx + 1)}
                </span>
                <span
                  className={cn(
                    "size-1.5 rounded-full transition-colors",
                    active ? "bg-fresh" : "bg-transparent",
                  )}
                  aria-hidden="true"
                />
              </div>
              <p className="font-display text-lg text-ink leading-tight mt-3">
                {option.label}
              </p>
              <p className="font-sans text-sm text-slate mt-1.5">
                {option.sub}
              </p>
            </button>
          );
        })}
      </div>

      <div className="space-y-6 pt-2">
        <FieldBlock>
          <label
            htmlFor="name"
            className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate"
          >
            Full name
          </label>
          <input
            id="name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Juan dela Cruz"
            autoComplete="name"
            required
            aria-invalid={!!error}
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors"
          />
        </FieldBlock>

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
            <span className="ml-2 text-slate-soft normal-case tracking-normal">
              min. 8 characters
            </span>
          </label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
            autoComplete="new-password"
            minLength={8}
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

      <button
        type="submit"
        disabled={isLoading}
        className="w-full bg-fresh hover:bg-fresh-deep active:bg-fresh-deep text-bone py-4 rounded-full font-mono uppercase tracking-[0.2em] text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? "Creating account…" : "Create account →"}
      </button>

      <div className="border-t border-line pt-6">
        <p className="text-center font-mono uppercase tracking-[0.2em] text-[10px] text-slate">
          Already have an account?{" "}
          <Link
            href={
              redirectTo === ROUTES.EVENTS
                ? ROUTES.LOGIN
                : `${ROUTES.LOGIN}?redirect=${encodeURIComponent(redirectTo)}`
            }
            className="text-ink hover:text-fresh transition-colors"
          >
            Sign in →
          </Link>
        </p>
      </div>
    </form>
  );
}

function FieldBlock({ children }: { children: React.ReactNode }) {
  return <div className="flex flex-col gap-2">{children}</div>;
}
