"use client";

import { useState, type FormEvent } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/hooks/use-auth";
import { ROUTES } from "@/lib/constants";
import { ApiError } from "@/lib/api";
import type { Role } from "@/types/user";

const ROLE_OPTIONS: ReadonlyArray<{ value: Role; label: string }> = [
  { value: "RUNNER", label: "I run" },
  { value: "PHOTOGRAPHER", label: "I shoot" },
];

function resolveInitialRole(param: string | null): Role {
  return param === "PHOTOGRAPHER" ? "PHOTOGRAPHER" : "RUNNER";
}

export function RegisterForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { register } = useAuth();

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
      router.push(ROUTES.HOME);
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

      <div
        role="group"
        aria-label="Account type"
        className="inline-flex border border-line rounded-full p-1 bg-bone-deep/50"
      >
        {ROLE_OPTIONS.map(({ value, label }) => {
          const active = role === value;
          return (
            <button
              key={value}
              type="button"
              aria-pressed={active}
              onClick={() => setRole(value)}
              className={`px-5 py-2 rounded-full font-mono uppercase tracking-[0.25em] text-[10px] transition-colors ${
                active
                  ? "bg-ink text-bone"
                  : "text-slate-soft hover:text-ink"
              }`}
            >
              {label}
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
            href={ROUTES.LOGIN}
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
