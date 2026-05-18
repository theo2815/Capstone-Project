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
import {
  AuthDivider,
  GoogleButton,
  OAUTH_ENABLED,
} from "@/components/auth/google-button";
import { FieldError } from "@/components/ui/field-error";
import {
  NAME_MAX,
  validateEmail,
  validateName,
  validatePassword,
} from "@/lib/auth-validation";

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

interface FieldErrors {
  name?: string | null;
  email?: string | null;
  password?: string | null;
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
  const [errors, setErrors] = useState<FieldErrors>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  function clearFieldError(field: keyof FieldErrors) {
    if (errors[field] || submitError) {
      setErrors((prev) => ({ ...prev, [field]: null }));
      setSubmitError(null);
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const next: FieldErrors = {
      name: validateName(name),
      email: validateEmail(email),
      password: validatePassword(password),
    };
    setErrors(next);
    setSubmitError(null);
    if (next.name || next.email || next.password) return;

    setIsLoading(true);
    try {
      await register({ name: name.trim(), email: email.trim(), password, role });
      router.push(redirectTo);
    } catch (err) {
      setSubmitError(
        err instanceof ApiError
          ? err.message
          : "Registration failed. Please try again.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} noValidate className="stagger-children space-y-7">
      <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-slate">
        Create account
      </p>

      <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
        Join
        <br />
        <span className="text-fresh">QuickPitik.</span>
      </h1>

      {OAUTH_ENABLED && (
        <div className="space-y-5">
          <GoogleButton disabled={isLoading} />
          <AuthDivider label="or with email" />
        </div>
      )}

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
            onChange={(e) => {
              setName(e.target.value);
              clearFieldError("name");
            }}
            placeholder="Juan dela Cruz"
            autoComplete="name"
            maxLength={NAME_MAX}
            aria-invalid={!!errors.name}
            aria-describedby={errors.name ? "name-error" : undefined}
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors"
          />
          <FieldError message={errors.name} id="name-error" density="tight" />
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
            onChange={(e) => {
              setEmail(e.target.value);
              clearFieldError("email");
            }}
            placeholder="you@example.com"
            autoComplete="email"
            aria-invalid={!!errors.email}
            aria-describedby={errors.email ? "email-error" : undefined}
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors"
          />
          <FieldError message={errors.email} id="email-error" density="tight" />
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
            onChange={(e) => {
              setPassword(e.target.value);
              clearFieldError("password");
            }}
            placeholder="••••••••"
            autoComplete="new-password"
            aria-invalid={!!errors.password}
            aria-describedby={errors.password ? "password-error" : undefined}
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors"
          />
          <FieldError
            message={errors.password}
            id="password-error"
            density="tight"
          />
        </FieldBlock>
      </div>

      <FieldError message={submitError} id="register-submit-error" />

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
