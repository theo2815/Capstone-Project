"use client";

import { useEffect, useState, type SVGProps } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/hooks/use-auth";
import { ApiError } from "@/lib/api";
import { ROUTES } from "@/lib/constants";
import { roleHome } from "@/lib/redirect";
import { cn } from "@/lib/utils";
import type { Role } from "@/types/user";

export function OnboardingForm() {
  const router = useRouter();
  const { pendingOAuth, completeOnboarding, cancelOnboarding } = useAuth();
  const [role, setRole] = useState<Role | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!pendingOAuth) {
      router.replace(ROUTES.LOGIN);
    }
  }, [pendingOAuth, router]);

  if (!pendingOAuth) return null;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!role || isSubmitting) return;
    setIsSubmitting(true);
    setError(null);
    try {
      await completeOnboarding(role);
      router.replace(roleHome(role));
    } catch (err) {
      // The parked Google token lives ~1h. If it expired while this page sat
      // open, the pick cannot complete — restart the flow from /login, where
      // one more button click mints a fresh token.
      if (
        err instanceof ApiError &&
        err.errors.some((issue) => issue.code === "INVALID_GOOGLE_TOKEN")
      ) {
        cancelOnboarding();
        router.replace(ROUTES.LOGIN);
        return;
      }
      setError(
        err instanceof ApiError
          ? err.message
          : "Something went wrong. Try again.",
      );
      setIsSubmitting(false);
    }
  }

  function handleSignOut() {
    cancelOnboarding();
    router.replace(ROUTES.LOGIN);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="grid gap-10 md:grid-cols-2 md:gap-12 lg:gap-20"
    >
      <div className="stagger-children space-y-8">
        <p className="font-mono uppercase tracking-[0.14em] text-[11px] text-slate">
          Welcome
        </p>

        <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
          Pick your
          <br />
          <span className="text-fresh">path.</span>
        </h1>

        <div className="space-y-2">
          <p className="font-sans text-base text-ink-soft">
            How will you use QuickPitik?
          </p>
          <p className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate">
            This choice is permanent.
          </p>
        </div>

        <div className="flex items-center justify-between gap-3 border-y border-line py-3">
          <div className="flex items-center gap-3 min-w-0">
            <span className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate-soft shrink-0">
              Signed in
            </span>
            <span className="font-sans text-sm text-ink truncate">
              {pendingOAuth.email}
            </span>
          </div>
          <button
            type="button"
            onClick={handleSignOut}
            className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate hover:text-ink transition-colors shrink-0"
          >
            Not you?
          </button>
        </div>
      </div>

      <div className="stagger-children space-y-5">
        <div
          role="radiogroup"
          aria-label="Account type"
          className="grid gap-3"
        >
          <RoleCard
            value="RUNNER"
            icon={RunnerIcon}
            label="Runner"
            headline="Find your photos"
            body="Search any Cebu race by selfie or bib number, save events, and buy your originals."
            active={role === "RUNNER"}
            onSelect={() => setRole("RUNNER")}
          />
          <RoleCard
            value="PHOTOGRAPHER"
            icon={CameraIcon}
            label="Photographer"
            headline="Sell your photos"
            body="Tether your camera, upload in real time, and sell directly to runners after every event."
            tag="+ Runner access included"
            active={role === "PHOTOGRAPHER"}
            onSelect={() => setRole("PHOTOGRAPHER")}
          />
        </div>

        {error && (
          <p role="alert" className="font-sans text-sm text-error">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={!role || isSubmitting}
          className="w-full bg-fresh hover:bg-fresh-deep active:bg-fresh-deep text-surface py-4 rounded-full font-display font-bold text-[15px] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isSubmitting ? "Setting up…" : "Continue →"}
        </button>
      </div>
    </form>
  );
}

interface RoleCardProps {
  value: Role;
  icon: (props: SVGProps<SVGSVGElement>) => React.ReactElement;
  label: string;
  headline: string;
  body: string;
  tag?: string;
  active: boolean;
  onSelect: () => void;
}

function RoleCard({
  icon: Icon,
  label,
  headline,
  body,
  tag,
  active,
  onSelect,
}: RoleCardProps) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      onClick={onSelect}
      className={cn(
        "group relative flex h-full flex-col text-left rounded-2xl border p-5 transition-all duration-200",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
        active
          ? "border-ink bg-bone-deep"
          : "border-line bg-bone hover:border-slate-soft hover:-translate-y-0.5",
      )}
    >
      <span
        className={cn(
          "absolute top-4 right-4 size-2 rounded-full transition-colors",
          active ? "bg-fresh" : "bg-line",
        )}
        aria-hidden="true"
      />

      <Icon
        aria-hidden="true"
        className={cn(
          "size-12 shrink-0 transition-colors",
          active ? "text-ink" : "text-slate",
        )}
      />

      <p className="mt-5 font-mono uppercase tracking-[0.14em] text-[10px] text-slate">
        {label}
      </p>

      <p className="mt-2 font-display text-2xl font-medium text-ink leading-tight">
        {headline}
      </p>

      <p className="mt-3 font-sans text-sm text-ink-soft leading-relaxed">
        {body}
      </p>

      {tag && (
        <p className="mt-auto pt-5 font-mono uppercase tracking-[0.14em] text-[10px] text-fresh-deep">
          {tag}
        </p>
      )}
    </button>
  );
}

function RunnerIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 80 80" fill="none" {...props}>
      <path
        d="M6 36 L22 36"
        className="stroke-fresh"
        strokeWidth="3"
        strokeLinecap="round"
      />
      <path
        d="M10 46 L24 46"
        className="stroke-fresh"
        strokeWidth="3"
        strokeLinecap="round"
        opacity="0.5"
      />
      <circle cx="50" cy="14" r="6" fill="currentColor" />
      <path
        d="M50 22 L46 40"
        stroke="currentColor"
        strokeWidth="3.5"
        strokeLinecap="round"
      />
      <path
        d="M48 26 L62 22"
        stroke="currentColor"
        strokeWidth="3.5"
        strokeLinecap="round"
      />
      <path
        d="M48 28 L36 32"
        stroke="currentColor"
        strokeWidth="3.5"
        strokeLinecap="round"
      />
      <path
        d="M46 40 L58 50 L52 64"
        stroke="currentColor"
        strokeWidth="3.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      <path
        d="M46 40 L34 56 L26 64"
        stroke="currentColor"
        strokeWidth="3.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}

function CameraIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 36 36"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <rect x="5" y="11" width="26" height="17" rx="2.5" />
      <path d="M12 11 L 14 8 L 22 8 L 24 11" />
      <circle cx="18" cy="20" r="5" />
      <circle cx="18" cy="20" r="2.2" />
      <circle cx="26" cy="14.5" r="1" fill="var(--fresh)" stroke="none" />
    </svg>
  );
}
