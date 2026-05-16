"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { ProtectedRoute } from "@/components/auth/protected-route";
import { SiteHeader } from "@/components/layout/site-header";
import {
  IdentityRail,
  Slab,
  type JumpSection,
} from "@/components/profile-shell";
import { Kicker } from "@/components/ui/kicker";
import { AvatarSlab } from "@/components/account/avatar-slab";
import { useAuth } from "@/hooks/use-auth";
import { useAuthStore } from "@/store/auth-store";
import { useToast } from "@/hooks/use-toast";
import { FieldError } from "@/components/ui/field-error";
import { ApiError } from "@/lib/api";
import { updateProfileName, changePassword } from "@/lib/api-account";
import {
  NAME_MAX,
  PASSWORD_MIN,
  validateName,
  validatePassword,
} from "@/lib/auth-validation";
import { ROUTES } from "@/lib/constants";
import type { User } from "@/types/user";

const RUNNER_JUMP_SECTIONS: ReadonlyArray<JumpSection> = [
  { id: "name", label: "Name" },
  { id: "picture", label: "Picture" },
  { id: "email", label: "Email" },
  { id: "password", label: "Password" },
  { id: "danger", label: "Danger" },
];

// Photographers manage their profile picture in /dashboard/settings (Slab 01)
// alongside their cover, brand, watermark, etc. — keeping a duplicate Picture
// slab here would split the photographer's identity-editing surface across two
// pages. Runner accounts are unaffected.
const PHOTOGRAPHER_JUMP_SECTIONS: ReadonlyArray<JumpSection> = [
  { id: "name", label: "Name" },
  { id: "email", label: "Email" },
  { id: "password", label: "Password" },
  { id: "danger", label: "Danger" },
];

export default function AccountPage() {
  return (
    <ProtectedRoute>
      <AccountBody />
    </ProtectedRoute>
  );
}

function AccountBody() {
  const { user } = useAuth();
  if (!user) return null;

  const isPhotographer = user.role === "PHOTOGRAPHER";
  const jumpSections = isPhotographer
    ? PHOTOGRAPHER_JUMP_SECTIONS
    : RUNNER_JUMP_SECTIONS;

  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col scroll-smooth">
      <SiteHeader />
      <div className="flex-1 max-w-7xl mx-auto w-full px-6 md:px-10 flex flex-col">
        <div className="md:grid md:grid-cols-[15rem_1fr] md:gap-12 lg:gap-20 flex-1">
          <IdentityRail
            user={user}
            kicker="Profile · Settings"
            headline={<span className="text-fresh">Account.</span>}
            subline={
              <span className="block max-w-xs">
                Edit how you appear and keep your account secure.
              </span>
            }
            jumpSections={jumpSections}
            currentPath={ROUTES.ACCOUNT}
          />
          <div className="stagger-children min-w-0 pb-8 md:pb-20 md:border-l md:border-line md:-ml-6 lg:-ml-10 md:pl-6 lg:pl-10">
            <NameSlab user={user} number="01" />
            {!isPhotographer && <PictureSlab number="02" />}
            <EmailSlab
              user={user}
              number={isPhotographer ? "02" : "03"}
            />
            <PasswordSlab number={isPhotographer ? "03" : "04"} />
            <DangerSlab number={isPhotographer ? "04" : "05"} />
            {process.env.NODE_ENV === "development" && (
              <DevRoleSlab
                number={isPhotographer ? "05" : "06"}
                user={user}
              />
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

function NameSlab({ user, number }: { user: User; number: string }) {
  const setUser = useAuthStore((s) => s.setUser);
  const { showToast } = useToast();
  const [name, setName] = useState(user.name);
  const [nameError, setNameError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  const dirty = name.trim() !== user.name && name.trim().length > 0;

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!dirty) return;
    const fieldError = validateName(name);
    setNameError(fieldError);
    setSubmitError(null);
    if (fieldError) return;

    setIsSaving(true);
    try {
      const updated = await updateProfileName(name.trim());
      setUser(updated);
      showToast({ kind: "success", message: "Name updated." });
    } catch (err) {
      if (err instanceof ApiError) {
        const apiMessage = err.errors[0]?.message;
        if (err.errors[0]?.field === "name") {
          setNameError(apiMessage ?? "Please check the name.");
        } else {
          setSubmitError(apiMessage ?? "Could not save. Try again.");
        }
      } else {
        setSubmitError("Could not save. Try again.");
      }
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <Slab id="name" number={number} title="Name">
      <form onSubmit={handleSubmit} noValidate className="space-y-6">
        <FieldShell id="name" label="Full name">
          <input
            id="name"
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              if (nameError) setNameError(null);
              if (submitError) setSubmitError(null);
            }}
            placeholder="Juan dela Cruz"
            autoComplete="name"
            maxLength={NAME_MAX}
            aria-invalid={!!nameError}
            aria-describedby={nameError ? "account-name-error" : undefined}
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-4 text-lg text-ink placeholder:text-slate-soft transition-colors"
          />
          <FieldError
            message={nameError}
            id="account-name-error"
            density="tight"
          />
        </FieldShell>

        <FieldError message={submitError} id="account-name-submit-error" />

        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <button
            type="submit"
            disabled={!dirty || isSaving}
            className="font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-6 rounded-full transition-colors disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center gap-2"
          >
            {isSaving ? "Saving…" : "Save name"}
            {!isSaving && <span aria-hidden="true">→</span>}
          </button>
          {dirty && !isSaving && (
            <button
              type="button"
              onClick={() => {
                setName(user.name);
                setNameError(null);
                setSubmitError(null);
              }}
              className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink hover:text-ink transition-colors"
            >
              Reset
            </button>
          )}
        </div>
      </form>
    </Slab>
  );
}

function PictureSlab({ number }: { number: string }) {
  return (
    <Slab id="picture" number={number} title="Picture" caption="Shown next to your name">
      <AvatarSlab />
    </Slab>
  );
}

function EmailSlab({ user, number }: { user: User; number: string }) {
  return (
    <Slab id="email" number={number} title="Email" caption="Used to sign in">
      <div className="space-y-5">
        <div className="border border-line rounded-2xl px-6 py-5 bg-bone-deep/40">
          <Kicker as="p" tone="soft">
            Sign-in email
          </Kicker>
          <p className="font-mono text-xl md:text-2xl text-ink mt-3 break-all">
            {user.email}
          </p>
        </div>
        <p className="font-sans text-sm text-slate max-w-md">
          Email can&apos;t be changed from here. Contact support if you need to
          update the address on your account.
        </p>
      </div>
    </Slab>
  );
}

interface PasswordErrors {
  current?: string | null;
  next?: string | null;
  confirm?: string | null;
}

function PasswordSlab({ number }: { number: string }) {
  const { showToast } = useToast();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [errors, setErrors] = useState<PasswordErrors>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  function reset() {
    setCurrent("");
    setNext("");
    setConfirm("");
    setErrors({});
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const nextErrors: PasswordErrors = {
      current: current ? null : "Current password is required.",
      next: validatePassword(next),
      confirm: !confirm
        ? "Confirm the new password."
        : next && next !== confirm
          ? "Passwords don't match."
          : null,
    };
    setErrors(nextErrors);
    setSubmitError(null);
    if (nextErrors.current || nextErrors.next || nextErrors.confirm) return;

    setIsSaving(true);
    try {
      await changePassword({ currentPassword: current, newPassword: next });
      reset();
      showToast({ kind: "success", message: "Password updated." });
    } catch (err) {
      if (err instanceof ApiError) {
        const code = err.errors[0]?.code;
        const message = err.errors[0]?.message;
        if (code === "INVALID_CREDENTIALS") {
          setErrors((prev) => ({
            ...prev,
            current: "Current password is incorrect.",
          }));
        } else if (code === "SAME_PASSWORD") {
          setErrors((prev) => ({
            ...prev,
            next: "New password must be different from current.",
          }));
        } else {
          setSubmitError(message ?? "Could not save. Try again.");
        }
      } else {
        setSubmitError("Could not save. Try again.");
      }
    } finally {
      setIsSaving(false);
    }
  }

  function clearFieldError(field: keyof PasswordErrors) {
    if (errors[field] || submitError) {
      setErrors((prev) => ({ ...prev, [field]: null }));
      setSubmitError(null);
    }
  }

  return (
    <Slab
      id="password"
      number={number}
      title="Password"
      caption={`Min. ${PASSWORD_MIN} characters`}
    >
      <form onSubmit={handleSubmit} noValidate className="space-y-6">
        <PasswordField
          id="current-password"
          label="Current password"
          autoComplete="current-password"
          value={current}
          error={errors.current}
          onChange={(v) => {
            setCurrent(v);
            clearFieldError("current");
          }}
        />
        <PasswordField
          id="new-password"
          label="New password"
          autoComplete="new-password"
          value={next}
          error={errors.next}
          onChange={(v) => {
            setNext(v);
            clearFieldError("next");
          }}
        />
        <PasswordField
          id="confirm-password"
          label="Confirm new password"
          autoComplete="new-password"
          value={confirm}
          error={errors.confirm}
          onChange={(v) => {
            setConfirm(v);
            clearFieldError("confirm");
          }}
        />

        <FieldError message={submitError} id="account-password-submit-error" />

        <button
          type="submit"
          disabled={isSaving}
          className="font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-6 rounded-full transition-colors disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center gap-2"
        >
          {isSaving ? "Saving…" : "Change password"}
          {!isSaving && <span aria-hidden="true">→</span>}
        </button>
      </form>
    </Slab>
  );
}

function DevRoleSlab({ user, number }: { user: User; number: string }) {
  const setUser = useAuthStore((s) => s.setUser);
  const { showToast } = useToast();

  function setRole(next: User["role"]) {
    if (next === user.role) return;
    setUser({ ...user, role: next });
    showToast({
      kind: "info",
      message: `Now signed in as ${next.toLowerCase()}.`,
    });
  }

  const roles: ReadonlyArray<{ value: User["role"]; label: string }> = [
    { value: "ADMIN", label: "Admin" },
    { value: "PHOTOGRAPHER", label: "Photographer" },
    { value: "RUNNER", label: "Runner" },
  ];

  return (
    <Slab
      id="dev-role"
      number={number}
      title="Dev"
      caption="Role swap (development only)"
    >
      <div className="border border-dashed border-line rounded-2xl px-6 py-5">
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink">
          Switch roles for testing.
        </p>
        <p className="font-sans text-sm text-slate mt-2 max-w-md">
          Demo helper while the backend is mocked. Picks the role on the
          current session without touching credentials. Won&apos;t render
          in production.
        </p>
        <div className="mt-5 flex flex-wrap items-center gap-3">
          {roles.map((role) => {
            const isActive = role.value === user.role;
            return (
              <button
                key={role.value}
                type="button"
                onClick={() => setRole(role.value)}
                disabled={isActive}
                className={`font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] rounded-full px-5 py-2 border transition-colors ${
                  isActive
                    ? "border-ink text-ink bg-ink/5 cursor-default"
                    : "border-line text-slate hover:text-ink hover:border-ink"
                }`}
              >
                {role.label}
                {isActive && (
                  <span className="ml-2 size-1.5 rounded-full bg-fresh inline-block align-middle" />
                )}
              </button>
            );
          })}
        </div>
      </div>
    </Slab>
  );
}

function DangerSlab({ number }: { number: string }) {
  const router = useRouter();
  const { logout } = useAuth();

  function handleSignOut() {
    logout();
    router.replace(ROUTES.HOME);
  }

  return (
    <Slab id="danger" number={number} title="Danger" caption="Account-wide actions">
      <div className="space-y-6">
        <div className="border border-line rounded-2xl px-6 py-5 bg-bone-deep/40">
          <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink">
            Sign out of QuickPitik.
          </p>
          <p className="font-sans text-sm text-slate mt-2 max-w-md">
            You&apos;ll need to sign in again on this browser to access your
            profile, race log, and orders.
          </p>
          <button
            type="button"
            onClick={handleSignOut}
            className="mt-5 font-sans text-base font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-3 px-6 rounded-full transition-colors inline-flex items-center gap-2"
          >
            Sign out
            <span aria-hidden="true">→</span>
          </button>
        </div>

        <p className="font-sans text-sm text-slate max-w-md">
          Need to delete your account? Contact{" "}
          <a
            href="mailto:support@quickpitik.com"
            className="text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
          >
            support@quickpitik.com
          </a>{" "}
          and we&apos;ll handle it within 7 days.
        </p>
      </div>
    </Slab>
  );
}

function FieldShell({
  id,
  label,
  children,
}: {
  id: string;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={id} className="font-sans text-sm text-slate">
        {label}
      </label>
      {children}
    </div>
  );
}

function PasswordField({
  id,
  label,
  autoComplete,
  value,
  error,
  onChange,
}: {
  id: string;
  label: string;
  autoComplete: string;
  value: string;
  error?: string | null;
  onChange: (v: string) => void;
}) {
  const errorId = `${id}-error`;
  return (
    <FieldShell id={id} label={label}>
      <input
        id={id}
        type="password"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="••••••••"
        autoComplete={autoComplete}
        aria-invalid={!!error}
        aria-describedby={error ? errorId : undefined}
        className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-4 text-lg text-ink placeholder:text-slate-soft transition-colors"
      />
      <FieldError message={error} id={errorId} density="tight" />
    </FieldShell>
  );
}
