"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
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
import { useCartStore } from "@/store/cart-store";
import { useSavedEventsStore } from "@/store/saved-events-store";
import { useToast } from "@/hooks/use-toast";
import { FieldError } from "@/components/ui/field-error";
import { ApiError } from "@/lib/api";
import { resetUserScopedStores } from "@/lib/auth-reset";
import {
  updateProfileName,
  changePassword,
  requestEmailChange,
} from "@/lib/api-account";
import {
  EMAIL_MAX,
  NAME_MAX,
  PASSWORD_MIN,
  validateEmail,
  validateName,
  validateNewPassword,
} from "@/lib/auth-validation";
import { ROUTES } from "@/lib/constants";
import type { User } from "@/types/user";

// "Sign-in email" rather than "Email": it names what the address is FOR, which
// matters now that the slab can change it — this is the credential that
// receives password resets, not a contact field.
const RUNNER_JUMP_SECTIONS: ReadonlyArray<JumpSection> = [
  { id: "name", label: "Name" },
  { id: "picture", label: "Picture" },
  { id: "email", label: "Sign-in email" },
  { id: "password", label: "Password" },
  { id: "danger", label: "Danger" },
];

// Photographers manage their profile picture in /dashboard/settings (Slab 01)
// alongside their cover, brand, watermark, etc. — keeping a duplicate Picture
// slab here would split the photographer's identity-editing surface across two
// pages. Runner accounts are unaffected.
const PHOTOGRAPHER_JUMP_SECTIONS: ReadonlyArray<JumpSection> = [
  { id: "name", label: "Name" },
  { id: "email", label: "Sign-in email" },
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
            className="font-display text-base font-bold bg-fresh hover:bg-fresh-deep text-surface py-3 px-6 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-2"
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

interface EmailErrors {
  next?: string | null;
  password?: string | null;
}

function EmailSlab({ user, number }: { user: User; number: string }) {
  const { showToast } = useToast();
  const [nextEmail, setNextEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errors, setErrors] = useState<EmailErrors>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const nextErrors: EmailErrors = {
      next:
        validateEmail(nextEmail) ??
        (nextEmail.trim().toLowerCase() === user.email.toLowerCase()
          ? "That's already your sign-in email."
          : null),
      password: password ? null : "Current password is required.",
    };
    setErrors(nextErrors);
    setSubmitError(null);
    if (nextErrors.next || nextErrors.password) return;

    setIsSaving(true);
    try {
      const message = await requestEmailChange({
        newEmail: nextEmail.trim(),
        currentPassword: password,
      });
      setNextEmail("");
      setPassword("");
      // The backend's own wording, because nothing has changed yet — a bare
      // "Saved." would claim a swap that only happens once the link in the new
      // inbox is opened. Long duration: the next step is in another app.
      showToast({ kind: "success", message, duration: 9000 });
    } catch (err) {
      if (err instanceof ApiError) {
        const code = err.errors[0]?.code;
        const message = err.errors[0]?.message;
        if (code === "INVALID_CREDENTIALS") {
          setErrors((prev) => ({
            ...prev,
            password: "Current password is incorrect.",
          }));
        } else if (code === "SAME_EMAIL" || code === "EMAIL_TAKEN") {
          setErrors((prev) => ({ ...prev, next: message ?? null }));
        } else {
          setSubmitError(message ?? "Could not send the confirmation. Try again.");
        }
      } else {
        setSubmitError("Could not send the confirmation. Try again.");
      }
    } finally {
      setIsSaving(false);
    }
  }

  function clearFieldError(field: keyof EmailErrors) {
    if (errors[field] || submitError) {
      setErrors((prev) => ({ ...prev, [field]: null }));
      setSubmitError(null);
    }
  }

  return (
    <Slab
      id="email"
      number={number}
      title="Sign-in email"
      caption="Confirmed from the new inbox"
    >
      <div className="space-y-7">
        <div className="border border-line rounded-2xl px-6 py-5 bg-bone-deep/40">
          <Kicker as="p" tone="soft">
            Current
          </Kicker>
          <p className="font-mono text-xl md:text-2xl text-ink mt-3 break-all">
            {user.email}
          </p>
        </div>

        <form onSubmit={handleSubmit} noValidate className="space-y-6">
          <FieldShell id="new-email" label="New email">
            <input
              id="new-email"
              type="email"
              value={nextEmail}
              onChange={(e) => {
                setNextEmail(e.target.value);
                clearFieldError("next");
              }}
              placeholder="you@example.com"
              autoComplete="email"
              maxLength={EMAIL_MAX}
              aria-invalid={!!errors.next}
              aria-describedby={errors.next ? "new-email-error" : undefined}
              className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-4 text-lg text-ink placeholder:text-slate-soft transition-colors"
            />
            <FieldError
              message={errors.next}
              id="new-email-error"
              density="tight"
            />
          </FieldShell>

          <PasswordField
            id="email-current-password"
            label="Current password"
            autoComplete="current-password"
            value={password}
            error={errors.password}
            onChange={(v) => {
              setPassword(v);
              clearFieldError("password");
            }}
          />

          <FieldError message={submitError} id="account-email-submit-error" />

          <button
            type="submit"
            disabled={isSaving}
            className="font-display text-base font-bold bg-fresh hover:bg-fresh-deep text-surface py-3 px-6 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-2"
          >
            {isSaving ? "Sending…" : "Send confirmation"}
            {!isSaving && <span aria-hidden="true">→</span>}
          </button>
        </form>

        <p className="font-sans text-sm text-slate max-w-md">
          We&apos;ll email a confirmation link to the new address. Your sign-in
          email stays the same until you open it &mdash; and confirming signs
          you out on every device.
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
      next: validateNewPassword(next),
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
      const { sessionKept } = await changePassword({
        currentPassword: current,
        newPassword: next,
      });
      reset();
      showToast({
        kind: "success",
        // Without a stored refresh token the BE couldn't exempt this device
        // from the revoke sweep, so the session dies with the access token.
        // Saying nothing here means a silent bounce ~15 min later.
        message: sessionKept
          ? "Password updated."
          : "Password updated. You'll need to sign in again on this device.",
        duration: sessionKept ? undefined : 8000,
      });
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
          className="font-display text-base font-bold bg-fresh hover:bg-fresh-deep text-surface py-3 px-6 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-2"
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
  const queryClient = useQueryClient();
  const { showToast } = useToast();

  function setRole(next: User["role"]) {
    if (next === user.role) return;
    // A role swap is an identity change, so it needs the same wipe every real
    // auth transition gets — otherwise the new role reads the old one's cart,
    // orders, selfies and admin caches.
    resetUserScopedStores();
    queryClient.clear();
    setUser({ ...user, role: next });
    // resetUserScopedStores() parks cart + saved-events in local-only mode,
    // and the only thing that flips them back is AuthHydrator's merge effect
    // — keyed on [isAuthenticated, userId] behind a one-shot ref, so a
    // same-user swap never re-runs it. Restore sync by hand or every
    // subsequent cart change stops reaching the server.
    useCartStore.getState().setSyncEnabled(true);
    useSavedEventsStore.getState().setSyncEnabled(true);
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
        <p className="font-display text-xl md:text-2xl font-bold tracking-tight text-ink">
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
          <p className="font-display text-xl md:text-2xl font-bold tracking-tight text-ink">
            Sign out of QuickPitik.
          </p>
          <p className="font-sans text-sm text-slate mt-2 max-w-md">
            You&apos;ll need to sign in again on this browser to access your
            profile, race log, and orders.
          </p>
          <button
            type="button"
            onClick={handleSignOut}
            className="mt-5 font-display text-base font-bold border border-ink text-ink hover:bg-ink hover:text-surface py-3 px-6 rounded-full transition-colors inline-flex items-center gap-2"
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
