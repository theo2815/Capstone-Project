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
import { AvatarSlab } from "@/components/account/avatar-slab";
import { useAuth } from "@/hooks/use-auth";
import { useAuthStore } from "@/store/auth-store";
import { useToast } from "@/hooks/use-toast";
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
      <div className="flex-1 max-w-7xl mx-auto w-full px-6 md:px-10">
        <div className="md:grid md:grid-cols-[15rem_1fr] md:gap-12 lg:gap-20">
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
          <div className="stagger-children min-w-0 pb-8 md:pb-20">
            <NameSlab user={user} number="01" />
            {!isPhotographer && <PictureSlab number="02" />}
            <EmailSlab
              user={user}
              number={isPhotographer ? "02" : "03"}
            />
            <PasswordSlab number={isPhotographer ? "03" : "04"} />
            <DangerSlab number={isPhotographer ? "04" : "05"} />
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
  const [status, setStatus] = useState<FormStatus>({ kind: "idle" });

  const dirty = name.trim() !== user.name && name.trim().length > 0;

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!dirty) return;
    setStatus({ kind: "loading" });

    try {
      // TODO(backend): swap setTimeout for `api.put<User>("/me/profile", { name: name.trim() })`
      // when Spring Boot exposes the profile-update endpoint.
      await new Promise((r) => setTimeout(r, 600));
      const next = { ...user, name: name.trim() };
      setUser(next);
      setStatus({ kind: "idle" });
      showToast({ kind: "success", message: "Name updated." });
    } catch {
      setStatus({ kind: "error", message: "Could not save. Try again." });
    }
  }

  return (
    <Slab id="name" number={number} title="Name">
      <form onSubmit={handleSubmit} className="space-y-6">
        <FieldShell id="name" label="Full name">
          <input
            id="name"
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              if (status.kind !== "idle" && status.kind !== "loading") {
                setStatus({ kind: "idle" });
              }
            }}
            placeholder="Juan dela Cruz"
            autoComplete="name"
            required
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-4 text-lg text-ink placeholder:text-slate-soft transition-colors"
          />
        </FieldShell>

        <FormStatusLine status={status} />

        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <button
            type="submit"
            disabled={!dirty || status.kind === "loading"}
            className="font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-6 rounded-full transition-colors disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center gap-2"
          >
            {status.kind === "loading" ? "Saving…" : "Save name"}
            {status.kind !== "loading" && <span aria-hidden="true">→</span>}
          </button>
          {dirty && status.kind !== "loading" && (
            <button
              type="button"
              onClick={() => {
                setName(user.name);
                setStatus({ kind: "idle" });
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
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
            Sign-in email
          </p>
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

function PasswordSlab({ number }: { number: string }) {
  const { showToast } = useToast();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [status, setStatus] = useState<FormStatus>({ kind: "idle" });

  function reset() {
    setCurrent("");
    setNext("");
    setConfirm("");
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!current || !next || !confirm) {
      setStatus({ kind: "error", message: "Fill in every field." });
      return;
    }
    if (next.length < 8) {
      setStatus({
        kind: "error",
        message: "New password must be at least 8 characters.",
      });
      return;
    }
    if (next !== confirm) {
      setStatus({ kind: "error", message: "New passwords don't match." });
      return;
    }

    setStatus({ kind: "loading" });

    try {
      // TODO(backend): swap setTimeout for `api.put("/me/password", { currentPassword: current, newPassword: next })`
      // when Spring Boot exposes the password-change endpoint.
      await new Promise((r) => setTimeout(r, 600));
      reset();
      setStatus({ kind: "idle" });
      showToast({ kind: "success", message: "Password updated." });
    } catch {
      setStatus({ kind: "error", message: "Could not save. Try again." });
    }
  }

  function clearStatusOnInput() {
    if (status.kind !== "idle" && status.kind !== "loading") {
      setStatus({ kind: "idle" });
    }
  }

  return (
    <Slab id="password" number={number} title="Password" caption="Min. 8 characters">
      <form onSubmit={handleSubmit} className="space-y-6">
        <PasswordField
          id="current-password"
          label="Current password"
          autoComplete="current-password"
          value={current}
          onChange={(v) => {
            setCurrent(v);
            clearStatusOnInput();
          }}
        />
        <PasswordField
          id="new-password"
          label="New password"
          autoComplete="new-password"
          value={next}
          onChange={(v) => {
            setNext(v);
            clearStatusOnInput();
          }}
        />
        <PasswordField
          id="confirm-password"
          label="Confirm new password"
          autoComplete="new-password"
          value={confirm}
          onChange={(v) => {
            setConfirm(v);
            clearStatusOnInput();
          }}
        />

        <FormStatusLine status={status} />

        <button
          type="submit"
          disabled={status.kind === "loading"}
          className="font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-6 rounded-full transition-colors disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center gap-2"
        >
          {status.kind === "loading" ? "Saving…" : "Change password"}
          {status.kind !== "loading" && <span aria-hidden="true">→</span>}
        </button>
      </form>
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
  onChange,
}: {
  id: string;
  label: string;
  autoComplete: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <FieldShell id={id} label={label}>
      <input
        id={id}
        type="password"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="••••••••"
        autoComplete={autoComplete}
        required
        className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-4 text-lg text-ink placeholder:text-slate-soft transition-colors"
      />
    </FieldShell>
  );
}

// Success cases are now surfaced via toasts; this status line only renders
// field-level errors close to the form they describe.
type FormStatus =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; message: string };

function FormStatusLine({ status }: { status: FormStatus }) {
  if (status.kind !== "error") return null;
  return (
    <p role="alert" className="font-sans text-sm text-error">
      {status.message}
    </p>
  );
}
