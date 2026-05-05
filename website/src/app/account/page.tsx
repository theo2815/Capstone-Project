"use client";

import Link from "next/link";
import { useEffect, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { ProtectedRoute } from "@/components/auth/protected-route";
import { SiteHeader } from "@/components/layout/site-header";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { AvatarSlab } from "@/components/account/avatar-slab";
import { useAuth } from "@/hooks/use-auth";
import { useAuthStore } from "@/store/auth-store";
import { useToast } from "@/hooks/use-toast";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { Role, User } from "@/types/user";

const JUMP_SECTIONS: ReadonlyArray<{ id: string; label: string }> = [
  { id: "name", label: "Name" },
  { id: "picture", label: "Picture" },
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

  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col scroll-smooth">
      <SiteHeader />
      <div className="flex-1 max-w-5xl mx-auto w-full px-6 md:px-10">
        <div className="md:grid md:grid-cols-[15rem_1fr] md:gap-12 lg:gap-20">
          <SettingsRail user={user} />
          <div className="stagger-children min-w-0 pb-8 md:pb-20">
            <NameSlab user={user} />
            <PictureSlab />
            <EmailSlab user={user} />
            <PasswordSlab />
            <DangerSlab />
          </div>
        </div>
      </div>
      <Footer />
    </main>
  );
}

function SettingsRail({ user }: { user: User }) {
  const router = useRouter();
  const { logout } = useAuth();
  const activeId = useActiveSection(JUMP_SECTIONS.map((s) => s.id));
  const moreLinks = moreLinksForRole(user.role, ROUTES.ACCOUNT);

  function handleSignOut() {
    logout();
    router.replace(ROUTES.HOME);
  }

  return (
    <aside className="md:sticky md:top-24 md:self-start pt-10 md:pt-20 pb-8 md:pb-12 border-b md:border-0 border-line">
      <div className="flex items-start gap-5 md:block">
        <AvatarDisc name={user.name} size="md" />
        <div className="flex-1 min-w-0 md:mt-7">
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
            Profile · Settings
          </p>
          <h1 className="font-display text-3xl md:text-4xl font-medium tracking-tight leading-[1.05] mt-3">
            <span className="text-fresh">Account.</span>
          </h1>
          <p className="font-sans text-sm text-slate mt-3 max-w-xs">
            Edit how you appear and keep your account secure.
          </p>
        </div>
      </div>

      <nav aria-label="Jump to section" className="hidden md:block mt-12">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
          Jump to
        </p>
        <ul className="mt-4 space-y-3">
          {JUMP_SECTIONS.map((section) => {
            const isActive = activeId === section.id;
            return (
              <li key={section.id}>
                <a
                  href={`#${section.id}`}
                  aria-current={isActive ? "true" : undefined}
                  className={cn(
                    "group flex items-center gap-3 font-display text-base transition-colors",
                    isActive ? "text-ink" : "text-slate hover:text-ink",
                  )}
                >
                  <span
                    aria-hidden="true"
                    className={cn(
                      "size-1.5 rounded-full transition-colors",
                      isActive ? "bg-fresh" : "bg-line group-hover:bg-slate",
                    )}
                  />
                  <span>{section.label}</span>
                </a>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="mt-8 md:mt-12">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
          More
        </p>
        <ul className="mt-4 space-y-3">
          {moreLinks.map((link) => (
            <li key={link.href}>
              <Link
                href={link.href}
                className="font-display text-base text-slate hover:text-ink transition-colors"
              >
                {link.label}
              </Link>
            </li>
          ))}
          <li>
            <button
              type="button"
              onClick={handleSignOut}
              className="font-display text-base text-slate hover:text-ink transition-colors"
            >
              Sign out
            </button>
          </li>
        </ul>
      </div>
    </aside>
  );
}

function NameSlab({ user }: { user: User }) {
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
    <Slab id="name" number="01" title="Name">
      <form onSubmit={handleSubmit} className="space-y-6">
        <FieldShell id="name" label="Display name">
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

function PictureSlab() {
  return (
    <Slab id="picture" number="02" title="Picture" caption="Shown next to your name">
      <AvatarSlab />
    </Slab>
  );
}

function EmailSlab({ user }: { user: User }) {
  return (
    <Slab id="email" number="03" title="Email" caption="Used to sign in">
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

function PasswordSlab() {
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
    <Slab id="password" number="04" title="Password" caption="Min. 8 characters">
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

function DangerSlab() {
  const router = useRouter();
  const { logout } = useAuth();

  function handleSignOut() {
    logout();
    router.replace(ROUTES.HOME);
  }

  return (
    <Slab id="danger" number="05" title="Danger" caption="Account-wide actions">
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

function Slab({
  id,
  number,
  title,
  caption,
  children,
}: {
  id?: string;
  number: string;
  title: string;
  caption?: string;
  children: React.ReactNode;
}) {
  return (
    <section
      id={id}
      className="border-t border-line py-12 md:py-14 scroll-mt-24 first:border-0 first:pt-10 md:first:pt-20"
    >
      <div className="flex items-baseline gap-4 min-w-0 mb-7 md:mb-8">
        <span className="font-mono text-[10px] tracking-[0.15em] text-slate-soft tnum">
          {number}
        </span>
        <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-ink shrink-0">
          {title}
        </p>
        {caption && (
          <p className="hidden md:block font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft truncate">
            {caption}
          </p>
        )}
      </div>
      {children}
    </section>
  );
}

function Footer() {
  return (
    <footer className="px-6 md:px-10 py-8 mt-12 flex flex-col md:flex-row items-center justify-between gap-4 border-t border-line bg-bone">
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        QuickPitik &middot; Cebu, Philippines
      </p>
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        © 2026
      </p>
    </footer>
  );
}

function useActiveSection(ids: ReadonlyArray<string>): string | null {
  const [active, setActive] = useState<string | null>(ids[0] ?? null);
  useEffect(() => {
    const elements = ids
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => el !== null);
    if (elements.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible.length > 0) {
          setActive(visible[0].target.id);
        }
      },
      {
        rootMargin: "-25% 0px -55% 0px",
        threshold: [0, 0.1, 0.5, 1],
      },
    );

    elements.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [ids]);
  return active;
}

function moreLinksForRole(
  role: Role,
  current: string,
): ReadonlyArray<{ label: string; href: string }> {
  const all: Array<{ label: string; href: string }> = [
    { label: "Profile", href: ROUTES.PROFILE },
    { label: "Account", href: ROUTES.ACCOUNT },
  ];
  if (role === "RUNNER") all.push({ label: "Orders", href: ROUTES.ORDERS });
  if (role === "PHOTOGRAPHER")
    all.push({ label: "Dashboard", href: ROUTES.DASHBOARD });
  if (role === "ADMIN") all.push({ label: "Admin", href: ROUTES.ADMIN });
  return all.filter((l) => l.href !== current);
}
