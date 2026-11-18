"use client";

import { useEffect, useRef, useState } from "react";
import { Kicker } from "@/components/ui/kicker";
import { BTN_PRIMARY, BTN_DANGER, BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { FieldError } from "@/components/ui/field-error";
import { Skeleton } from "@/components/ui/skeleton";
import { AdminFieldLabel } from "@/components/admin/admin-form-fields";
import { ApiError } from "@/lib/api";
import { cn, copyToClipboard } from "@/lib/utils";
import { useAdminMutation } from "@/hooks/use-admin-mutation";
import { useToast } from "@/hooks/use-toast";
import { useAuthStore } from "@/store/auth-store";
import { captureAuthSession } from "@/lib/auth-session";
import {
  confirmMfa,
  disableMfa,
  enrollMfa,
  fetchMfaStatus,
  type MfaEnrollResponse,
} from "@/lib/api-admin-mfa";

// /admin/security — admin TOTP two-step verification management. Enroll →
// scan/enter the secret → confirm with a code → save recovery codes; or disable
// with a code. Enforcement at login is handled by the backend; this is the
// setup surface. QR rendering is owed (no QR dependency yet) — the secret is
// shown for manual entry, which every authenticator app accepts.
export default function AdminSecurityPage() {
  const [loading, setLoading] = useState(true);
  const [enabled, setEnabled] = useState(false);
  const [enroll, setEnroll] = useState<MfaEnrollResponse | null>(null);
  const [recoveryCodes, setRecoveryCodes] = useState<string[] | null>(null);
  const [code, setCode] = useState("");
  const [disarming, setDisarming] = useState(false);
  const mutation = useAdminMutation();
  const userId = useAuthStore((state) => state.user?.id);
  const targets = [{ kind: "security" as const, id: userId ?? "" }];
  const busy = mutation.isPending(targets);
  const [error, setError] = useState<string | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [loadAttempt, setLoadAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    fetchMfaStatus()
      .then((s) => {
        if (!active) return;
        if (s) setEnabled(s.enabled);
        else setLoadFailed(true);
      })
      .catch(() => {
        // Unknown is not "off": showing Enable here would invite a second enrollment.
        if (active) setLoadFailed(true);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [loadAttempt]);

  function retryLoad() {
    setLoadFailed(false);
    setLoading(true);
    setLoadAttempt((n) => n + 1);
  }

  function messageFor(err: unknown, fallback: string): string {
    return err instanceof ApiError
      ? (err.errors[0]?.message ?? fallback)
      : fallback;
  }

  async function handleEnroll() {
    await run("Starting…", "Could not start setup. Please try again.", async (current) => {
      const response = await enrollMfa();
      if (current()) setEnroll(response);
    });
  }

  async function run(action: string, fallback: string, operation: (current: () => boolean) => Promise<void>) {
    const current = captureAuthSession();
    try {
      await mutation.run({ targets, action, execute: async () => {
        setError(null);
        // Sensitive results stay in local state, never in mutation results or receipts.
        await operation(current);
      } });
    } catch (err) {
      if (current()) setError(messageFor(err, fallback));
    }
  }

  async function handleConfirm() {
    if (busy) return;
    if (!code.trim()) {
      setError("Enter the 6-digit code from your authenticator app.");
      return;
    }
    await run("Verifying…", "That code didn't work. Please try again.", async (current) => {
      const res = await confirmMfa(code.trim());
      if (!current()) return;
      setRecoveryCodes(res.recoveryCodes);
      setEnroll(null);
      setEnabled(true);
      setCode("");
    });
  }

  async function handleDisable() {
    if (busy) return;
    if (!code.trim()) {
      setError("Enter a current code to turn off two-step verification.");
      return;
    }
    await run("Turning off…", "That code didn't work. Please try again.", async (current) => {
      await disableMfa(code.trim());
      if (!current()) return;
      setEnabled(false);
      setDisarming(false);
      setCode("");
    });
  }

  function cancelFlow() {
    if (mutation.isPending(targets)) return;
    setEnroll(null);
    setDisarming(false);
    setCode("");
    setError(null);
  }

  return (
    <>
      <header className="pb-8 md:pb-12 border-b border-line">
        <Kicker as="p">Security · Two-step verification</Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
          Security.
        </h1>
        <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
          Add a second step to admin sign-in with an authenticator app. Required
          for the accounts that approve and pay out real money.
        </p>
      </header>

      <section className="pt-8 md:pt-10 max-w-xl">
        {loading ? (
          <div className="space-y-4" aria-busy="true">
            <Skeleton className="h-4 w-40" />
            <Skeleton className="h-11 w-52" />
          </div>
        ) : loadFailed ? (
          <div role="alert" className="space-y-3">
            <p className="font-sans text-sm text-error">
              Couldn&apos;t load your two-step verification status. Please try again.
            </p>
            <button
              type="button"
              onClick={retryLoad}
              className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
            >
              Retry
            </button>
          </div>
        ) : recoveryCodes ? (
          <RecoveryCodes codes={recoveryCodes} onDone={() => setRecoveryCodes(null)} />
        ) : enroll ? (
          <EnrollPanel
            enroll={enroll}
            code={code}
            busy={busy}
            error={error}
            onCode={(v) => {
              setCode(v);
              if (error) setError(null);
            }}
            onConfirm={handleConfirm}
            onCancel={cancelFlow}
          />
        ) : enabled ? (
          <EnabledPanel
            disarming={disarming}
            code={code}
            busy={busy}
            error={error}
            onStartDisable={() => {
              setDisarming(true);
              setError(null);
            }}
            onCode={(v) => {
              setCode(v);
              if (error) setError(null);
            }}
            onDisable={handleDisable}
            onCancel={cancelFlow}
          />
        ) : (
          <div className="space-y-5">
            <Kicker as="p" tone="soft">
              Status · Off
            </Kicker>
            <p className="font-sans text-base text-ink-soft">
              Two-step verification is off for your account.
            </p>
            <FieldError message={error} id="mfa-error" />
            <button
              type="button"
              onClick={handleEnroll}
              disabled={busy}
              aria-busy={busy}
              className={`${BTN_PRIMARY} ${BTN_SIZE.md}`}
            >
              {busy ? "Starting…" : "Enable 2-step verification"}
            </button>
          </div>
        )}
      </section>
    </>
  );
}

function EnrollPanel({
  enroll,
  code,
  busy,
  error,
  onCode,
  onConfirm,
  onCancel,
}: {
  enroll: MfaEnrollResponse;
  code: string;
  busy: boolean;
  error: string | null;
  onCode: (v: string) => void;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const { showToast } = useToast();
  const secretRef = useRef<HTMLParagraphElement>(null);
  return (
    <form
      className="space-y-6"
      onSubmit={(e) => {
        e.preventDefault();
        onConfirm();
      }}
    >
      <div className="space-y-3">
        <Kicker as="p" tone="soft">
          Step 1 · Add the key to your app
        </Kicker>
        <p className="font-sans text-sm text-ink-soft">
          In your authenticator app, add an account by entering this key
          manually. (QR scanning is coming soon.)
        </p>
        <div className="rounded-xl border border-line bg-bone-deep p-4">
          <p ref={secretRef} className="font-mono text-base text-ink break-all tracking-[0.08em]">
            {enroll.secret}
          </p>
        </div>
        <button
          type="button"
          onClick={() => void copyOrSelect(enroll.secret, "Key", secretRef.current, showToast)}
          className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
        >
          Copy key
        </button>
      </div>

      <div className="space-y-3">
        <Kicker as="p" tone="soft">
          Step 2 · Enter the code it shows
        </Kicker>
        <AdminFieldLabel htmlFor="mfa-confirm-code">6-digit code</AdminFieldLabel>
        <input
          id="mfa-confirm-code"
          type="text"
          inputMode="numeric"
          autoComplete="one-time-code"
          disabled={busy}
          value={code}
          onChange={(e) => onCode(e.target.value)}
          placeholder="123456"
          aria-invalid={error ? true : undefined}
          aria-describedby={error ? "mfa-error" : undefined}
          className="w-full bg-transparent border-b border-line focus:border-fresh py-3 text-base text-ink placeholder:text-slate-soft transition-colors font-mono tnum"
        />
        <FieldError message={error} id="mfa-error" />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="submit"
          disabled={busy}
          aria-busy={busy}
          className={`${BTN_PRIMARY} ${BTN_SIZE.md}`}
        >
          {busy ? "Verifying…" : "Turn on"}
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={busy}
          aria-busy={busy}
          className={cn(BTN_GHOST, BTN_SIZE.sm)}
        >
          Cancel
        </button>
      </div>
    </form>
  );
}

// Copy, or select the text so the admin can copy it by hand.
async function copyOrSelect(
  text: string,
  label: string,
  target: HTMLElement | null,
  showToast: ReturnType<typeof useToast>["showToast"],
) {
  if (await copyToClipboard(text)) {
    showToast({ kind: "success", message: `${label} copied` });
    return;
  }
  if (target) window.getSelection()?.selectAllChildren(target);
  showToast({ kind: "error", message: "Could not copy. The text is selected: copy it by hand." });
}

function downloadCodes(codes: string[]) {
  const text = `QuickPitik admin recovery codes\nEach code works once.\n\n${codes.join("\n")}\n`;
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "quickpitik-recovery-codes.txt";
  link.click();
  // Next tick: some browsers start the download after the click handler returns.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function RecoveryCodes({
  codes,
  onDone,
}: {
  codes: string[];
  onDone: () => void;
}) {
  const { showToast } = useToast();
  const listRef = useRef<HTMLUListElement>(null);
  return (
    <div className="space-y-5">
      <Kicker as="p" tone="soft">
        Two-step verification is on
      </Kicker>
      <p className="font-sans text-base text-ink">
        Save these recovery codes somewhere safe. Each works once if you lose
        your authenticator — this is the only time they&apos;re shown.
      </p>
      <ul ref={listRef} className="grid grid-cols-2 gap-x-8 gap-y-2 rounded-xl border border-line bg-bone-deep p-5">
        {codes.map((c) => (
          <li key={c} className="font-mono tnum text-base text-ink">
            {c}
          </li>
        ))}
      </ul>
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => void copyOrSelect(codes.join("\n"), "Recovery codes", listRef.current, showToast)}
          className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
        >
          Copy codes
        </button>
        <button
          type="button"
          onClick={() => downloadCodes(codes)}
          className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
        >
          Download codes
        </button>
      </div>
      <button
        type="button"
        onClick={onDone}
        className={`${BTN_PRIMARY} ${BTN_SIZE.md}`}
      >
        I&apos;ve saved them
      </button>
    </div>
  );
}

function EnabledPanel({
  disarming,
  code,
  busy,
  error,
  onStartDisable,
  onCode,
  onDisable,
  onCancel,
}: {
  disarming: boolean;
  code: string;
  busy: boolean;
  error: string | null;
  onStartDisable: () => void;
  onCode: (v: string) => void;
  onDisable: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="space-y-5">
      <Kicker as="p" tone="soft">
        Status · On
      </Kicker>
      <p className="font-sans text-base text-ink-soft">
        Sign-in asks for a code from your authenticator app.
      </p>
      {disarming ? (
        <form
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            onDisable();
          }}
        >
          <p className="font-sans text-sm text-ink-soft">
            Enter a current code to turn two-step verification off.
          </p>
          <AdminFieldLabel htmlFor="mfa-disable-code">6-digit code</AdminFieldLabel>
          <input
            id="mfa-disable-code"
            type="text"
            inputMode="numeric"
            autoComplete="one-time-code"
            disabled={busy}
            value={code}
            onChange={(e) => onCode(e.target.value)}
            placeholder="123456"
            aria-invalid={error ? true : undefined}
            aria-describedby={error ? "mfa-error" : undefined}
            className="w-full max-w-xs bg-transparent border-b border-line focus:border-fresh py-3 text-base text-ink placeholder:text-slate-soft transition-colors font-mono tnum"
          />
          <FieldError message={error} id="mfa-error" />
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="submit"
              disabled={busy}
              aria-busy={busy}
              className={`${BTN_DANGER} ${BTN_SIZE.md}`}
            >
              {busy ? "Turning off…" : "Turn off"}
            </button>
            <button
              type="button"
              onClick={onCancel}
              disabled={busy}
              aria-busy={busy}
              className={cn(BTN_GHOST, BTN_SIZE.sm)}
            >
              Cancel
            </button>
          </div>
        </form>
      ) : (
        <button
          type="button"
          onClick={onStartDisable}
          className={`${BTN_DANGER} ${BTN_SIZE.md}`}
        >
          Turn off
        </button>
      )}
    </div>
  );
}
