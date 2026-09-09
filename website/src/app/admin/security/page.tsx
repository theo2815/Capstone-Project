"use client";

import { useEffect, useState } from "react";
import { Kicker } from "@/components/ui/kicker";
import { BTN_PRIMARY, BTN_DANGER, BTN_SIZE } from "@/components/ui/button-styles";
import { FieldError } from "@/components/ui/field-error";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError } from "@/lib/api";
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
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchMfaStatus()
      .then((s) => {
        if (active) setEnabled(s?.enabled ?? false);
      })
      .catch(() => {
        /* leave disabled; the action buttons still work + surface errors */
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  function messageFor(err: unknown, fallback: string): string {
    return err instanceof ApiError
      ? (err.errors[0]?.message ?? fallback)
      : fallback;
  }

  async function handleEnroll() {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      setEnroll(await enrollMfa());
    } catch (err) {
      setError(messageFor(err, "Could not start setup. Please try again."));
    } finally {
      setBusy(false);
    }
  }

  async function handleConfirm() {
    if (busy) return;
    if (!code.trim()) {
      setError("Enter the 6-digit code from your authenticator app.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const res = await confirmMfa(code.trim());
      setRecoveryCodes(res.recoveryCodes);
      setEnroll(null);
      setEnabled(true);
      setCode("");
    } catch (err) {
      setError(messageFor(err, "That code didn't work. Please try again."));
    } finally {
      setBusy(false);
    }
  }

  async function handleDisable() {
    if (busy) return;
    if (!code.trim()) {
      setError("Enter a current code to turn off two-step verification.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await disableMfa(code.trim());
      setEnabled(false);
      setDisarming(false);
      setCode("");
    } catch (err) {
      setError(messageFor(err, "That code didn't work. Please try again."));
    } finally {
      setBusy(false);
    }
  }

  function cancelFlow() {
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
          <div className="space-y-4">
            <Skeleton className="h-4 w-40" />
            <Skeleton className="h-11 w-52" />
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
  return (
    <div className="space-y-6">
      <div className="space-y-3">
        <Kicker as="p" tone="soft">
          Step 1 · Add the key to your app
        </Kicker>
        <p className="font-sans text-sm text-ink-soft">
          In your authenticator app, add an account by entering this key
          manually. (QR scanning is coming soon.)
        </p>
        <div className="rounded-xl border border-line bg-bone-deep p-4">
          <p className="font-mono text-base text-ink break-all tracking-[0.08em]">
            {enroll.secret}
          </p>
        </div>
      </div>

      <div className="space-y-3">
        <Kicker as="p" tone="soft">
          Step 2 · Enter the code it shows
        </Kicker>
        <input
          id="mfa-confirm-code"
          type="text"
          inputMode="numeric"
          autoComplete="one-time-code"
          value={code}
          onChange={(e) => onCode(e.target.value)}
          placeholder="123456"
          className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors font-mono tnum"
        />
        <FieldError message={error} id="mfa-error" />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={onConfirm}
          disabled={busy}
          className={`${BTN_PRIMARY} ${BTN_SIZE.md}`}
        >
          {busy ? "Verifying…" : "Turn on"}
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={busy}
          className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors px-4 py-2 disabled:opacity-50"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

function RecoveryCodes({
  codes,
  onDone,
}: {
  codes: string[];
  onDone: () => void;
}) {
  return (
    <div className="space-y-5">
      <Kicker as="p" tone="soft">
        Two-step verification is on
      </Kicker>
      <p className="font-sans text-base text-ink">
        Save these recovery codes somewhere safe. Each works once if you lose
        your authenticator — this is the only time they&apos;re shown.
      </p>
      <ul className="grid grid-cols-2 gap-x-8 gap-y-2 rounded-xl border border-line bg-bone-deep p-5">
        {codes.map((c) => (
          <li key={c} className="font-mono tnum text-base text-ink">
            {c}
          </li>
        ))}
      </ul>
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
        <div className="space-y-4">
          <p className="font-sans text-sm text-ink-soft">
            Enter a current code to turn two-step verification off.
          </p>
          <input
            id="mfa-disable-code"
            type="text"
            inputMode="numeric"
            autoComplete="one-time-code"
            value={code}
            onChange={(e) => onCode(e.target.value)}
            placeholder="123456"
            className="w-full max-w-xs bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors font-mono tnum"
          />
          <FieldError message={error} id="mfa-error" />
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={onDisable}
              disabled={busy}
              className={`${BTN_DANGER} ${BTN_SIZE.md}`}
            >
              {busy ? "Turning off…" : "Turn off"}
            </button>
            <button
              type="button"
              onClick={onCancel}
              disabled={busy}
              className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors px-4 py-2 disabled:opacity-50"
            >
              Cancel
            </button>
          </div>
        </div>
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
