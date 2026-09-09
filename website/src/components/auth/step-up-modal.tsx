"use client";

import { useState, type FormEvent } from "react";
import { Modal } from "@/components/ui/modal";
import { FieldError } from "@/components/ui/field-error";
import { BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ApiError } from "@/lib/api";
import { reauthForStepUp } from "@/lib/api-step-up";

// Password prompt shown when a sensitive payout action needs step-up re-auth.
// Resolves the caller's promise with a fresh step-up token, or null on cancel.
// Driven entirely by <StepUpProvider>, which the api client invokes on a
// STEP_UP_REQUIRED 403.
export function StepUpModal({
  open,
  onResolve,
}: {
  open: boolean;
  onResolve: (token: string | null) => void;
}) {
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function reset() {
    setPassword("");
    setBusy(false);
    setError(null);
  }

  function cancel() {
    reset();
    onResolve(null);
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (busy) return;
    if (!password) {
      setError("Enter your password to continue.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const token = await reauthForStepUp(password);
      reset();
      onResolve(token);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? (err.errors[0]?.message ?? "That didn't work. Please try again.")
          : "That didn't work. Please try again.",
      );
      setBusy(false);
    }
  }

  return (
    <Modal isOpen={open} onClose={cancel} title="Confirm it's you">
      <form onSubmit={handleSubmit} className="space-y-5">
        <p className="font-sans text-sm text-ink-soft">
          For your security, re-enter your password to continue with this payout action.
        </p>
        <div className="flex flex-col gap-2">
          <label htmlFor="stepup-password" className="kicker">
            Password
          </label>
          <input
            id="stepup-password"
            type="password"
            autoFocus
            autoComplete="current-password"
            value={password}
            onChange={(e) => {
              setPassword(e.target.value);
              if (error) setError(null);
            }}
            placeholder="Your password"
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors"
          />
          <FieldError message={error} id="stepup-error" density="tight" />
        </div>
        <div className="flex flex-wrap items-center justify-end gap-3 pt-2 border-t border-line">
          <button
            type="button"
            onClick={cancel}
            disabled={busy}
            className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors px-4 py-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={busy}
            className={`${BTN_PRIMARY} ${BTN_SIZE.sm}`}
          >
            {busy ? "Confirming…" : "Confirm"}
          </button>
        </div>
      </form>
    </Modal>
  );
}
