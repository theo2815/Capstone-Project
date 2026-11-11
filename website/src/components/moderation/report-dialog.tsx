"use client";

import { useId, useState } from "react";
import { Modal } from "@/components/ui/modal";
import { Kicker } from "@/components/ui/kicker";
import { CharCounter } from "@/components/ui/char-counter";
import { FieldError } from "@/components/ui/field-error";
import { BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ApiError } from "@/lib/api";
import { sendReport, type ReportTarget } from "@/lib/api-reports";
import { useAuthStore } from "@/store/auth-store";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";
import { cn } from "@/lib/utils";

// Same bounds as the backend ReportRequest (AA-R9).
const NOTE_MIN = 10;
const NOTE_MAX = 500;

// Quiet "Report" link + dialog. Signed-in only, and never on your own profile.
// The server decides what happens to the report; this only sends the note.
export function ReportButton({ target, className }: { target: ReportTarget; className?: string }) {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const ownHandle = usePhotographerSettingsStore((s) => s.handle);
  const [open, setOpen] = useState(false);

  if (!isAuthenticated) return null;
  if (target.kind === "photographer" && ownHandle && ownHandle === target.handle) return null;

  return (
    <>
      <Kicker
        as="button"
        type="button"
        onClick={() => setOpen(true)}
        className={cn("min-h-[44px] hover:text-ink transition-colors", className)}
      >
        Report
      </Kicker>
      {open && <ReportDialog target={target} onClose={() => setOpen(false)} />}
    </>
  );
}

function ReportDialog({ target, onClose }: { target: ReportTarget; onClose: () => void }) {
  const noteId = useId();
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);

  const trimmed = note.trim();
  const canSend = trimmed.length >= NOTE_MIN && trimmed.length <= NOTE_MAX && !sending;
  const what = target.kind === "event" ? "event" : "photographer";

  async function handleSend() {
    if (!canSend) return;
    setError(null);
    setSending(true);
    try {
      await sendReport(target, trimmed);
      setSent(true);
    } catch (err) {
      // Keep the note so the user can retry; 429 and validation messages come from the server.
      setError(err instanceof ApiError ? err.message : "Could not send the report. Try again.");
    } finally {
      setSending(false);
    }
  }

  const close = () => {
    if (!sending) onClose();
  };

  return (
    <Modal isOpen onClose={close} hideClose={sending} title={`Report this ${what}`}>
      {sent ? (
        <>
          <p className="font-sans text-base text-ink">Thanks, an admin will look at it.</p>
          <div className="mt-6 flex justify-end">
            <button type="button" onClick={onClose} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>
              Close
            </button>
          </div>
        </>
      ) : (
        <>
          <fieldset disabled={sending} className="min-w-0">
            <Kicker as="label" tone="soft" htmlFor={noteId} className="mb-2 block">
              What is wrong?
            </Kicker>
            <textarea
              id={noteId}
              value={note}
              onChange={(e) => setNote(e.target.value)}
              maxLength={NOTE_MAX}
              placeholder={`Tell the admins what is wrong with this ${what} (at least ${NOTE_MIN} characters).`}
              className="w-full font-sans text-sm text-ink bg-bone border border-line rounded-xl px-4 py-3 min-h-[120px] resize-y focus:outline-none focus:border-ink placeholder:text-slate-soft"
              aria-describedby={`${noteId}-counter`}
            />
            <CharCounter id={`${noteId}-counter`} current={note.length} max={NOTE_MAX} className="mt-1.5" />
            <FieldError id={`${noteId}-error`} message={error ?? undefined} />
          </fieldset>
          <div className="mt-6 flex items-center justify-end gap-3">
            <button type="button" onClick={close} disabled={sending} className={cn(BTN_GHOST, BTN_SIZE.sm)}>
              Cancel
            </button>
            <button type="button" onClick={handleSend} disabled={!canSend} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>
              {sending ? "Sending…" : "Send report"}
            </button>
          </div>
        </>
      )}
    </Modal>
  );
}
