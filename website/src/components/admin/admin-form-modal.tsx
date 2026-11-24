"use client";

import { type FormEvent, type ReactNode } from "react";
import { Modal } from "@/components/ui/modal";
import { BTN_DANGER, BTN_GHOST, BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import { cn } from "@/lib/utils";

// Admin modal shells. Every form-bearing modal (reject / suspend / hide /
// deny / hold / escalate / mark-paid / resolve / message / edit-photographer)
// wraps `AdminFormModal`; confirm-only modals wrap `AdminConfirmModal`.
// Both own the Modal portal + title, the body wrapper, an optional inline
// error, and the Finish Line footer: Cancel is BTN_GHOST, the action is
// BTN_PRIMARY or, with `tone="danger"`, BTN_DANGER (design-system hard rule 7).

type Tone = "primary" | "danger";
const FOOTER_CLS = "flex flex-wrap items-center justify-end gap-3 pt-2";
const CANCEL_BTN_CLS = cn(BTN_GHOST, BTN_SIZE.sm);
const actionCls = (tone: Tone) => cn(tone === "danger" ? BTN_DANGER : BTN_PRIMARY, BTN_SIZE.sm);

function FooterError({ error }: { error?: string | null }) {
  return error ? <p role="alert" className="text-sm text-error">{error}</p> : null;
}

interface AdminFormModalProps {
  open: boolean;
  onClose: () => void;
  /** Called when the user submits the form (Enter or footer Submit). */
  onSubmit: () => void;
  title: string;
  /**
   * Explanatory paragraph rendered above the form fields. Pass
   * `<>… <span className="text-ink tnum">{value}</span> …</>` for
   * inline numerics. Optional — escalate has no intro because the title
   * already carries the target.
   */
  intro?: ReactNode;
  submitLabel: string;
  submitDisabled?: boolean;
  /** Id of the line that says why the submit is off, read out with the button. */
  submitDescribedBy?: string;
  submitting?: boolean;
  pendingLabel?: string;
  cancelLabel?: string;
  /** Pass true while a stacked confirmation child is open above this form. */
  escDisabled?: boolean;
  /** "danger" for a destructive action (Suspend, Reject, Hide, Hold…). */
  tone?: Tone;
  /** Why the last attempt failed; shown above the footer. */
  error?: string | null;
  /** Skip the browser's own "Please fill out this field." bubble; the form reports its own errors. */
  noValidate?: boolean;
  children: ReactNode;
}
export function AdminFormModal({
  open,
  onClose,
  onSubmit,
  title,
  intro,
  submitLabel,
  submitDisabled = false,
  submitDescribedBy,
  submitting = false,
  pendingLabel = "Saving…",
  cancelLabel = "Cancel",
  escDisabled = false,
  tone = "primary",
  error,
  noValidate = false,
  children,
}: AdminFormModalProps) {
  function handle(e: FormEvent) {
    e.preventDefault();
    if (submitDisabled || submitting) return;
    onSubmit();
  }
  function handleClose() {
    if (!submitting) onClose();
  }
  return (
    <Modal
      isOpen={open}
      onClose={handleClose}
      title={title}
      escDisabled={escDisabled || submitting}
      hideClose={submitting}
    >
      <form onSubmit={handle} noValidate={noValidate} className="space-y-5">
        {intro && (
          <p className="font-sans text-sm text-ink-soft">{intro}</p>
        )}
        <fieldset disabled={submitting} className="min-w-0 space-y-5">
          {children}
        </fieldset>
        <FooterError error={error} />
        <div className={FOOTER_CLS}>
          <button type="button" onClick={handleClose} disabled={submitting} className={CANCEL_BTN_CLS}>
            {cancelLabel}
          </button>
          <button
            type="submit"
            aria-busy={submitting}
            aria-describedby={submitDescribedBy}
            disabled={submitDisabled || submitting}
            className={actionCls(tone)}
          >
            {submitting ? pendingLabel : submitLabel}
          </button>
        </div>
      </form>
    </Modal>
  );
}

interface AdminConfirmModalProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  confirmLabel: string;
  confirmDisabled?: boolean;
  submitting?: boolean;
  pendingLabel?: string;
  cancelLabel?: string;
  escDisabled?: boolean;
  tone?: Tone;
  error?: string | null;
  children: ReactNode;
}
// Confirm-only variant: no form, no fields.
export function AdminConfirmModal({
  open,
  onClose,
  onConfirm,
  title,
  confirmLabel,
  confirmDisabled = false,
  submitting = false,
  pendingLabel = "Saving…",
  cancelLabel = "Cancel",
  escDisabled = false,
  tone = "primary",
  error,
  children,
}: AdminConfirmModalProps) {
  function handleClose() {
    if (!submitting) onClose();
  }
  return (
    <Modal
      isOpen={open}
      onClose={handleClose}
      title={title}
      escDisabled={escDisabled || submitting}
      hideClose={submitting}
    >
      <div className="space-y-5">
        {children}
        <FooterError error={error} />
        <div className={FOOTER_CLS}>
          <button type="button" onClick={handleClose} disabled={submitting} className={CANCEL_BTN_CLS}>
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            aria-busy={submitting}
            disabled={confirmDisabled || submitting}
            className={actionCls(tone)}
          >
            {submitting ? pendingLabel : confirmLabel}
          </button>
        </div>
      </div>
    </Modal>
  );
}
