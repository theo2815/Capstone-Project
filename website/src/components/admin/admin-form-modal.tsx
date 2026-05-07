"use client";

import { type FormEvent, type ReactNode } from "react";
import { Modal } from "@/components/ui/modal";

// Phase 6 admin destructive-modal shells. Every form-bearing destructive
// modal (reject / suspend / hide / deny / hold / escalate / mark-paid /
// resolve / message / edit-photographer) wraps `AdminFormModal`. Confirm-
// only modals (refund-confirm / bulk-approve) wrap `AdminConfirmModal`.
//
// Both shells own:
//   - The Modal portal + title (Quiet Studio mono kicker).
//   - The body wrapper (`space-y-5`) + optional explanatory intro line.
//   - The footer pair (Cancel + Submit/Confirm) with monochrome
//     outline-pill styling that matches the rest of the admin redesign.
//
// Submit styling is intentionally identical across all 12 modals — the
// destructive intent reads through the title, intro copy, and disabled-
// until-valid gate, not through a danger color (locked Phase 6
// decision: keep Quiet Studio monochrome).

const FOOTER_CLS = "flex items-center justify-end gap-3 pt-2";
const CANCEL_BTN_CLS =
  "font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors px-4 py-2";
const SUBMIT_BTN_CLS =
  "font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink border border-line hover:bg-ink hover:text-bone transition-colors rounded-full px-5 py-2 disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-ink disabled:cursor-not-allowed";

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
  cancelLabel?: string;
  /** Pass true while a stacked confirmation child is open above this form. */
  escDisabled?: boolean;
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
  cancelLabel = "Cancel",
  escDisabled = false,
  children,
}: AdminFormModalProps) {
  function handle(e: FormEvent) {
    e.preventDefault();
    if (submitDisabled) return;
    onSubmit();
  }
  return (
    <Modal
      isOpen={open}
      onClose={onClose}
      title={title}
      escDisabled={escDisabled}
    >
      <form onSubmit={handle} className="space-y-5">
        {intro && (
          <p className="font-sans text-sm text-ink-soft">{intro}</p>
        )}
        {children}
        <div className={FOOTER_CLS}>
          <button type="button" onClick={onClose} className={CANCEL_BTN_CLS}>
            {cancelLabel}
          </button>
          <button
            type="submit"
            disabled={submitDisabled}
            className={SUBMIT_BTN_CLS}
          >
            {submitLabel}
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
  cancelLabel?: string;
  escDisabled?: boolean;
  children: ReactNode;
}
// Confirm-only variant — no form, no fields. Used for the two pure
// confirmation modals (refund-confirm child of resolve, bulk-approve).
// Both previously used `bg-fresh text-bone` submits; Phase 6 normalizes
// them to the same outline pill the other 10 modals use, so the page
// stays under the Quiet Studio one-fresh-element-per-viewport floor.
export function AdminConfirmModal({
  open,
  onClose,
  onConfirm,
  title,
  confirmLabel,
  confirmDisabled = false,
  cancelLabel = "Cancel",
  escDisabled = false,
  children,
}: AdminConfirmModalProps) {
  return (
    <Modal
      isOpen={open}
      onClose={onClose}
      title={title}
      escDisabled={escDisabled}
    >
      <div className="space-y-5">
        {children}
        <div className={FOOTER_CLS}>
          <button type="button" onClick={onClose} className={CANCEL_BTN_CLS}>
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={confirmDisabled}
            className={SUBMIT_BTN_CLS}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </Modal>
  );
}
