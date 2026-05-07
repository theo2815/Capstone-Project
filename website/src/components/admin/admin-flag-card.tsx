"use client";

import { useState } from "react";
import {
  type Flag,
  type FlagStatus,
  FLAG_REASON_LABEL,
} from "@/lib/admin-flags";
import { useAdminFlagStore } from "@/store/admin-flag-store";
import { useToast } from "@/hooks/use-toast";
import { getEventById } from "@/lib/event-catalog";
import { syntheticCoverGradient } from "@/lib/admin-photographer-view";
import { AdminStatusPill, type AdminStatusPillTone } from "./admin-status-pill";
import { AdminFlagHideModal } from "./admin-flag-hide-modal";
import { AdminEscalateModal } from "./admin-escalate-modal";
import { formatLongDate } from "@/lib/format";

interface AdminFlagCardProps {
  flag: Flag;
  /** True only for the first open card on the page so its Hide button is the
   *  page's single fresh accent. Subsequent open cards get bordered Hide. */
  primary?: boolean;
  /**
   * Phase 3: when provided, clicking the card body (not the inline
   * action buttons) calls this — typically the drawer host's setRowId.
   * Inline Hide/Dismiss/Escalate keep the fast path so triage doesn't
   * always require opening the drawer.
   */
  onOpen?: () => void;
  /** Phase 4: J/K keyboard focus indicator. Adds an ink ring + offset. */
  focused?: boolean;
}

const STATUS_LABEL: Record<FlagStatus, string> = {
  open: "Open",
  hidden: "Hidden",
  dismissed: "Dismissed",
  escalated: "Escalated",
};

function statusTone(status: FlagStatus): AdminStatusPillTone {
  switch (status) {
    case "open":
      return "amber";
    case "hidden":
      return "ink";
    case "dismissed":
      return "slate";
    case "escalated":
      return "muted";
  }
}

export function AdminFlagCard({
  flag,
  primary = false,
  onOpen,
  focused = false,
}: AdminFlagCardProps) {
  const { showToast } = useToast();
  const hide = useAdminFlagStore((s) => s.hide);
  const dismiss = useAdminFlagStore((s) => s.dismiss);
  const escalate = useAdminFlagStore((s) => s.escalate);

  const [hideOpen, setHideOpen] = useState(false);
  const [escalateOpen, setEscalateOpen] = useState(false);
  const [confirmDismiss, setConfirmDismiss] = useState(false);

  const event = getEventById(flag.eventId);
  const eventName = event?.name ?? `Event ${flag.eventId}`;
  const cover = syntheticCoverGradient(flag.id);
  const isOpen = flag.status === "open";

  function handleHide(reason: string | null) {
    hide(flag.id, reason);
    setHideOpen(false);
    showToast({ kind: "info", message: `Hidden · ${flag.id}` });
  }
  function handleDismiss() {
    dismiss(flag.id);
    setConfirmDismiss(false);
    showToast({ kind: "info", message: `Dismissed · ${flag.id}` });
  }
  function handleEscalate(note: string | null) {
    escalate(flag.id, note);
    setEscalateOpen(false);
    showToast({ kind: "info", message: `Escalated · ${flag.id}` });
  }

  function handleArticleClick() {
    if (onOpen) onOpen();
  }
  function handleArticleKeyDown(e: React.KeyboardEvent<HTMLElement>) {
    if (!onOpen) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onOpen();
    }
  }

  const interactive = onOpen !== undefined;

  return (
    <article
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : undefined}
      data-row-id={flag.id}
      onClick={interactive ? handleArticleClick : undefined}
      onKeyDown={interactive ? handleArticleKeyDown : undefined}
      aria-label={interactive ? `Open flag ${flag.id}` : undefined}
      className={`rounded-2xl border border-line bg-bone overflow-hidden ${
        interactive
          ? "cursor-pointer transition-colors hover:border-ink"
          : ""
      } ${focused ? "ring-2 ring-ink ring-offset-2 ring-offset-bone" : ""}`}
    >
      <div
        className="aspect-[3/2] border-b border-line"
        style={{
          background: `linear-gradient(135deg, ${cover.from}, ${cover.to})`,
        }}
        aria-label={flag.photoSnapshot.alt}
      />
      <div className="p-5 md:p-6 space-y-4">
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div className="min-w-0">
            <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate tnum">
              {flag.id}
            </p>
            <h3 className="font-display text-lg md:text-xl font-medium text-ink mt-1">
              {FLAG_REASON_LABEL[flag.reason]}
            </h3>
          </div>
          <AdminStatusPill
            label={STATUS_LABEL[flag.status]}
            tone={statusTone(flag.status)}
          />
        </div>

        <p className="font-sans text-sm text-ink-soft">{flag.note}</p>

        <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft tnum">
          {flag.reportedBy === "system"
            ? "System"
            : `@${flag.reportedBy}`}
          <span className="text-slate-soft"> · </span>
          @{flag.photographerHandle}
          <span className="text-slate-soft"> · </span>
          {eventName}
          <span className="text-slate-soft"> · </span>
          {formatLongDate(flag.reportedAt)}
        </p>

        {!isOpen && flag.reviewerNote && (
          <div className="rounded-xl border border-line bg-bone-deep p-4">
            <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
              Reviewer note
            </p>
            <p className="font-sans text-sm text-ink-soft mt-1">
              {flag.reviewerNote}
            </p>
          </div>
        )}

        {isOpen && (
          <div className="flex items-center gap-3 flex-wrap pt-2">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setHideOpen(true);
              }}
              className={primary ? primaryFreshBtn : secondaryBtn}
            >
              Hide…
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setConfirmDismiss((v) => !v);
              }}
              className={tertiaryBtn}
            >
              {confirmDismiss ? "Cancel dismiss" : "Dismiss"}
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setEscalateOpen(true);
              }}
              className={tertiaryBtn}
            >
              Escalate…
            </button>
          </div>
        )}

        {confirmDismiss && isOpen && (
          <div
            onClick={(e) => e.stopPropagation()}
            className="rounded-xl border border-line bg-bone-deep p-4 space-y-3"
          >
            <p className="font-sans text-sm text-ink-soft">
              Mark this flag a false alarm. The photo stays live, no action
              against the photographer. The dismissal is logged.
            </p>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                handleDismiss();
              }}
              className={`${secondaryBtn} w-full`}
            >
              Confirm dismiss
            </button>
          </div>
        )}
      </div>

      <AdminFlagHideModal
        open={hideOpen}
        onClose={() => setHideOpen(false)}
        onSubmit={handleHide}
        flagId={flag.id}
      />
      <AdminEscalateModal
        open={escalateOpen}
        onClose={() => setEscalateOpen(false)}
        onSubmit={handleEscalate}
        targetLabel={`Escalate ${flag.id}`}
        body="Push this flag to the next review tier. The photo stays live in the meantime; higher tier decides hide vs. dismiss."
      />
    </article>
  );
}

const primaryFreshBtn =
  "font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] bg-fresh text-bone hover:bg-fresh-deep transition-colors rounded-full px-5 py-2";

const secondaryBtn =
  "font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink border border-line hover:bg-ink hover:text-bone hover:border-ink transition-colors rounded-full px-5 py-2";

const tertiaryBtn =
  "font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors px-3 py-2";
