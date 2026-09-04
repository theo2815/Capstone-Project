"use client";

import { useState } from "react";
import {
  type Flag,
  type FlagStatus,
  canDismiss,
  canEscalate,
  canHide,
  flagEventName,
  flagReasonLabel,
} from "@/lib/admin-flags";
import { useFlagActions } from "@/hooks/use-admin-data";
import { useToast } from "@/hooks/use-toast";
import { syntheticCoverGradient } from "@/lib/admin-photographer-view";
import { AdminStatusPill, type AdminStatusPillTone } from "./admin-status-pill";
import { AdminFlagHideModal } from "./admin-flag-hide-modal";
import { AdminEscalateModal } from "./admin-escalate-modal";
import { formatLongDate } from "@/lib/format";
import {
  BTN_DANGER,
  BTN_GHOST,
  BTN_SECONDARY,
  BTN_SIZE,
} from "@/components/ui/button-styles";

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

const DONE_LABEL = {
  hide: "Hidden",
  dismiss: "Dismissed",
  escalate: "Escalated",
} as const;

const STATUS_LABEL: Record<FlagStatus, string> = {
  open: "Open",
  resolved: "Resolved",
  hidden: "Hidden",
  dismissed: "Dismissed",
  escalated: "Escalated",
};

function statusTone(status: FlagStatus): AdminStatusPillTone {
  switch (status) {
    case "open":
      return "amber";
    case "resolved":
      return "fresh";
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
  const { hide, dismiss, escalate } = useFlagActions();

  const [hideOpen, setHideOpen] = useState(false);
  const [escalateOpen, setEscalateOpen] = useState(false);
  const [confirmDismiss, setConfirmDismiss] = useState(false);

  const eventName = flagEventName(flag);
  const cover = syntheticCoverGradient(flag.id);
  const isOpen = flag.status === "open";
  const actionable = canDismiss(flag);

  async function run(
    verb: "hide" | "dismiss" | "escalate",
    action: Promise<void>,
  ) {
    try {
      await action;
      showToast({ kind: "info", message: `${DONE_LABEL[verb]} · ${flag.id}` });
    } catch {
      showToast({
        kind: "error",
        message: `Could not ${verb} ${flag.id} — nothing changed. Try again.`,
      });
    }
  }
  function handleHide(reason: string | null) {
    setHideOpen(false);
    void run("hide", hide(flag.id, reason));
  }
  function handleDismiss() {
    setConfirmDismiss(false);
    void run("dismiss", dismiss(flag.id));
  }
  function handleEscalate(note: string | null) {
    setEscalateOpen(false);
    void run("escalate", escalate(flag.id, note));
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
      {flag.photoSnapshot.thumbnailUrl ? (
        <div className="aspect-[3/2] border-b border-line overflow-hidden bg-bone-deep relative">
          <img
            src={flag.photoSnapshot.thumbnailUrl}
            alt={flag.photoSnapshot.alt}
            loading="lazy"
            decoding="async"
            className="w-full h-full object-cover"
          />
        </div>
      ) : (
        <div
          className="aspect-[3/2] border-b border-line"
          style={{
            background: `linear-gradient(135deg, ${cover.from}, ${cover.to})`,
          }}
          aria-label={flag.photoSnapshot.alt}
        />
      )}
      <div className="p-5 md:p-6 space-y-4">
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div className="min-w-0">
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate tnum">
              {flag.id}
            </p>
            <h3 className="font-display text-lg md:text-xl font-medium text-ink mt-1">
              {flagReasonLabel(flag.reason)}
            </h3>
          </div>
          <AdminStatusPill
            label={STATUS_LABEL[flag.status]}
            tone={statusTone(flag.status)}
          />
        </div>

        <p className="font-sans text-sm text-ink-soft">{flag.note}</p>

        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft tnum">
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
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
              Reviewer note
            </p>
            <p className="font-sans text-sm text-ink-soft mt-1">
              {flag.reviewerNote}
            </p>
          </div>
        )}

        {actionable && (
          <div className="flex items-center gap-3 flex-wrap pt-2">
            {canHide(flag) && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setHideOpen(true);
                }}
                className={
                  primary ? `${dangerBtn} shadow-[var(--shadow-card)]` : dangerBtn
                }
              >
                Hide…
              </button>
            )}
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setConfirmDismiss((v) => !v);
              }}
              className={tertiaryBtn}
            >
              {confirmDismiss
                ? "Cancel"
                : flag.status === "hidden"
                  ? "Restore photo & dismiss"
                  : "Dismiss"}
            </button>
            {canEscalate(flag) && (
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
            )}
          </div>
        )}

        {confirmDismiss && actionable && (
          <div
            onClick={(e) => e.stopPropagation()}
            className="rounded-xl border border-line bg-bone-deep p-4 space-y-3"
          >
            <p className="font-sans text-sm text-ink-soft">
              {flag.status === "hidden"
                ? "Reopen the photo for runners (unless another hidden flag still targets it) and mark this flag a false alarm."
                : "Mark this flag a false alarm. The photo stays live, no action against the photographer. The dismissal is logged."}
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

const secondaryBtn = `${BTN_SECONDARY} ${BTN_SIZE.sm}`;
const tertiaryBtn = `${BTN_GHOST} ${BTN_SIZE.sm} px-3`;
const dangerBtn = `${BTN_DANGER} ${BTN_SIZE.sm}`;
