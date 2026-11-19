"use client";

import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { useId, useState } from "react";
import {
  type Flag,
  type FlagStatus,
  canDismiss,
  canEscalate,
  canConfirmHold,
  canHide,
  canResolve,
  flagEventName,
  flagReasonLabel,
  relevanceConfirmText,
  type FlagConfirmKind,
} from "@/lib/admin-flags";
import { LogoPlaceholder } from "@/components/ui/logo-placeholder";
import { AdminStatusPill, type AdminStatusPillTone } from "./admin-status-pill";
import { formatLongDate } from "@/lib/format";
import { ROUTES } from "@/lib/constants";
import {
  BTN_DANGER,
  BTN_GHOST,
  BTN_SECONDARY,
  BTN_SIZE,
  ROW_OPEN_BUTTON,
} from "@/components/ui/button-styles";

/** The queue's one action hook and modals, shared by every card (FLG6). */
export interface FlagCardActions {
  isPending: (flagId: string) => boolean;
  pendingAction: (flagId: string) => string | null;
  onHide: (flagId: string) => void;
  onEscalate: (flagId: string) => void;
  /** Runs a confirmed dismiss, hold or resolve; true when it went through. */
  onConfirm: (flagId: string, kind: FlagConfirmKind) => Promise<boolean>;
}

interface AdminFlagCardProps {
  flag: Flag;
  actions: FlagCardActions;
  /** True only for the first open card, which lifts its Hide button. */
  primary?: boolean;
  /**
   * Phase 3: when provided, the title button (which covers the card, AX-4)
   * calls this — typically the drawer host's setRowId.
   * Inline Hide/Dismiss/Escalate keep the fast path so triage doesn't
   * always require opening the drawer.
   */
  onOpen?: () => void;
  /** Phase 4: J/K keyboard focus indicator. Adds an ink ring + offset. */
  focused?: boolean;
}

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

/** One status pill for the card and the drawer. */
export function FlagStatusPill({ status }: { status: FlagStatus }) {
  return <AdminStatusPill label={STATUS_LABEL[status]} tone={statusTone(status)} />;
}

export function AdminFlagCard({
  flag,
  actions,
  primary = false,
  onOpen,
  focused = false,
}: AdminFlagCardProps) {
  const submitting = actions.isPending(flag.id);
  // One inline confirm at a time: hold, resolve or dismiss.
  const [confirm, setConfirm] = useState<FlagConfirmKind | null>(null);
  const confirmDismiss = confirm === "dismiss";
  const relevance = canResolve(flag);
  const toggleConfirm = (kind: FlagConfirmKind) => setConfirm((c) => (c === kind ? null : kind));

  const eventName = flagEventName(flag);
  const isOpen = flag.status === "open";
  const actionable = canDismiss(flag);

  async function handleConfirm(kind: FlagConfirmKind) {
    if (await actions.onConfirm(flag.id, kind)) setConfirm(null);
  }

  const interactive = onOpen !== undefined;
  const idLineId = useId();

  return (
    <article
      data-row-id={flag.id}
      className={`relative rounded-2xl border border-line bg-bone overflow-hidden ${
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
        <LogoPlaceholder size="lg" label={flag.photoSnapshot.alt} className="aspect-[3/2] border-b border-line" />
      )}
      <div className="p-5 md:p-6 space-y-4">
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div className="min-w-0">
            <Kicker as="p" id={idLineId} className="tnum">
              {flag.id}
            </Kicker>
            <h3 className="font-display text-lg md:text-xl font-medium text-ink mt-1">
              {interactive ? (
                <button type="button" onClick={onOpen} aria-describedby={idLineId} className={`${ROW_OPEN_BUTTON} after:rounded-2xl`}>
                  {flagReasonLabel(flag.reason)}
                </button>
              ) : (
                flagReasonLabel(flag.reason)
              )}
            </h3>
          </div>
          <FlagStatusPill status={flag.status} />
        </div>

        <p className="font-sans text-sm text-ink-soft">{flag.note}</p>

        {flag.payoutHoldAt && (
          <p className="font-sans text-sm font-medium text-ink tnum">
            Payouts on hold since {formatLongDate(flag.payoutHoldAt)}
          </p>
        )}

        <Kicker as="p" tone="soft" className="tnum">
          {flag.reportedBy === "system"
            ? "System"
            : `@${flag.reportedBy}`}
          <span className="text-slate-soft"> · </span>
          {flag.targetKind === "user" ? (
            <Link
              href={`${ROUTES.ADMIN_PHOTOGRAPHERS}/${encodeURIComponent(flag.photographerHandle)}`}
              className="relative z-10 underline underline-offset-2 hover:text-ink"
            >
              @{flag.photographerHandle}
            </Link>
          ) : (
            <>@{flag.photographerHandle}</>
          )}
          <span className="text-slate-soft"> · </span>
          {/* A photographer flag has no event; photo and event flags always name one. */}
          {(flag.targetKind !== "user" || flag.eventName) && (
            <>
              {eventName}
              <span className="text-slate-soft"> · </span>
            </>
          )}
          {formatLongDate(flag.reportedAt)}
        </Kicker>

        {!isOpen && flag.reviewerNote && (
          <div className="rounded-xl border border-line bg-bone-deep p-4">
            <Kicker as="p" tone="soft">
              Reviewer note
            </Kicker>
            <p className="font-sans text-sm text-ink-soft mt-1">
              {flag.reviewerNote}
            </p>
          </div>
        )}

        {submitting && <p role="status" className="font-sans text-sm text-slate">{actions.pendingAction(flag.id)}</p>}
        {actionable && (
          <div className="relative z-10 flex items-center gap-3 flex-wrap pt-2">
            {canHide(flag) && (
              <button
                type="button"
                disabled={submitting}
                onClick={() => actions.onHide(flag.id)}
                className={
                  primary ? `${dangerBtn} shadow-[var(--shadow-card)]` : dangerBtn
                }
              >
                Hide…
              </button>
            )}
            {canConfirmHold(flag) && (
              <button
                type="button"
                disabled={submitting}
                onClick={() => toggleConfirm("hold")}
                className={dangerBtn}
              >
                {confirm === "hold" ? "Cancel" : "Hold payouts"}
              </button>
            )}
            {relevance && (
              <button
                type="button"
                disabled={submitting}
                onClick={() => toggleConfirm("resolve")}
                className={tertiaryBtn}
              >
                {confirm === "resolve" ? "Cancel" : "Resolve"}
              </button>
            )}
            <button
              type="button"
                disabled={submitting}
              onClick={() => toggleConfirm("dismiss")}
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
                disabled={submitting}
                onClick={() => actions.onEscalate(flag.id)}
                className={tertiaryBtn}
              >
                Escalate…
              </button>
            )}
          </div>
        )}

        {confirm && actionable && (
          <div className="relative z-10 rounded-xl border border-line bg-bone-deep p-4 space-y-3">
            <p className="font-sans text-sm text-ink-soft">
              {relevance || confirm === "hold"
                ? relevanceConfirmText(confirm)
                : flag.status === "hidden"
                  ? "Reopen the photo for runners (unless another hidden flag still targets it) and mark this flag a false alarm."
                  : "Mark this flag a false alarm. The photo stays live, no action against the photographer. The dismissal is logged."}
            </p>
            <button
              type="button"
                disabled={submitting}
              onClick={() => void handleConfirm(confirm)}
              className={`${secondaryBtn} w-full`}
            >
              {confirm === "hold" ? "Confirm hold" : confirm === "resolve" ? "Confirm resolve" : "Confirm dismiss"}
            </button>
          </div>
        )}
      </div>
    </article>
  );
}

const secondaryBtn = `${BTN_SECONDARY} ${BTN_SIZE.sm}`;
const tertiaryBtn = `${BTN_GHOST} ${BTN_SIZE.sm} px-3`;
const dangerBtn = `${BTN_DANGER} ${BTN_SIZE.sm}`;
