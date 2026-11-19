"use client";

import { useRef } from "react";
import { AdminConfirmModal } from "@/components/admin/admin-form-modal";
import { Thumb } from "@/components/admin/flag-relevance-evidence";
import { Kicker } from "@/components/ui/kicker";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { BTN_DANGER, BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { useRelevanceActions, useRelevanceHiddenPhotos } from "@/hooks/use-admin-relevance";
import { useToast } from "@/hooks/use-toast";
import { serverMessage } from "@/lib/admin-flags";
import {
  CLOSED_RESTORE_CONFIRM,
  CONSEQUENCES_TEXT,
  HIDDEN_PHOTOS_EMPTY,
  NO_CONSENT_HIDE_CONFIRM,
  PHOTO_RESTORED,
  PHOTO_RESTORE_FAILED,
  PHOTO_STILL_HIDDEN,
  RELEVANCE_ACTIONS,
  heldByLabel,
  isNoConsent,
  isReleasable,
  photoRestoreLabel,
  restoresHolds,
  type RelevanceCaseAction,
  type RelevanceCaseDetail,
  type RelevanceRestoreResult,
} from "@/lib/admin-relevance";
import { formatLongDate, sentenceCaseDate } from "@/lib/format";

const labelCls = "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft";
const BTN_CLS: Record<RelevanceCaseAction, string> = {
  restore: `${BTN_SECONDARY} ${BTN_SIZE.sm}`,
  keep_hidden: `${BTN_GHOST} ${BTN_SIZE.sm}`,
  looks_fine: `${BTN_SECONDARY} ${BTN_SIZE.sm}`,
  hide_batch: `${BTN_DANGER} ${BTN_SIZE.sm}`,
  end_hold: `${BTN_SECONDARY} ${BTN_SIZE.sm}`,
  hold_payouts: `${BTN_DANGER} ${BTN_SIZE.sm}`,
};

/** The drawer's action bar and its confirm. The host owns `confirming`, so the drawer can stand its ESC down. */
export function RelevanceCaseActions({
  detail,
  kinds,
  confirming,
  onConfirming,
}: {
  detail: RelevanceCaseDetail;
  kinds: RelevanceCaseAction[];
  confirming: RelevanceCaseAction | null;
  onConfirming: (kind: RelevanceCaseAction | null) => void;
}) {
  const { runAction, isPending } = useRelevanceActions();
  const { showToast } = useToast();
  const row = detail.case;
  // Two clicks in one tick both pass the disabled check (isPending lags a render): only the first is sent.
  const sending = useRef(false);

  async function confirm(kind: RelevanceCaseAction) {
    if (sending.current) return;
    sending.current = true;
    const text = RELEVANCE_ACTIONS[kind];
    try {
      const result = await runAction({ kind, caseId: row.caseId });
      const count = kind === "restore" ? ` (${(result as RelevanceRestoreResult).restoredPhotos.toLocaleString()} photos)` : "";
      showToast({ kind: "success", message: `${text.done}: ${row.eventName}${count}` });
      onConfirming(null);
    } catch (error) {
      // The confirm stays open: nothing changed, the admin can read the reason and try again.
      showToast({ kind: "error", message: serverMessage(error) ?? text.failed });
    } finally {
      sending.current = false;
    }
  }

  // A re-read can offer a different set of actions; a confirm for one no longer offered closes.
  const open = confirming && kinds.includes(confirming) && RELEVANCE_ACTIONS[confirming];
  return (
    <>
      {kinds.includes("end_hold") && (
        <p className="basis-full sm:basis-auto sm:flex-1 font-sans text-sm text-ink-soft">{CONSEQUENCES_TEXT.endHoldHint}</p>
      )}
      {kinds.map((kind) => (
        <button key={kind} type="button" disabled={isPending} onClick={() => onConfirming(kind)} className={BTN_CLS[kind]}>
          {RELEVANCE_ACTIONS[kind].label}
        </button>
      ))}
      {confirming && open && (
        <AdminConfirmModal
          open
          onClose={() => onConfirming(null)}
          onConfirm={() => void confirm(confirming)}
          title={open.title}
          confirmLabel={open.confirmLabel}
          tone={confirming === "hide_batch" || confirming === "hold_payouts" ? "danger" : "primary"}
          submitting={isPending}
        >
          <p className="font-sans text-sm text-ink-soft">
            {confirming === "hide_batch" && isNoConsent(row) ? NO_CONSENT_HIDE_CONFIRM
              : confirming === "restore" && detail.stage === "CLOSED" ? CLOSED_RESTORE_CONFIRM : open.confirm}
          </p>
          <p className="font-sans text-sm text-ink">{row.eventName}</p>
        </AdminConfirmModal>
      )}
    </>
  );
}

/** The batch's holds, restored ones greyed, with a single-photo restore while the case is current or closed. */
export function RelevanceHiddenPhotos({ detail }: { detail: RelevanceCaseDetail }) {
  const caseId = detail.case.caseId;
  const query = useRelevanceHiddenPhotos(caseId);
  const { restorePhoto, isPending } = useRelevanceActions();
  const { showToast } = useToast();
  const current = restoresHolds(detail.stage);
  const items = query.items;

  async function handleRestore(photoId: string) {
    try {
      const result = await restorePhoto({ caseId, photoId });
      // Still hidden by another hold is not a success: it stays a plain notice.
      showToast(result.status === "HIDDEN" ? { kind: "info", message: PHOTO_STILL_HIDDEN } : { kind: "success", message: PHOTO_RESTORED });
    } catch (error) {
      showToast({ kind: "error", message: serverMessage(error) ?? PHOTO_RESTORE_FAILED });
    }
  }

  return (
    <section className="space-y-3">
      <Kicker as="p" tone="soft">Held photos</Kicker>
      {query.isLoading ? (
        <p role="status" className="font-sans text-sm text-slate-soft">Loading held photos…</p>
      ) : query.error && items.length === 0 ? (
        <div role="alert" className="space-y-2">
          <p className="font-sans text-sm text-ink-soft">Could not load the held photos.</p>
          <button type="button" onClick={() => query.refetch()} className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`}>Retry</button>
        </div>
      ) : items.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">{HIDDEN_PHOTOS_EMPTY}</p>
      ) : (
        <>
          <ul aria-label="Held photos" className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {items.map((p, i) => (
              <li key={p.id} className={p.restoredAt ? "opacity-50" : undefined}>
                <Thumb url={p.thumbnailUrl} alt={`Held photo ${i + 1} of ${query.total}, ${detail.case.eventName}${p.hiddenAt ? `, hidden ${sentenceCaseDate(formatLongDate(p.hiddenAt))}` : ""}`} />
                <p className={`${labelCls} mt-1 tnum`}>
                  {p.restoredAt ? `Restored ${formatLongDate(p.restoredAt)}` : `Hidden ${formatLongDate(p.hiddenAt)}`}
                  {` · ${heldByLabel(p.heldBy)}`}
                </p>
                {current && isReleasable(p) && (
                  <button
                    type="button"
                    aria-label={photoRestoreLabel(i + 1, formatLongDate(p.hiddenAt))}
                    disabled={isPending}
                    onClick={() => void handleRestore(p.photoId)}
                    className={`${BTN_GHOST} ${BTN_SIZE.sm} px-3 mt-1`}
                  >
                    Restore
                  </button>
                )}
              </li>
            ))}
          </ul>
          <LoadMoreButton
            shown={items.length}
            total={query.total}
            increment={100}
            onLoadMore={query.fetchNextPage}
            isLoading={query.isFetchingNextPage}
            label="Load more"
          />
        </>
      )}
    </section>
  );
}
