"use client";

import { useRef, useState } from "react";
import { useAdminRemovalActions, useAdminRemovalRequest } from "@/hooks/use-photo-removal";
import { useConfirmation } from "@/hooks/use-confirmation";
import { AdminFieldLabel } from "@/components/admin/admin-form-fields";
import { FieldError } from "@/components/ui/field-error";
import { BTN_DANGER, BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { formatLongDate } from "@/lib/format";
import { cn } from "@/lib/utils";
import { REMOVAL_INPUT } from "./request-intake";
import { ReviewCurrentState, groupByOwner } from "./request-review";
import type {
  PhotoRemovalItemDto, PhotoRemovalReviewDto, PhotographerRemovalResponse, RemovalDecision,
  RemovalEvidenceCleanupState, RemovalItemState,
} from "@/types/photo-removal";

type Draft = { decision: RemovalDecision | ""; reason: string; reviewDate: string };
const EMPTY: Draft = { decision: "", reason: "", reviewDate: "" };
const DECIDABLE = new Set<RemovalItemState>(["VERIFIED", "RETAINED_RESTRICTED"]);
const DECISIONS: [RemovalDecision, string][] = [
  ["REMOVE", "Remove"], ["RETAIN", "Retain"], ["MORE_EVIDENCE", "Ask for more evidence"], ["RETAIN_RESTRICTED", "Retain restricted (stays hidden)"],
];
const RESPONSE: Record<PhotographerRemovalResponse, string> = { AGREE: "agreed", OBJECT: "objected", ALREADY_DELETED: "already deleted" };
// Only COMPLETE is proof of erasure; anything else is shown as what it is.
const CLEANUP: Record<string, string> = {
  PENDING: "cleanup pending", RETRY_REQUIRED: "cleanup failed", UNCONFIRMED: "cleanup unconfirmed", COMPLETE: "erasure complete",
};
const EVIDENCE: Record<RemovalEvidenceCleanupState, string> = {
  NONE: "No selfie evidence on file.", PRESENT: "Selfie evidence on file (private to admins).", PENDING: "Evidence deletion pending.",
  DELETED: "Evidence deleted.", RETRY_REQUIRED: "Evidence deletion blocked — it has not been deleted yet.",
};
const errorText = (error: unknown) => error instanceof Error ? error.message : "Something went wrong. Try again.";

function itemStatus(item: PhotoRemovalItemDto): string {
  switch (item.state) {
    case "REMOVAL_AUTHORIZED":
    case "REMOVED":
      return `${item.state === "REMOVED" ? "Removed" : "Removal authorized"} · ${CLEANUP[item.cleanupState ?? ""] ?? "cleanup not started"}`;
    case "RETAINED_RESTRICTED":
      return `Retained restricted · still hidden, not erased${item.retentionReviewAt ? ` · review ${formatLongDate(item.retentionReviewAt)}` : ""}`;
    case "RETAINED": return "Retained";
    case "VERIFIED": return item.adminDecision === "MORE_EVIDENCE" ? "Hidden · more evidence requested" : "Hidden · awaiting decision";
    case "EXCLUDED": return "Excluded";
    default: return "Candidate";
  }
}

function ownerState(items: PhotoRemovalItemDto[], review: PhotoRemovalReviewDto | undefined): string {
  if (!review || review.unavailableReason) return "unavailable";
  const responses = new Set(items.map((item) => item.ownerResponse));
  if (responses.size > 1) return "mixed responses";
  const [response] = responses;
  return response ? RESPONSE[response] : review.overdue ? "overdue" : "no response";
}

// Mirrors the backend rule: removing without the owner's agreement before the
// server-marked deadline (or with no reachable owner) is an urgent override.
function needsUrgentReason(item: PhotoRemovalItemDto, review: PhotoRemovalReviewDto | undefined) {
  return item.ownerResponse !== "AGREE" && (!review || !!review.unavailableReason || !review.overdue);
}

export function RemovalRequestDecision({ requestId }: { requestId: string }) {
  const detail = useAdminRemovalRequest(requestId);
  const actions = useAdminRemovalActions();
  const { confirm } = useConfirmation();
  const [drafts, setDrafts] = useState(new Map<string, Draft>());
  const [urgentReason, setUrgentReason] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  // A retry of the same action after a failure reuses its command ID; any changed action gets a new one.
  const commands = useRef(new Map<string, string>());
  const commandFor = (fingerprint: string) => {
    if (!commands.current.has(fingerprint)) commands.current.set(fingerprint, crypto.randomUUID());
    return commands.current.get(fingerprint)!;
  };

  const request = detail.data;
  if (!request || request.status === "DRAFT") return null;
  const id = request.id;
  const reviews = new Map(request.reviews.map((review) => [review.id, review]));
  const reviewOf = (item: PhotoRemovalItemDto) => item.reviewId ? reviews.get(item.reviewId) : undefined;
  const chosen = request.items.flatMap((item) => {
    const draft = drafts.get(item.id);
    return draft?.decision && DECIDABLE.has(item.state) ? [{ item, ...draft, decision: draft.decision }] : [];
  });
  const hasRemove = chosen.some((entry) => entry.decision === "REMOVE");
  const urgentRequired = chosen.some((entry) => entry.decision === "REMOVE" && needsUrgentReason(entry.item, reviewOf(entry.item)));
  const incomplete = chosen.some((entry) => !entry.reason.trim() || (entry.decision === "RETAIN_RESTRICTED" && !entry.reviewDate));
  const reviewRequired = actions.reviewRequired(id);
  const busy = actions.decide.isPending || actions.retryCleanup.isPending;
  const canSubmit = chosen.length > 0 && chosen.length <= 100 && !incomplete && (!urgentRequired || !!urgentReason.trim()) &&
    !busy && !detail.isFetching && !reviewRequired;

  const setDraft = (itemId: string, change: Partial<Draft>) =>
    setDrafts((current) => new Map(current).set(itemId, { ...(current.get(itemId) ?? EMPTY), ...change }));

  async function submit() {
    if (!canSubmit || !request) return;
    setError(null); setNotice(null);
    if (hasRemove && !(await confirm({
      title: "Authorize removal?",
      message: "The selected photos come off sale, search and buyer downloads, and their files are queued for erasure. Buyers keep their receipts and refund access.",
      confirmLabel: "Authorize removal", danger: true,
    }))) return;
    const body = {
      expectedVersion: request.version,
      decisions: chosen.map(({ item, decision, reason, reviewDate }) => ({
        itemId: item.id, decision, reason: reason.trim(),
        ...(decision === "RETAIN_RESTRICTED" ? { retentionReviewAt: new Date(`${reviewDate}T00:00`).toISOString() } : {}),
      })),
      ...(urgentRequired ? { urgentReason: urgentReason.trim() } : {}),
    };
    const fingerprint = `decide:${JSON.stringify(body)}`;
    try {
      await actions.decide.mutateAsync({ id, command: { ...body, commandId: commandFor(fingerprint) } });
      commands.current.delete(fingerprint);
      setDrafts(new Map()); setUrgentReason(""); setNotice("Decisions recorded.");
    } catch (failure) {
      setError(errorText(failure));
    }
  }

  async function retryCleanup(item: PhotoRemovalItemDto) {
    if (!request) return;
    setError(null); setNotice(null);
    const body = { expectedVersion: request.version, itemIds: [item.id] };
    const fingerprint = `retry:${JSON.stringify(body)}`;
    try {
      await actions.retryCleanup.mutateAsync({ id, command: { ...body, commandId: commandFor(fingerprint) } });
      commands.current.delete(fingerprint);
      setNotice("Cleanup retry requested.");
    } catch (failure) {
      setError(errorText(failure));
    }
  }

  return (
    <section aria-labelledby="removal-decisions" className="mt-12 space-y-8 border-t border-line pt-8">
      <div className="space-y-2">
        <h2 id="removal-decisions" className="font-display text-xl font-extrabold text-ink">Owner responses and decisions</h2>
        <p className="text-sm text-slate">QuickPitik decides each photo. An owner&apos;s response is input, not a veto or consent, and silence after the deadline means overdue — never approval.</p>
      </div>
      <ReviewCurrentState request={request} actions={actions} refreshing={detail.isFetching} refresh={detail.refetch} />

      {groupByOwner(request.items, (item) => [item.reviewId, item.ownerNameSnapshot]).map(([key, owner, items]) => {
        const review = key ? reviews.get(key) : undefined;
        return (
          <div key={key} className="space-y-3">
            <h3 className="font-display font-bold text-ink">{key ? `${owner} — ${ownerState(items, review)}` : `${owner} — no owner on record`}</h3>
            {review && (
              <p className="text-sm text-slate">
                {review.notifiedAt ? <>Notice stored <span className="tnum">{formatLongDate(review.notifiedAt)}</span></> : "Notice not stored yet"}
                {review.deadline && <> · Respond by <span className="tnum">{formatLongDate(review.deadline)}</span></>}
                {review.overdue && <> · Overdue{review.overdueAt && <> since <span className="tnum">{formatLongDate(review.overdueAt)}</span></>}</>}
                {review.unavailableReason && <> · Owner unavailable: {review.unavailableReason}</>}
              </p>
            )}
            <ul className="space-y-3">
              {items.map((item) => (
                <ItemRow key={item.id} item={item} draft={drafts.get(item.id) ?? EMPTY} busy={busy || reviewRequired}
                  onChange={(change) => setDraft(item.id, change)} onRetry={() => void retryCleanup(item)} />
              ))}
            </ul>
          </div>
        );
      })}

      {chosen.length > 0 && (
        <div className="space-y-4">
          {urgentRequired && (
            <div className="space-y-2">
              <AdminFieldLabel htmlFor="removal-urgent">Urgent override reason</AdminFieldLabel>
              <p className="text-sm text-amber-700">An urgent reason is required: a photo&apos;s owner has not agreed and their review deadline has not passed, or there is no reachable owner.</p>
              <textarea id="removal-urgent" rows={2} maxLength={2000} value={urgentReason} disabled={busy}
                onChange={(e) => setUrgentReason(e.target.value)} className={REMOVAL_INPUT} />
            </div>
          )}
          {chosen.length > 100 && <p className="text-sm text-amber-700">Decide at most <span className="tnum">100</span> photos at a time.</p>}
          <button type="button" disabled={!canSubmit} onClick={() => void submit()}
            className={cn(hasRemove ? BTN_DANGER : BTN_PRIMARY, BTN_SIZE.md)}>
            {actions.decide.isPending ? "Saving…" : hasRemove ? "Authorize removal" : "Record decision"}
          </button>
        </div>
      )}
      <FieldError message={error} id="removal-decision-error" />
      {notice && <p role="status" className="text-sm text-ink">{notice}</p>}

      <div className="space-y-2 border-t border-line pt-6 text-sm">
        <h3 className="font-display font-bold text-ink">Evidence and copies</h3>
        <p className="text-ink">{EVIDENCE[request.evidenceCleanupState]}</p>
        {request.outstandingEvidenceCount > 0 && (
          <p className="text-slate"><span className="tnum">{request.outstandingEvidenceCount}</span> evidence file{request.outstandingEvidenceCount === 1 ? "" : "s"} still awaiting deletion.</p>
        )}
        <p className="text-slate">Removal covers QuickPitik only. QuickPitik cannot recall photos that buyers already downloaded, screenshots, or copies posted on other sites.</p>
      </div>
    </section>
  );
}

function ItemRow({ item, draft, busy, onChange, onRetry }: {
  item: PhotoRemovalItemDto; draft: Draft; busy: boolean; onChange: (change: Partial<Draft>) => void; onRetry: () => void;
}) {
  const photo = item.photoIdSnapshot;
  const retryable = item.cleanupState === "RETRY_REQUIRED" && (item.state === "REMOVAL_AUTHORIZED" || item.state === "REMOVED");
  return (
    <li className="rounded-xl border border-line p-4 space-y-2 text-sm">
      <p className="flex flex-wrap items-baseline gap-x-3">
        <span className="font-mono tnum text-ink">{photo}</span>
        <span className="text-ink">{itemStatus(item)}</span>
      </p>
      {item.ownerResponse ? (
        <p className="text-slate">
          Owner {RESPONSE[item.ownerResponse]}
          {item.ownerRespondedAt && <> · <span className="tnum">{formatLongDate(item.ownerRespondedAt)}</span></>}
          {item.ownerResponseReason && <> — {item.ownerResponseReason}</>}
        </p>
      ) : item.reviewId && <p className="text-slate">No owner response recorded.</p>}
      {item.adminDecision && (
        <p className="text-slate">
          Decision: {item.adminDecision}{item.adminDecisionReason && <> — {item.adminDecisionReason}</>}
          {item.adminDecidedAt && <> · <span className="tnum">{formatLongDate(item.adminDecidedAt)}</span></>}
        </p>
      )}
      {retryable && (
        <button type="button" disabled={busy} onClick={onRetry} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>
          Retry cleanup<span className="sr-only"> for {photo}</span>
        </button>
      )}
      {DECIDABLE.has(item.state) && (
        <div className="grid gap-3 sm:grid-cols-2 pt-1">
          <div className="space-y-1">
            <AdminFieldLabel htmlFor={`decision-${item.id}`}>Decision<span className="sr-only"> for {photo}</span></AdminFieldLabel>
            <select id={`decision-${item.id}`} value={draft.decision} disabled={busy} className={REMOVAL_INPUT}
              onChange={(e) => onChange({ decision: e.target.value as Draft["decision"] })}>
              <option value="">No decision</option>
              {DECISIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </div>
          {draft.decision && (
            <div className="space-y-1">
              <AdminFieldLabel htmlFor={`reason-${item.id}`}>Reason<span className="sr-only"> for {photo}</span></AdminFieldLabel>
              <input id={`reason-${item.id}`} type="text" maxLength={2000} value={draft.reason} disabled={busy} className={REMOVAL_INPUT}
                aria-invalid={!draft.reason.trim()} onChange={(e) => onChange({ reason: e.target.value })} />
            </div>
          )}
          {draft.decision === "RETAIN_RESTRICTED" && (
            <div className="space-y-1">
              <AdminFieldLabel htmlFor={`review-${item.id}`}>Retention review date<span className="sr-only"> for {photo}</span></AdminFieldLabel>
              <input id={`review-${item.id}`} type="date" value={draft.reviewDate} disabled={busy} className={REMOVAL_INPUT}
                aria-invalid={!draft.reviewDate} onChange={(e) => onChange({ reviewDate: e.target.value })} />
            </div>
          )}
        </div>
      )}
    </li>
  );
}
