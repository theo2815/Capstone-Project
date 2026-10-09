"use client";

import Link from "next/link";
import { useRef, useState } from "react";
import { usePhotographerRemovalActions, usePhotographerRemovalRequest, usePhotographerRemovalRequests } from "@/hooks/use-photo-removal";
import { Kicker } from "@/components/ui/kicker";
import { FieldError } from "@/components/ui/field-error";
import { BTN_GHOST, BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ApiError } from "@/lib/api";
import { ROUTES } from "@/lib/constants";
import { formatLongDate, formatSalesDate } from "@/lib/format";
import { cn } from "@/lib/utils";
import type {
  PhotographerRemovalItemDto, PhotographerRemovalRequestDto, PhotographerRemovalResponse, RemovalRequestStatus,
} from "@/types/photo-removal";

// Photographer view of their own removal case. It renders only the photographer
// DTO (own items, sanitized reason) — never admin routes, runner contact or evidence.
const PAGE = 20;
const STATUS: Record<RemovalRequestStatus, string> = {
  DRAFT: "Being prepared", OPEN: "Under review", RESOLVING: "Decided · finishing up", RESOLVED: "Decided", CANCELLED: "Cancelled",
};
const RESPONSES: [PhotographerRemovalResponse, string, string][] = [
  ["AGREE", "Agree to removal", "You agree these photos can be removed. QuickPitik still decides."],
  ["OBJECT", "Object", "Tell QuickPitik why the photos should stay. An objection is not a veto."],
  ["ALREADY_DELETED", "Already deleted", "You already deleted them. QuickPitik checks this against the platform."],
];
const RESPONDED: Record<PhotographerRemovalResponse, string> = {
  AGREE: "You agreed", OBJECT: "You objected", ALREADY_DELETED: "You said it was already deleted",
};
const INPUT = "w-full rounded-2xl border border-line bg-surface px-4 py-3 font-sans text-sm text-ink focus:outline-none focus:border-fresh";
const errorText = (error: unknown) => error instanceof Error ? error.message : "Something went wrong. Try again.";
const requestHref = (id: string) => `${ROUTES.DASHBOARD_PHOTO_REMOVAL_REQUESTS}/${encodeURIComponent(id)}`;

// Mirrors the server's "awaiting decision" rule; a sent response stays read-only here.
function canRespond(request: PhotographerRemovalRequestDto, item: PhotographerRemovalItemDto) {
  return request.status === "OPEN" && item.response === null &&
    (item.ownerReviewPending || (item.state === "VERIFIED" && (item.decision === null || item.decision === "MORE_EVIDENCE")));
}

// Only server cleanup COMPLETE counts as erased.
function outcome(item: PhotographerRemovalItemDto): string {
  switch (item.state) {
    case "REMOVAL_AUTHORIZED":
    case "REMOVED":
      return item.cleanupState === "COMPLETE" ? "Removed · erased from QuickPitik"
        : item.cleanupState === "RETRY_REQUIRED" ? "Removal authorized · cleanup delayed, QuickPitik is retrying"
        : item.cleanupState === "UNCONFIRMED" ? "Removal authorized · cleanup not yet confirmed"
        : "Removal authorized · cleanup still running";
    case "RETAINED_RESTRICTED":
      return `Kept restricted · still hidden, not erased${item.retentionReviewAt ? ` · next review ${formatLongDate(item.retentionReviewAt)}` : ""}`;
    case "RETAINED": return "Kept";
    case "EXCLUDED": return "Not part of this request";
    default: return item.decision === "MORE_EVIDENCE" ? "Hidden · QuickPitik asked for more evidence" : "Hidden during review";
  }
}

export function PhotographerRemovalRequestList() {
  const [offset, setOffset] = useState(0);
  const list = usePhotographerRemovalRequests({ offset, limit: PAGE });
  const rows = list.data?.items ?? [];
  const total = list.data?.total ?? 0;
  if (list.isPending) return <p role="status" className="mt-8 text-slate">Loading requests…</p>;
  if (list.isError) return (
    <div role="alert" className="mt-8 space-y-4">
      <p>Couldn&apos;t load removal requests.</p>
      <button type="button" className={cn(BTN_SECONDARY, BTN_SIZE.sm)} onClick={() => void list.refetch()}>Try again</button>
    </div>
  );
  return (
    <div className="mt-8 space-y-4">
      {rows.length === 0 && <p className="text-slate">No removal requests involve your photos.</p>}
      <ul className="divide-y divide-line border-y border-line">
        {rows.map((row) => {
          const waiting = row.items.some((item) => canRespond(row, item));
          return (
            <li key={row.id}>
              <Link href={requestHref(row.id)} className="flex flex-wrap items-center gap-x-4 gap-y-1 py-4 px-2 hover:bg-bone-deep/50">
                <span className="min-w-0 flex-1">
                  <span className="block font-display font-bold text-ink truncate">{row.event.name}</span>
                  <span className="block text-sm text-slate">{waiting ? "Needs your response" : STATUS[row.status]}</span>
                </span>
                <span className="font-mono tnum text-sm text-slate">{row.items.length} photo{row.items.length === 1 ? "" : "s"} · {formatLongDate(row.createdAt)}</span>
              </Link>
            </li>
          );
        })}
      </ul>
      {total > PAGE && (
        <div className="flex items-center gap-2">
          <button type="button" className={cn(BTN_GHOST, BTN_SIZE.sm)} disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>← Previous</button>
          <span className="font-mono tnum text-sm text-slate">{offset + 1}–{Math.min(offset + PAGE, total)} of {total}</span>
          <button type="button" className={cn(BTN_GHOST, BTN_SIZE.sm)} disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>Next →</button>
        </div>
      )}
    </div>
  );
}

export function PhotographerPhotoRemovalRequest({ requestId }: { requestId: string }) {
  const detail = usePhotographerRemovalRequest(requestId);
  const actions = usePhotographerRemovalActions();
  const [selected, setSelected] = useState(new Set<string>());
  const [response, setResponse] = useState<PhotographerRemovalResponse | "">("");
  const [reason, setReason] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [reviewStatus, setReviewStatus] = useState<string | null>(null);
  // A retry of the same response after a failure reuses its command ID; any changed response gets a new one.
  const commands = useRef(new Map<string, string>());

  if (detail.isPending) return <p role="status" className="mt-8 text-slate">Loading request…</p>;
  // A failed background refresh keeps showing the last server copy.
  if (!detail.data) return (
    <div role="alert" className="mt-8 space-y-4">
      {detail.error instanceof ApiError && detail.error.status === 404
        ? <p>This removal request isn&apos;t available to your account.</p>
        : <>
          <p>Couldn&apos;t load this removal request.</p>
          <button type="button" className={cn(BTN_SECONDARY, BTN_SIZE.sm)} onClick={() => void detail.refetch()}>Try again</button>
        </>}
    </div>
  );

  const request = detail.data;
  const id = request.id;
  const open = request.items.filter((item) => canRespond(request, item));
  // Bulk responses target only currently displayed, still-open own items.
  const itemIds = open.filter((item) => selected.has(item.id)).map((item) => item.id);
  const reviewRequired = actions.reviewRequired(id);
  const busy = actions.respond.isPending;
  const reasonMissing = response === "OBJECT" && !reason.trim();
  const canSubmit = itemIds.length > 0 && itemIds.length <= 100 && !!response && !reasonMissing && !busy && !detail.isFetching && !reviewRequired;

  const toggle = (itemId: string) => setSelected((current) => {
    const next = new Set(current);
    if (!next.delete(itemId)) next.add(itemId);
    return next;
  });
  const allSelected = open.length > 0 && open.every((item) => selected.has(item.id));

  async function submit() {
    if (!canSubmit || !response) return;
    setError(null); setNotice(null);
    const body = { expectedVersion: request.version, itemIds, response, ...(response === "OBJECT" ? { reason: reason.trim() } : {}) };
    const fingerprint = JSON.stringify(body);
    if (!commands.current.has(fingerprint)) commands.current.set(fingerprint, crypto.randomUUID());
    try {
      await actions.respond.mutateAsync({ id, command: { ...body, commandId: commands.current.get(fingerprint)! } });
      commands.current.delete(fingerprint);
      setSelected(new Set()); setResponse(""); setReason("");
      setNotice("Response sent. QuickPitik will decide each photo.");
    } catch (failure) {
      setError(errorText(failure));
    }
  }

  async function reviewCurrent() {
    let version = request.version;
    if (!actions.reviewCurrentState(id, version)) {
      setReviewStatus("Refreshing the request…");
      const read = await detail.refetch();
      if (read.isError || !read.data) { setReviewStatus("Couldn't refresh the request. Try again."); return; }
      version = read.data.version;
      if (!actions.reviewCurrentState(id, version)) { setReviewStatus("Couldn't refresh the request. Try again."); return; }
    }
    setReviewStatus(null); setError(null);
  }

  return (
    <article className="space-y-8">
      <header className="pb-8 border-b border-line">
        <Kicker as="p" tnum>Photo removal review · {formatLongDate(request.createdAt)}</Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">{request.event.name}</h1>
        <p className="mt-2 text-sm text-slate">{STATUS[request.status]}</p>
        <p className="mt-4 text-ink break-words">{request.sanitizedReason}</p>
        <p className="mt-3 max-w-xl text-sm text-ink-soft">
          QuickPitik makes the final decision on each photo. Your response is input: agreeing doesn&apos;t delete a photo by itself,
          and objecting isn&apos;t a veto. These photos are temporarily hidden from sale and search while the request is reviewed.
        </p>
      </header>

      {request.reviews.map((review) => (
        <p key={review.id} className="text-sm text-ink">
          {review.deadline && <>Respond by <span className="tnum">{formatSalesDate(review.deadline)}</span>. </>}
          {review.overdue && (
            <span className="text-warning">
              Overdue{review.overdueAt && <> since <span className="tnum">{formatSalesDate(review.overdueAt)}</span></>}. QuickPitik can decide
              without your response — no response is never counted as agreement.
            </span>
          )}
        </p>
      ))}

      {reviewRequired && (
        <div role="alert" className="rounded-2xl border border-line bg-bone-deep p-4 space-y-3">
          <p className="text-sm text-ink">This request changed while you were working. Review its current state before responding.</p>
          <button type="button" className={cn(BTN_SECONDARY, BTN_SIZE.sm)} disabled={detail.isFetching} onClick={() => void reviewCurrent()}>
            Review current state
          </button>
          {reviewStatus && <p role="status" className="text-sm text-slate">{reviewStatus}</p>}
        </div>
      )}

      <section aria-labelledby="removal-photos" className="space-y-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="removal-photos" className="font-display text-xl font-extrabold text-ink">Your photos in this request</h2>
          {open.length > 0 && (
            <label className="flex items-center gap-2 text-sm text-ink">
              <input type="checkbox" checked={allSelected} disabled={busy}
                onChange={() => setSelected(allSelected ? new Set() : new Set(open.map((item) => item.id)))} />
              Select all of mine
            </label>
          )}
        </div>
        <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {request.items.map((item) => (
            <li key={item.id} className="flex min-w-0 gap-3 rounded-xl border border-line p-3 text-sm">
              <span className="size-20 shrink-0 overflow-hidden rounded-lg bg-bone-deep">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                {item.photo?.imageUrl && <img src={item.photo.imageUrl} alt="" loading="lazy" className="h-full w-full object-cover" />}
              </span>
              <div className="min-w-0 flex-1 space-y-1 break-words">
                <p className="font-mono tnum text-ink truncate">{item.photoIdSnapshot}</p>
                <p className="text-ink">{outcome(item)}</p>
                {item.response && (
                  <p className="text-slate break-words">
                    {RESPONDED[item.response]}
                    {item.respondedAt && <> · <span className="tnum">{formatSalesDate(item.respondedAt)}</span></>}
                    {item.responseReason && <> — {item.responseReason}</>}
                  </p>
                )}
                {item.decisionReason && <p className="text-slate break-words">QuickPitik: {item.decisionReason}</p>}
                {canRespond(request, item) && (
                  <label className="flex min-w-0 items-center gap-2 text-ink">
                    <input type="checkbox" checked={selected.has(item.id)} disabled={busy} onChange={() => toggle(item.id)}
                      aria-label={`Select ${item.photoIdSnapshot}`} />
                    Include in response
                  </label>
                )}
              </div>
            </li>
          ))}
        </ul>
      </section>

      {open.length > 0 && (
        <section aria-labelledby="removal-response" className="space-y-4 border-t border-line pt-6">
          <h2 id="removal-response" className="font-display text-xl font-extrabold text-ink">Your response</h2>
          <fieldset className="space-y-2" disabled={busy}>
            <legend className="sr-only">Response</legend>
            {RESPONSES.map(([value, label, hint]) => (
              <label key={value} className="flex items-start gap-2 text-sm">
                <input type="radio" name="removal-response" value={value} checked={response === value} onChange={() => setResponse(value)} className="mt-1" />
                <span className="min-w-0 break-words"><span className="font-semibold text-ink">{label}</span> <span className="text-slate">— {hint}</span></span>
              </label>
            ))}
          </fieldset>
          {response === "OBJECT" && (
            <div className="space-y-1">
              <label htmlFor="removal-objection" className="text-sm font-semibold text-ink">Why you object</label>
              <textarea id="removal-objection" rows={3} maxLength={2000} value={reason} disabled={busy} className={INPUT}
                aria-invalid={reasonMissing} onChange={(e) => setReason(e.target.value)} />
            </div>
          )}
          {itemIds.length > 100 && <p className="text-sm text-warning">Respond to at most <span className="tnum">100</span> photos at a time.</p>}
          <button type="button" disabled={!canSubmit} onClick={() => void submit()} className={cn(BTN_PRIMARY, BTN_SIZE.md)}>
            {busy ? "Sending…" : "Send response"}
          </button>
        </section>
      )}
      <FieldError message={error} id="removal-response-error" />
      {notice && <p role="status" className="text-sm text-ink">{notice}</p>}
      <p className="border-t border-line pt-6 text-sm text-slate">
        Removal covers QuickPitik only. Copies buyers already downloaded or posted elsewhere can&apos;t be recalled.
      </p>
    </article>
  );
}
