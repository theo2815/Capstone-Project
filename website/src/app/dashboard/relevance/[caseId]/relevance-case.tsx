"use client";

import { useRef, useState } from "react";
import { Kicker } from "@/components/ui/kicker";
import { FieldError } from "@/components/ui/field-error";
import { Skeleton } from "@/components/ui/skeleton";
import { BTN_GHOST, BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ApiError } from "@/lib/api";
import type { PhotographerRelevanceCase } from "@/lib/api-photographer";
import { formatLongDate } from "@/lib/format";
import { cn } from "@/lib/utils";
import {
  APPEAL_FORM, APPEAL_MAX, CASE_LABELS, appealText, blurryText, hiddenCaseText, philippineToday, shownAgainText, type CaseText,
} from "@/lib/photographer-relevance-copy";
import { useFileRelevanceAppeal, usePhotographerRelevanceCase } from "@/hooks/use-photographer-relevance";
import { RelevanceBlurryReview } from "./relevance-blurry";

// RV-3.4a: what happened to the photographer's photos in one event, and the appeal. It renders only the named fields
// of the photographer's case view; nothing else the server might send (the model's reason above all) reaches the page.
export function RelevanceCase({ caseId }: { caseId: string }) {
  const query = usePhotographerRelevanceCase(caseId);
  const c = query.data;
  const text = c ? caseText(c) : null;

  if (query.isPending) return (
    <div role="status" className="mt-8 space-y-3">
      <span className="sr-only">{CASE_LABELS.loading}</span>
      <Skeleton className="h-7 w-64 max-w-full" />
      <Skeleton className="h-4 w-full max-w-md" />
      <Skeleton className="h-4 w-80 max-w-full" />
    </div>
  );
  // A failed background refresh keeps showing the last server copy. A case with nothing to tell reads as not found.
  if (!c || !text) return (
    <div role="alert" className="mt-8 space-y-4">
      {c || (query.error instanceof ApiError && query.error.status === 404) ? <p className="text-ink">{CASE_LABELS.notFound}</p> : <>
        <p className="text-ink">{CASE_LABELS.loadFailed}</p>
        <button type="button" className={cn(BTN_SECONDARY, BTN_SIZE.sm)} onClick={() => void query.refetch()}>{CASE_LABELS.retry}</button>
      </>}
    </div>
  );

  // A closed case shows only its title (and the appeal only through APPEAL_WON's own block).
  const status = c.current && c.appeal && c.appeal.status !== "WON" ? appealText(c) : null;
  const canAppeal = isHiding(c) && !c.appeal && !!c.appealDeadline;
  // RV-3.4b: a current Fits case that hides nothing and whose full check found blurry photos (the V2-T15 note).
  const reviewing = c.current && c.action === "NONE" && c.verdict === "FITS" && !c.noticeAt && (c.scanBlurry ?? 0) > 0;
  const windowOpen = canAppeal && philippineToday() <= c.appealDeadline!;

  return (
    <article className="max-w-2xl space-y-8">
      <header className="pb-8 border-b border-line">
        <Kicker as="p">{CASE_LABELS.kicker} · {c.eventName}</Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3 break-words">{text.title}</h1>
        {text.body.length > 0 && <p className="mt-4 text-ink-soft leading-relaxed break-words">{text.body.join(" ")}</p>}
      </header>

      {reviewing && <RelevanceBlurryReview caseId={c.caseId} />}

      {status && (
        <section aria-labelledby="relevance-appeal-status" className="space-y-2">
          <Kicker as="p" tone="soft" tnum>{formatLongDate(c.appeal!.decidedAt ?? c.appeal!.createdAt)}</Kicker>
          <h2 id="relevance-appeal-status" className="font-display text-xl font-extrabold text-ink">{status.title}</h2>
          <p className="text-ink-soft leading-relaxed">{status.body.join(" ")}</p>
        </section>
      )}
      {canAppeal && (windowOpen ? <AppealForm c={c} outcome={text.outcome} />
        : <p className="text-ink">{APPEAL_FORM.ended}</p>)}
    </article>
  );
}

const isHiding = (c: PhotographerRelevanceCase) => c.current && (c.action === "HIDE_ALL" || c.action === "HIDE_JUNK");

// The page's text is the note the server sent for this case's state (RelevanceNotifier, RelevanceAppealService); null
// when no note tells of it. No count of photos off sale is shown: hidden_photos is stamped once, when a hide first
// finishes, and never kept up to date (new uploads join a whole-batch hide; a won appeal brings photos back).
function caseText(c: PhotographerRelevanceCase): CaseText | null {
  const won = c.appeal?.status === "WON";
  if (!c.current) {
    if (c.closeReason === "RECHECK_LIFTED") return shownAgainText(c.eventName, "recheck");
    if (c.closeReason === "APPEAL_WON" && won) return appealText(c);
    // No approved text for the other close reasons (listed as a gap in RV-3.4a): the plain status word.
    return { title: CASE_LABELS.closed, body: [], outcome: [] };
  }
  if (won) return appealText(c);
  if (isHiding(c)) return hiddenCaseText(c, c.eventDate ? formatLongDate(c.eventDate, true) : null);
  // RV-3.4b: a Fits case's blurry note (no appeal clock), with its "Review and hide" grid below. With no blurry photos
  // found, no note was sent and there is nothing to tell.
  if (c.verdict === "FITS" && !c.noticeAt) return (c.scanBlurry ?? 0) > 0 ? blurryText(c) : null;
  // Only a Fits (scan flip) or Mixed case can have had photos put back; any other case that hides nothing never hid any.
  if (c.verdict !== "FITS" && c.verdict !== "MIXED") return null;
  return shownAgainText(c.eventName, c.verdict === "FITS" ? "scan" : null);
}

function AppealForm({ c, outcome }: { c: PhotographerRelevanceCase; outcome: string[] }) {
  const file = useFileRelevanceAppeal(c.caseId);
  const [reason, setReason] = useState("");
  const [error, setError] = useState<string | null>(null);
  const trimmed = reason.trim();
  const tooLong = trimmed.length > APPEAL_MAX;
  const busy = file.isPending;
  const canSend = trimmed.length > 0 && !tooLong && !busy;
  // isPending reaches the button only on the next render; this stops a second click in the same tick.
  const sending = useRef(false);

  async function send() {
    if (!canSend || sending.current) return;
    sending.current = true;
    setError(null);
    try {
      await file.mutateAsync(trimmed);
    } catch (failure) {
      // The server's refusal is shown as it words it (V2-T22 for the three 409s); the reason stays for a retry.
      setError(failure instanceof ApiError ? (failure.errors[0]?.message ?? failure.message) : failure instanceof Error ? failure.message : null);
    } finally {
      sending.current = false;
    }
  }

  return (
    <section aria-labelledby="relevance-appeal" className="space-y-5 border-t border-line pt-8">
      <div className="space-y-2">
        <h2 id="relevance-appeal" className="font-display text-xl font-extrabold text-ink">{APPEAL_FORM.title}</h2>
        <p className="text-sm text-ink-soft leading-relaxed">{APPEAL_FORM.line(formatLongDate(c.appealDeadline, true))}</p>
        {outcome.length > 0 && <p className="text-sm text-slate leading-relaxed">{outcome.join(" ")}</p>}
      </div>
      <fieldset disabled={busy} className="min-w-0 space-y-2">
        <Kicker as="label" tone="soft" htmlFor="relevance-appeal-reason" className="block">{APPEAL_FORM.label}</Kicker>
        <textarea
          id="relevance-appeal-reason"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          aria-invalid={tooLong}
          aria-describedby="relevance-appeal-hint relevance-appeal-error"
          className="w-full font-sans text-sm text-ink bg-surface border border-line rounded-xl px-4 py-3 min-h-[160px] resize-y focus:outline-none focus:border-ink"
        />
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <p id="relevance-appeal-hint" className="text-sm text-slate">{APPEAL_FORM.hint}</p>
          <Kicker as="p" tone="soft" tnum aria-hidden="true" className={cn(tooLong && "text-error")}>{APPEAL_FORM.counter(trimmed.length)}</Kicker>
        </div>
      </fieldset>
      <FieldError id="relevance-appeal-error" message={tooLong ? APPEAL_FORM.tooLong : error} />
      <div className="flex flex-wrap items-center justify-end gap-3">
        <button type="button" disabled={busy} onClick={() => { setReason(""); setError(null); }} className={cn(BTN_GHOST, BTN_SIZE.sm)}>
          {APPEAL_FORM.cancel}
        </button>
        <button type="button" disabled={!canSend} onClick={() => void send()} className={cn(BTN_PRIMARY, BTN_SIZE.md)}>
          {busy ? CASE_LABELS.sending : APPEAL_FORM.send}
        </button>
      </div>
    </section>
  );
}
