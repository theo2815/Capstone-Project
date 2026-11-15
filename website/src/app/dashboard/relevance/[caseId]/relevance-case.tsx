"use client";

import { useState } from "react";
import { Kicker } from "@/components/ui/kicker";
import { FieldError } from "@/components/ui/field-error";
import { BTN_GHOST, BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ApiError } from "@/lib/api";
import type { PhotographerRelevanceCase } from "@/lib/api-photographer";
import { formatLongDate } from "@/lib/format";
import { cn } from "@/lib/utils";
import {
  APPEAL_FORM, APPEAL_MAX, CASE_LABELS, appealText, blurryText, hiddenCaseText, philippineToday, shownAgainText, type CaseText,
} from "@/lib/photographer-relevance-copy";
import { useFileRelevanceAppeal, usePhotographerRelevanceCase } from "@/hooks/use-photographer-relevance";
import { usePhotographerEventDetail } from "@/hooks/use-photographer-data";

// RV-3.4a: what happened to the photographer's photos in one event, and the appeal. It renders only the named fields
// of the photographer's case view; nothing else the server might send (the model's reason above all) reaches the page.
export function RelevanceCase({ caseId }: { caseId: string }) {
  const query = usePhotographerRelevanceCase(caseId);
  const c = query.data;
  // V2-T13 names the event's date, which the case view does not carry.
  const event = usePhotographerEventDetail(c?.verdict === "WRONG_EVENT" ? c.eventId : null);

  if (query.isPending) return <p role="status" className="mt-8 text-slate">{CASE_LABELS.loading}</p>;
  // A failed background refresh keeps showing the last server copy.
  if (!c) return (
    <div role="alert" className="mt-8 space-y-4">
      {query.error instanceof ApiError && query.error.status === 404 ? <p className="text-ink">{CASE_LABELS.notFound}</p> : <>
        <p className="text-ink">{CASE_LABELS.loadFailed}</p>
        <button type="button" className={cn(BTN_SECONDARY, BTN_SIZE.sm)} onClick={() => void query.refetch()}>{CASE_LABELS.retry}</button>
      </>}
    </div>
  );

  const hiding = c.current && (c.action === "HIDE_ALL" || c.action === "HIDE_JUNK");
  const won = c.appeal?.status === "WON";
  const eventDate = event.detail?.date ? formatLongDate(event.detail.date, true) : null;
  const text: CaseText = !c.current ? { title: CASE_LABELS.closed, body: [], outcome: [] }
    : won ? appealText(c)!
    : hiding ? hiddenCaseText(c, eventDate)
    // RV-3.4b: a Fits case's blurry note (no appeal clock) gets its "Review and hide" grid here.
    : c.verdict === "FITS" && !c.noticeAt ? blurryText(c)
    : shownAgainText(c.eventName, c.verdict === "FITS");
  const status = c.appeal && !(won && c.current) ? appealText(c) : null;
  const canAppeal = hiding && !c.appeal && !!c.appealDeadline;
  const windowOpen = canAppeal && philippineToday() <= c.appealDeadline!;

  return (
    <article className="max-w-2xl space-y-8">
      <header className="pb-8 border-b border-line">
        <Kicker as="p">{CASE_LABELS.kicker} · {c.eventName}</Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3 break-words">{text.title}</h1>
        {text.body.length > 0 && <p className="mt-4 text-ink-soft leading-relaxed break-words">{text.body.join(" ")}</p>}
        <dl className="mt-6 flex flex-wrap gap-x-8 gap-y-3">
          <Stat label={CASE_LABELS.photos} value={c.batchPhotos} />
          <Stat label={CASE_LABELS.offSale} value={c.hiddenPhotos} />
        </dl>
      </header>

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

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <Kicker as="dt" tone="soft">{label}</Kicker>
      <dd className="font-mono tnum text-2xl text-ink mt-1">{value.toLocaleString("en-US")}</dd>
    </div>
  );
}

function AppealForm({ c, outcome }: { c: PhotographerRelevanceCase; outcome: string[] }) {
  const file = useFileRelevanceAppeal(c.caseId);
  const [reason, setReason] = useState("");
  const [error, setError] = useState<string | null>(null);
  const trimmed = reason.trim();
  const tooLong = trimmed.length > APPEAL_MAX;
  const busy = file.isPending;
  const canSend = trimmed.length > 0 && !tooLong && !busy;

  async function send() {
    if (!canSend) return;
    setError(null);
    try {
      await file.mutateAsync(trimmed);
    } catch (failure) {
      // The server's refusal is shown as it words it (V2-T22 for the three 409s); the reason stays for a retry.
      setError(failure instanceof ApiError ? (failure.errors[0]?.message ?? failure.message) : failure instanceof Error ? failure.message : null);
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
          <Kicker as="p" tone="soft" tnum aria-hidden="true" className={cn(tooLong && "text-error")}>{APPEAL_FORM.counter(reason.length)}</Kicker>
        </div>
      </fieldset>
      <FieldError id="relevance-appeal-error" message={error} />
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
