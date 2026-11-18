"use client";

import { useRef, useState } from "react";
import { AdminDetailDrawer } from "@/components/admin/admin-detail-drawer";
import { AdminConfirmModal } from "@/components/admin/admin-form-modal";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { Kicker } from "@/components/ui/kicker";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { BTN_DANGER, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { useRelevanceAppealActions, useRelevanceAppeals, useRelevanceCase } from "@/hooks/use-admin-relevance";
import { useToast } from "@/hooks/use-toast";
import { ApiError } from "@/lib/api";
import { serverMessage } from "@/lib/admin-flags";
import {
  APPEAL_NOTE_MAX,
  APPEAL_SECTIONS,
  APPEAL_TEXT,
  verdictLabel,
  type RelevanceAppealRow,
} from "@/lib/admin-relevance";
import { formatLongDate } from "@/lib/format";
import { RelevanceCaseBody, RelevanceCaseSamples } from "./relevance-case-drawer";

const labelCls = "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft";
const who = (a: RelevanceAppealRow) => (a.photographerHandle ? `@${a.photographerHandle}` : (a.photographerName ?? "—"));

// RV-3.5: the Appeals tab. Open appeals first, then the decided ones; `?row=` names the open appeal's id.
export function RelevanceAppeals({ appealId, onOpen }: { appealId: string; onOpen: (id: string) => void }) {
  const lists = { OPEN: useRelevanceAppeals("OPEN"), WON: useRelevanceAppeals("WON"), LOST: useRelevanceAppeals("LOST") };
  // shortcut: a ?row= link opens only an appeal in the loaded pages (there is no single-appeal read); fine at today's volume.
  const open = appealId ? Object.values(lists).flatMap((l) => l.items).find((a) => a.appealId === appealId) ?? null : null;

  return (
    <>
      {APPEAL_SECTIONS.map(({ status, label, empty }) => {
        const list = lists[status];
        return (
          <section key={status} aria-label={label} className="space-y-3">
            <Kicker as="h2" tone="soft">{label}</Kicker>
            {list.isLoading ? (
              <p role="status" className="font-sans text-sm text-slate-soft">Loading appeals…</p>
            ) : list.error && list.items.length === 0 ? (
              <div role="alert" className="space-y-3">
                <p className="font-sans text-sm text-ink-soft">Could not load the appeals.</p>
                <button type="button" onClick={list.refetch} className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`}>Retry</button>
              </div>
            ) : list.items.length === 0 ? (
              <p className="font-sans text-sm text-slate-soft">{empty}</p>
            ) : (
              <>
                <ul className="space-y-3">
                  {list.items.map((a) => (
                    <li key={a.appealId}>
                      <AppealRow appeal={a} onOpen={() => onOpen(a.appealId)} />
                    </li>
                  ))}
                </ul>
                <LoadMoreButton shown={list.items.length} total={list.total} increment={50}
                  onLoadMore={list.fetchNextPage} isLoading={list.isFetchingNextPage} label="Load more" />
              </>
            )}
          </section>
        );
      })}
      {/* Keyed by appeal: opening another one starts with the samples closed and an empty note. */}
      {open && <AppealDrawer key={open.appealId} appeal={open} onClose={() => onOpen("")} />}
    </>
  );
}

function AppealRow({ appeal, onOpen }: { appeal: RelevanceAppealRow; onOpen: () => void }) {
  return (
    <button type="button" onClick={onOpen}
      className="w-full text-left rounded-2xl border border-line bg-surface p-4 md:p-5 transition-colors hover:bg-bone-deep/50">
      <span className="flex flex-wrap items-center gap-2">
        <AdminStatusPill label={verdictLabel(appeal.verdict)} tone="slate" />
      </span>
      <span className="mt-3 block font-display text-base md:text-lg font-bold text-ink">{appeal.eventName ?? "—"}</span>
      <span className={`${labelCls} mt-1 block tnum`}>
        {who(appeal)} · {APPEAL_TEXT.filed} {formatLongDate(appeal.createdAt)}
        {appeal.decidedAt && ` · ${APPEAL_TEXT.decided} ${formatLongDate(appeal.decidedAt)}`}
      </span>
    </button>
  );
}

function AppealDrawer({ appeal, onClose }: { appeal: RelevanceAppealRow; onClose: () => void }) {
  const detail = useRelevanceCase(appeal.caseId);
  const { openEvidence, decide, evidencePending, decidePending } = useRelevanceAppealActions();
  const { showToast } = useToast();
  const [opened, setOpened] = useState(false);
  const [note, setNote] = useState("");
  const [confirming, setConfirming] = useState<"WON" | "LOST" | null>(null);
  // isPending reaches the button only on the next render; these stop a second click in the same tick.
  const opening = useRef(false);
  const deciding = useRef(false);
  const isOpen = appeal.status === "OPEN";
  const status = APPEAL_SECTIONS.find((s) => s.status === appeal.status)?.label ?? appeal.status;

  async function openSamples() {
    if (opening.current || opened) return;
    opening.current = true;
    try {
      await openEvidence(appeal.appealId);
      setOpened(true);
    } catch (error) {
      showToast({ kind: "error", message: serverMessage(error) ?? APPEAL_TEXT.openFailed });
    } finally {
      opening.current = false;
    }
  }

  async function run(outcome: "WON" | "LOST") {
    if (deciding.current) return;
    deciding.current = true;
    try {
      await decide({ appealId: appeal.appealId, outcome, note: note.trim() || undefined });
      showToast({ kind: "info", message: `${APPEAL_TEXT.done[outcome]}: ${appeal.eventName ?? appeal.appealId}` });
      onClose();
    } catch (error) {
      // The drawer stays open: the server's reason is shown, the note is kept for a retry.
      showToast({ kind: "error", message: serverMessage(error) ?? APPEAL_TEXT.decideFailed });
      // Another admin opened the samples since: this admin must open them again before deciding.
      if (error instanceof ApiError && error.errors[0]?.code === "EVIDENCE_NOT_OPENED") {
        setOpened(false);
        setConfirming(null);
      }
    } finally {
      deciding.current = false;
    }
  }

  const locked = !opened || decidePending;
  return (
    <AdminDetailDrawer
      open
      onClose={onClose}
      escDisabled={confirming !== null}
      kicker={`Appeal · ${appeal.appealId}`}
      title={appeal.eventName ?? "Appeal"}
      rightHeader={<AdminStatusPill label={status} tone={isOpen ? "amber" : "slate"} />}
      subtitle={<p className={`${labelCls} tnum`}>{who(appeal)} · {APPEAL_TEXT.filed} {formatLongDate(appeal.createdAt)}</p>}
      actions={isOpen && (
        <>
          <button type="button" disabled={locked} onClick={() => setConfirming("WON")} className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`}>{APPEAL_TEXT.won}</button>
          <button type="button" disabled={locked} onClick={() => setConfirming("LOST")} className={`${BTN_DANGER} ${BTN_SIZE.sm}`}>{APPEAL_TEXT.lost}</button>
          {confirming && (
            <AdminConfirmModal
              open
              onClose={() => setConfirming(null)}
              onConfirm={() => void run(confirming)}
              title={APPEAL_TEXT.confirm[confirming].title}
              confirmLabel={APPEAL_TEXT.confirm[confirming].confirmLabel}
              tone={confirming === "LOST" ? "danger" : "primary"}
              submitting={decidePending}
            >
              <p className="font-sans text-sm text-ink-soft">
                {appeal.caseStage === "APPLIED" ? APPEAL_TEXT.confirm[confirming].text : APPEAL_TEXT.confirmClosed}
              </p>
            </AdminConfirmModal>
          )}
        </>
      )}
    >
      <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
        <div className="space-y-6">
          <section className="space-y-3">
            <Kicker as="p" tone="soft">{APPEAL_TEXT.reason}</Kicker>
            {/* The photographer's own words: plain text only. */}
            <p className="font-sans text-sm md:text-base text-ink whitespace-pre-line break-words">{appeal.reason}</p>
          </section>
          {!isOpen && (
            <section className="space-y-2">
              <Kicker as="p" tone="soft" tnum>{status} · {formatLongDate(appeal.decidedAt)}</Kicker>
              {appeal.decisionNote && (
                <p className="font-sans text-sm text-ink-soft whitespace-pre-line break-words">
                  {APPEAL_TEXT.decisionNote}: {appeal.decisionNote}
                </p>
              )}
            </section>
          )}
          {isOpen && (
            <section className="space-y-2">
              <Kicker as="label" tone="soft" htmlFor="relevance-appeal-note" className="block">{APPEAL_TEXT.note}</Kicker>
              <textarea id="relevance-appeal-note" value={note} maxLength={APPEAL_NOTE_MAX} disabled={decidePending}
                onChange={(e) => setNote(e.target.value)} aria-describedby="relevance-appeal-note-hint"
                className="w-full font-sans text-sm text-ink bg-surface border border-line rounded-xl px-4 py-3 min-h-[120px] resize-y focus:outline-none focus:border-ink" />
              <p id="relevance-appeal-note-hint" className="font-sans text-sm text-slate-soft">{APPEAL_TEXT.noteHint}</p>
            </section>
          )}
        </div>

        {detail.isPending ? (
          <p role="status" className="font-sans text-sm text-slate-soft">Loading case…</p>
        ) : detail.isError ? (
          <div role="alert" className="space-y-3">
            <p className="font-sans text-sm text-ink-soft">Could not load the case.</p>
            <button type="button" onClick={() => void detail.refetch()} className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`}>Retry</button>
          </div>
        ) : (
          <div className="space-y-10">
            <RelevanceCaseBody detail={detail.data} samples={false} />
            {!isOpen || opened ? (
              <RelevanceCaseSamples detail={detail.data} />
            ) : (
              <section className="space-y-3">
                <Kicker as="p" tone="soft">{APPEAL_TEXT.samples}</Kicker>
                <p className="font-sans text-sm text-ink-soft">{APPEAL_TEXT.openSamplesHint}</p>
                <button type="button" disabled={evidencePending} onClick={() => void openSamples()} className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`}>
                  {APPEAL_TEXT.openSamples}
                </button>
              </section>
            )}
          </div>
        )}
      </div>
    </AdminDetailDrawer>
  );
}
