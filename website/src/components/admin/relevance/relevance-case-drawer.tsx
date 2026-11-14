"use client";

import { Kicker } from "@/components/ui/kicker";
import { Thumb } from "@/components/admin/flag-relevance-evidence";
import {
  actionLabel,
  alertKindLabel,
  bandLabel,
  evidenceLabel,
  isWouldHave,
  sampleLabel,
  verdictLabel,
  verdictNoteLabel,
  type RelevanceCaseDetail,
  type RelevanceCaseRow,
  type RelevanceSample,
} from "@/lib/admin-relevance";
import { formatLongDate } from "@/lib/format";

const labelCls = "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft";
const n = (v: number | null) => (v === null ? "—" : v.toLocaleString());
const score = (v: number | null) => (v === null ? "—" : v.toFixed(1));

export function RelevanceCaseSubtitle({ row }: { row: RelevanceCaseRow }) {
  return (
    <p className={`${labelCls} tnum`}>
      {row.photographerName}
      {row.photographerHandle && ` · @${row.photographerHandle}`}
      {" · "}
      {formatLongDate(row.decidedAt)}
    </p>
  );
}

/** The read-only body of a case: verdict, the model's reason, the numbers, words, samples and alerts. No actions yet. */
export function RelevanceCaseBody({ detail }: { detail: RelevanceCaseDetail }) {
  const row = detail.case;
  const note = verdictNoteLabel(row.verdictNote);
  const overridden = detail.modelVerdict !== row.verdict;
  return (
    <div className="space-y-10">
      <section className="space-y-3">
        <Kicker as="p" tone="soft">Verdict</Kicker>
        <p className="font-display text-xl md:text-2xl font-extrabold text-ink">{verdictLabel(row.verdict)}</p>
        {isWouldHave(row) ? (
          <div className="rounded-xl border border-line bg-bone-deep p-4 space-y-1">
            <p className="font-sans text-sm font-medium text-ink">Would have hidden: {actionLabel(row.action)}.</p>
            <p className="font-sans text-sm text-ink-soft">Record-only: nothing was hidden, held or sent to the photographer.</p>
          </div>
        ) : row.action && row.action !== "NONE" ? (
          <p className="font-sans text-sm text-ink">Action: {actionLabel(row.action)}</p>
        ) : null}
        {row.verdictNote && (
          <div className="rounded-xl border border-line p-4 space-y-1">
            {overridden && (
              <p className="font-sans text-sm font-medium text-ink">
                {detail.modelVerdict
                  ? `The model said ${verdictLabel(detail.modelVerdict)}; code stored ${verdictLabel(row.verdict)}.`
                  : `The model gave no usable verdict; code stored ${verdictLabel(row.verdict)}.`}
              </p>
            )}
            <p className="font-sans text-sm text-ink-soft">{note}</p>
          </div>
        )}
      </section>

      <section className="space-y-3">
        <Kicker as="p" tone="soft">Model&apos;s reason · admins only</Kicker>
        <p className="font-sans text-sm md:text-base text-ink-soft whitespace-pre-line">{detail.adminReason ?? "No reason recorded."}</p>
        {detail.evidenceUsed.length > 0 && (
          <p className={labelCls}>Evidence used: {detail.evidenceUsed.map(evidenceLabel).join(", ")}</p>
        )}
      </section>

      <section className="space-y-3">
        <Kicker as="p" tone="soft">Numbers</Kicker>
        <dl className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          <Field label="Batch photos" value={n(row.batchPhotos)} />
          <Field label="Sampled" value={n(detail.sampleSize)} />
          <Field label="Measured" value={n(detail.measured)} />
          <Field label="Blurry" value={n(detail.blurry)} />
          <Field label="Dark" value={n(detail.dark)} />
          <Field label="Not a race" value={n(detail.notRace)} />
          <Field label="Junk" value={row.junkPercent === null ? n(detail.junk) : `${n(detail.junk)} (${row.junkPercent}%)`} />
          <Field label="Words read" value={`${n(detail.wordsPhotos)} of ${n(row.batchPhotos)} photos`} />
          <Field label="Junk band" value={bandLabel(detail.band)} />
        </dl>
      </section>

      {detail.topWords.length > 0 && (
        <section className="space-y-3">
          <Kicker as="p" tone="soft">Top words on the photos</Kicker>
          <ul className="flex flex-wrap gap-2">
            {detail.topWords.map((w) => (
              <li key={w.word} className={`${labelCls} text-ink rounded-full border border-line px-3 py-1 tnum`}>
                {w.word} · {w.photos}
              </li>
            ))}
          </ul>
        </section>
      )}

      <Samples title="Junk samples" samples={detail.junkSamples} empty="No sampled photo counts as junk." />
      <Samples title="Clean samples" samples={detail.cleanSamples} empty="No sampled photo is listed as clean or not judged." />

      {detail.alerts.length > 0 && (
        <section className="space-y-3">
          <Kicker as="p" tone="soft">Alerts for this batch</Kicker>
          <ul className="space-y-2">
            {detail.alerts.map((a) => (
              <li key={a.alertId} className="font-sans text-sm text-ink tnum">
                {alertKindLabel(a.kind)}
                {a.wouldHave && " (would have)"}
                {a.photos !== null && ` · ${a.photos.toLocaleString()} photos`}
                {" · "}
                {formatLongDate(a.createdAt)}
                {a.seenAt ? " · seen" : " · unseen"}
              </li>
            ))}
          </ul>
        </section>
      )}

      {detail.summarySent && (
        <section className="space-y-3">
          <Kicker as="p" tone="soft">Summary sent to the model</Kicker>
          <pre className="max-h-64 overflow-auto rounded-xl border border-line bg-bone-deep p-4 font-mono text-[13px] text-ink whitespace-pre-wrap break-words tnum">
            {detail.summarySent}
          </pre>
          <p className={labelCls}>
            {detail.model ?? "Model unknown"} · prompt {detail.promptVersion ?? "unknown"}
          </p>
        </section>
      )}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className={labelCls}>{label}</dt>
      <dd className="mt-1 font-sans text-sm text-ink tnum">{value}</dd>
    </div>
  );
}

function Samples({ title, samples, empty }: { title: string; samples: RelevanceSample[]; empty: string }) {
  return (
    <section className="space-y-3">
      <Kicker as="p" tone="soft">{title}</Kicker>
      {samples.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">{empty}</p>
      ) : (
        <ul className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {samples.map((s) => (
            <li key={s.photoId}>
              <Thumb url={s.thumbnailUrl} alt="Sampled photo" />
              <p className="mt-1 font-sans text-sm font-medium text-ink">{sampleLabel(s)}</p>
              <p className={`${labelCls} tnum`}>
                Sharp {score(s.fgSharpness ?? s.sharpness)} · Bright {score(s.brightness)}
              </p>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
