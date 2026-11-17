"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Kicker } from "@/components/ui/kicker";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { BTN_DANGER, BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import {
  COPY_PAIR_NOTICE,
  canHideBatch,
  serverMessage,
  type CopySample,
  type Flag,
  type RelevanceBatch,
  type RelevanceEvidence,
} from "@/lib/admin-flags";
import { formatLongDate } from "@/lib/format";
import { useFlagActions, useFlagHiddenPhotos, useFlagRelevance } from "@/hooks/use-admin-data";
import { useToast } from "@/hooks/use-toast";

const MAX_SAMPLES = 12;

const labelCls =
  "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft";
const secondaryBtn = `${BTN_SECONDARY} ${BTN_SIZE.sm}`;
const tertiaryBtn = `${BTN_GHOST} ${BTN_SIZE.sm} px-3`;
const dangerBtn = `${BTN_DANGER} ${BTN_SIZE.sm}`;

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

export function Thumb({ url, alt }: { url: string | null; alt: string }) {
  return url ? (
    <img src={url} alt={alt} loading="lazy" decoding="async" className="aspect-[3/2] w-full rounded-lg border border-line object-cover bg-bone-deep" />
  ) : (
    <div role="img" aria-label={`${alt} (no longer available)`} className="aspect-[3/2] w-full rounded-lg border border-line bg-bone-deep" />
  );
}

function Count({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className={labelCls}>{label}</dt>
      <dd className="mt-1 font-sans text-sm text-ink tnum">{value}</dd>
    </div>
  );
}

function ofTotal(part: number | null, total: number | null): string {
  return part === null || total === null ? "Not available" : `${part} of ${total}`;
}

function CopyPair({ sample }: { sample: CopySample }) {
  // An exact copy names the earlier same file; a near copy names the nearest photo in the event.
  const otherUrl = sample.exact ? sample.exactThumbnailUrl : sample.matchThumbnailUrl;
  const handle = sample.exact ? sample.exactPhotographerHandle : sample.matchPhotographerHandle;
  return (
    <li className="rounded-xl border border-line p-3">
      <div className="grid grid-cols-2 gap-3">
        <Thumb url={sample.thumbnailUrl} alt="Uploaded photo" />
        <Thumb url={otherUrl} alt="Earlier photo" />
      </div>
      <p className={`${labelCls} mt-2 tnum`}>
        {sample.exact ? "Exact copy" : `Distance ${sample.distance ?? "n/a"}`}
        <span> · </span>
        {handle ? `earlier upload by @${handle}` : "earlier owner no longer on QuickPitik"}
      </p>
    </li>
  );
}

function BatchCard({ flag, batch }: { flag: Flag; batch: RelevanceBatch }) {
  const qc = useQueryClient();
  const { hideBatch, isPending, pendingAction } = useFlagActions();
  const { showToast } = useToast();
  const [confirming, setConfirming] = useState(false);
  const submitting = isPending(flag.id);
  // A photographer's own event has no copy or date rule (C8): exact and marker counts only.
  const gone = batch.eventName === "";
  const own = !batch.sharedRace && !gone;
  const rule =
    batch.copyLow && batch.dateLow ? "Copy rule and date rule" : batch.copyLow ? "Copy rule" : batch.dateLow ? "Date rule" : "No rule fired";
  const name = gone ? "Event no longer available" : batch.eventName;

  async function handleHide() {
    try {
      if (!(await hideBatch(flag.id, batch.eventId))) return;
      // The action refetched the evidence; read the batch's count from the fresh data.
      const fresh = qc
        .getQueriesData<RelevanceEvidence>({ queryKey: ["admin", "flags", "relevance", flag.id] })
        .map(([, data]) => data?.batches.find((b) => b.eventId === batch.eventId))
        .find(Boolean);
      const hidden = fresh?.hiddenPhotos ?? batch.hiddenPhotos;
      showToast({ kind: "info", message: `${plural(hidden, "photo", "photos")} hidden in ${batch.eventName}` });
      setConfirming(false);
    } catch (error) {
      showToast({
        kind: "error",
        message: serverMessage(error) ?? `Could not hide the batch for ${batch.eventName} — check the flag before retrying.`,
      });
    }
  }

  return (
    <section aria-label={`Batch in ${gone ? "an event that is no longer available" : batch.eventName}`} className="rounded-2xl border border-line p-4 md:p-5 space-y-4">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h4 className="font-display text-base md:text-lg font-medium text-ink">{name}</h4>
          <p className={`${labelCls} mt-1 tnum`}>
            {!gone && (
              <>
                {formatLongDate(batch.eventDate, true)}
                <span> · </span>
              </>
            )}
            {own ? "Own event" : rule}
          </p>
        </div>
        <p className="font-sans text-sm text-ink tnum">Hidden ({batch.hiddenPhotos})</p>
      </div>

      <dl className="grid grid-cols-2 sm:grid-cols-3 gap-4">
        {!own && <Count label="Look-ups" value={`${batch.tightHits} tight of ${batch.lookups}`} />}
        {!own && <Count label="Weak hits" value={String(batch.weakHits)} />}
        <Count label="Exact copies" value={String(batch.exactMatches)} />
        <Count label="Marker matches" value={String(batch.xmpMatches)} />
        {!own && <Count label="Dated outside window" value={ofTotal(batch.datedOutside, batch.datedPhotos)} />}
        {!own && <Count label="Bib overlap" value={ofTotal(batch.bibOverlapPhotos, batch.bibbedPhotos)} />}
      </dl>

      {batch.copySamples.length > 0 && (
        <div className="space-y-3">
          <p className="font-sans text-sm text-ink-soft">{COPY_PAIR_NOTICE}</p>
          <ul className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {batch.copySamples.slice(0, MAX_SAMPLES).map((s) => (
              <CopyPair key={s.photoId} sample={s} />
            ))}
          </ul>
        </div>
      )}

      {batch.dateSamples.length > 0 && (
        <div className="space-y-3">
          <p className={labelCls}>Photos dated outside the event window</p>
          <ul className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {batch.dateSamples.slice(0, MAX_SAMPLES).map((s) => (
              <li key={s.photoId}>
                <Thumb url={s.thumbnailUrl} alt="Uploaded photo" />
                <p className={`${labelCls} mt-1 tnum`}>{formatLongDate(s.takenOn, true)}</p>
              </li>
            ))}
          </ul>
        </div>
      )}

      {canHideBatch(flag) && (
        <div className="space-y-3">
          {submitting && <p role="status">{pendingAction(flag.id)}</p>}
          {confirming ? (
            <div className="rounded-xl border border-line bg-bone-deep p-4 space-y-3">
              <p className="font-sans text-sm text-ink-soft">
                Takes every photo this photographer uploaded to {batch.eventName} off sale. The flag stays open. You can restore photos one by one below.
              </p>
              <div className="flex gap-3 flex-wrap">
                <button type="button" disabled={submitting} onClick={() => setConfirming(false)} className={tertiaryBtn}>
                  Cancel
                </button>
                <button type="button" disabled={submitting} onClick={handleHide} className={dangerBtn}>
                  Confirm hide batch
                </button>
              </div>
            </div>
          ) : (
            <button type="button" disabled={submitting} onClick={() => setConfirming(true)} className={dangerBtn}>
              Hide batch…
            </button>
          )}
        </div>
      )}
    </section>
  );
}

function HiddenPhotos({ flag }: { flag: Flag }) {
  const query = useFlagHiddenPhotos(flag.id);
  const { restorePhoto, isPending } = useFlagActions();
  const { showToast } = useToast();
  const submitting = isPending(flag.id);
  const items = query.items;

  async function handleRestore(photoId: string) {
    try {
      if (await restorePhoto(flag.id, photoId)) showToast({ kind: "info", message: "Photo restored" });
    } catch (error) {
      showToast({
        kind: "error",
        message: serverMessage(error) ?? "Could not restore the photo — check the flag before retrying.",
      });
    }
  }

  if (query.isLoading) return <p role="status" className="font-sans text-sm text-slate-soft">Loading hidden photos…</p>;
  if (query.error && items.length === 0) {
    return (
      <div role="alert" className="space-y-2">
        <p className="font-sans text-sm text-ink-soft">Could not load the hidden photos.</p>
        <button type="button" onClick={() => query.refetch()} className={secondaryBtn}>Retry</button>
      </div>
    );
  }
  if (items.length === 0) return <p className="font-sans text-sm text-slate-soft">No photos hidden by this flag.</p>;
  return (
    <>
    <ul className="grid grid-cols-2 sm:grid-cols-3 gap-3">
      {items.map((p) => (
        <li key={p.photoId} data-restored={p.restoredAt ? "true" : undefined} className={p.restoredAt ? "opacity-50" : undefined}>
          <Thumb url={p.thumbnailUrl} alt="Hidden photo" />
          <p className={`${labelCls} mt-1 tnum`}>
            {p.restoredAt ? `Restored ${formatLongDate(p.restoredAt)}` : `Hidden ${formatLongDate(p.hiddenAt)}`}
          </p>
          {!p.restoredAt && (
            <button type="button" disabled={submitting} onClick={() => handleRestore(p.photoId)} className={`${tertiaryBtn} mt-1`}>
              Restore
            </button>
          )}
        </li>
      ))}
    </ul>
    <LoadMoreButton
      shown={items.length}
      total={query.total}
      increment={PAGE_SIZE.ADMIN_INCREMENT}
      onLoadMore={query.fetchNextPage}
      isLoading={query.isFetchingNextPage}
      label="Load more"
    />
    </>
  );
}

/** Drawer section for a relevance flag: the batches behind it, the hide action and the hidden photos. */
export function FlagRelevanceEvidence({ flag }: { flag: Flag }) {
  const query = useFlagRelevance(flag.id);
  return (
    <section aria-label="Relevance evidence" className="space-y-6">
      <Kicker as="p" tone="soft">Relevance evidence</Kicker>
      {query.isPending ? (
        <p role="status" className="font-sans text-sm text-slate-soft">Loading evidence…</p>
      ) : query.isError ? (
        <div role="alert" className="space-y-2">
          <p className="font-sans text-sm text-ink-soft">Could not load the evidence.</p>
          <button type="button" onClick={() => query.refetch()} className={secondaryBtn}>Retry</button>
        </div>
      ) : query.data.batches.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">No batch is linked to this flag.</p>
      ) : (
        <div className="space-y-5">
          {query.data.batches.map((b) => (
            <BatchCard key={b.eventId} flag={flag} batch={b} />
          ))}
        </div>
      )}
      <div className="space-y-3">
        <p className={labelCls}>Hidden photos</p>
        <HiddenPhotos flag={flag} />
      </div>
    </section>
  );
}
