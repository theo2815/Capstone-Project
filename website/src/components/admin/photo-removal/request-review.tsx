"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { useAdminRemovalActions, useAdminRemovalRequest, useRemovalEvidence, useRemovalPhotos } from "@/hooks/use-photo-removal";
import { useConfirmation } from "@/hooks/use-confirmation";
import { FindPhotosControls } from "@/components/events/find-photos-controls";
import { PhotoPreviewCard } from "@/components/photos/photo-preview-card";
import { AdminFieldLabel } from "@/components/admin/admin-form-fields";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { Kicker } from "@/components/ui/kicker";
import { FieldError } from "@/components/ui/field-error";
import { BTN_DANGER, BTN_GHOST, BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ApiError } from "@/lib/api";
import { validateImageFile } from "@/lib/image-utils";
import { formatLongDate } from "@/lib/format";
import { cn } from "@/lib/utils";
import { REMOVAL_INPUT } from "./request-intake";
import type { PhotoRemovalItemDto, PhotoRemovalRequestDto, RemovalMatchResult, RemovalMatchSource, RemovalPhotoDto } from "@/types/photo-removal";

const PAGE = 24;
const MAX_EVIDENCE_BYTES = 5 * 1024 * 1024;
const UNASSIGNED = "Unassigned owner";
type Selected = { photo: RemovalPhotoDto; source: RemovalMatchSource; note: string };
type MatchState = { kind: "idle" | "running" } | { kind: "result"; result: RemovalMatchResult } | { kind: "error"; message: string };
const errorText = (error: unknown) => error instanceof Error ? error.message : "Something went wrong. Try again.";
const ownerOf = (id: string | null, name: string | null) => id ? name ?? "Unnamed photographer" : UNASSIGNED;
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

export function RemovalRequestReview({ requestId }: { requestId: string }) {
  const detail = useAdminRemovalRequest(requestId);
  if (detail.isPending) return <p role="status" className="mt-8 text-slate">Loading case…</p>;
  // A failed background refresh keeps showing the last server copy.
  if (!detail.data) return (
    <div role="alert" className="mt-8 space-y-4">
      <p>Couldn&apos;t load this case.</p>
      <button type="button" className={cn(BTN_SECONDARY, BTN_SIZE.sm)} onClick={() => void detail.refetch()}>Try again</button>
    </div>
  );
  const request = detail.data;
  return (
    <>
      <CaseHeader request={request} />
      {request.status === "DRAFT"
        ? <DraftWorkspace request={request} refreshing={detail.isFetching} refresh={detail.refetch} />
        : isRunner(request)
          ? <RunnerSuggestions request={request} refreshing={detail.isFetching} refresh={detail.refetch} />
          : <VerifiedSummary request={request} />}
    </>
  );
}

// Enum display labels (like FLAG_REASON_LABEL); an unknown code shows as itself.
const RISK_FLAG_LABEL: Record<string, string> = {
  LIVENESS_UNCERTAIN: "The live check was uncertain",
  UNVERIFIED_EMAIL: "Email not verified",
  PRIOR_REJECTION: "An earlier request from this runner was rejected",
  POSSIBLY_TRUNCATED: "There may be more matches than were checked",
  OVER_REQUEST_CAP: "More matches than one request may hide automatically",
  PHOTOGRAPHER_BUDGET: "A photographer's daily hide limit was reached",
  AUTO_HIDE_OFF: "Automatic hiding is switched off",
  GROUP_PHOTOS: "Some matched photos show other people",
  MATCHING_UNAVAILABLE: "Face matching was unavailable: match by hand",
};
const BAND_LABEL = { PASS: "passed", UNCERTAIN: "uncertain" } as const;
const isRunner = (request: PhotoRemovalRequestDto) => request.source === "RUNNER";

function CaseHeader({ request }: { request: PhotoRemovalRequestDto }) {
  const runner = isRunner(request);
  return (
    <header className="pb-8 border-b border-line">
      <Kicker as="p" tnum>{runner ? "Runner request" : "Removal case"} · {formatLongDate(request.createdAt)}</Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">{request.event.name}</h1>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <AdminStatusPill label={runner ? "Runner" : "Support"} tone={runner ? "ink" : "muted"} />
        <AdminStatusPill label={request.status} tone={request.status === "DRAFT" ? "amber" : "ink"} />
      </div>
      <dl className="mt-5 grid gap-3 sm:grid-cols-2 text-sm">
        {runner ? (
          <div>
            <dt className="text-slate">Requester account</dt>
            {/* Shown, not a gate (Q6). */}
            <dd className="text-ink tnum">Account created {formatLongDate(request.requesterAccountCreatedAt)} · Email verified {request.requesterEmailVerified ? "yes" : "no"}</dd>
            {request.livenessBand && <dd className="text-ink">Live check: {BAND_LABEL[request.livenessBand]}</dd>}
          </div>
        ) : <div><dt className="text-slate">Support reference</dt><dd className="text-ink break-words">{request.supportReference}</dd></div>}
        <div><dt className="text-slate">Requester contact</dt><dd className="text-ink break-words">{request.requesterContact}</dd></div>
        <div className="sm:col-span-2"><dt className="text-slate">Reason (admin only)</dt><dd className="text-ink break-words">{request.requesterReason}</dd></div>
        {runner && !!request.riskFlags?.length && (
          <div className="sm:col-span-2">
            <dt className="text-slate">Needs a closer look</dt>
            {request.riskFlags.map((flag) => <dd key={flag} className="text-amber-700">{RISK_FLAG_LABEL[flag] ?? flag}</dd>)}
          </div>
        )}
      </dl>
    </header>
  );
}

type RefreshCase = () => Promise<{ data?: PhotoRemovalRequestDto; isError: boolean }>;

// After a 409 the hook blocks new commands until this panel's admin reviews a fresh, idle read of the case.
export function ReviewCurrentState({ request, actions, refreshing, refresh, onReviewed }: {
  request: PhotoRemovalRequestDto; actions: ReturnType<typeof useAdminRemovalActions>; refreshing: boolean;
  refresh: RefreshCase; onReviewed?: () => void;
}) {
  const [status, setStatus] = useState<string | null>(null);
  if (!actions.reviewRequired(request.id)) return null;
  async function review() {
    let version = request.version;
    if (!actions.reviewCurrentState(request.id, version)) {
      setStatus("Refreshing the case…");
      const read = await refresh();
      if (read.isError || !read.data) { setStatus("Couldn't refresh the case. Try again."); return; }
      version = read.data.version;
      if (!actions.reviewCurrentState(request.id, version)) { setStatus("Couldn't refresh the case. Try again."); return; }
    }
    setStatus(null);
    onReviewed?.();
  }
  return (
    <div role="alert" className="rounded-2xl border border-amber-600/30 bg-bone-deep p-4 space-y-3">
      <p className="text-sm text-ink">This case changed while you were working. Review the current state before continuing.</p>
      <button type="button" className={cn(BTN_SECONDARY, BTN_SIZE.sm)} disabled={refreshing} onClick={() => void review()}>
        Review current state
      </button>
      {status && <p role="status" className="text-sm text-slate">{status}</p>}
    </div>
  );
}

function DraftWorkspace({ request, refreshing, refresh }: { request: PhotoRemovalRequestDto; refreshing: boolean; refresh: RefreshCase }) {
  const actions = useAdminRemovalActions();
  const id = request.id;
  const [file, setFile] = useState<File | null>(null);
  const [bib, setBib] = useState("");
  const [bibFilter, setBibFilter] = useState("");
  const [offset, setOffset] = useState(0);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [match, setMatch] = useState<MatchState>({ kind: "idle" });
  const [selection, setSelection] = useState(new Map<string, Selected>());
  const [method, setMethod] = useState("");
  const [note, setNote] = useState("");
  const [verifyError, setVerifyError] = useState<string | null>(null);
  const [preview, setPreview] = useState<{ items: RemovalPhotoDto[]; index: number } | null>(null);
  const gallery = useRemovalPhotos(id, { offset, limit: PAGE, bib: bibFilter || undefined });
  // A retry of the same action after a failure reuses its command ID; any changed action gets a new one.
  const commands = useRef(new Map<string, string>());
  const commandFor = (fingerprint: string) => {
    if (!commands.current.has(fingerprint)) commands.current.set(fingerprint, crypto.randomUUID());
    return commands.current.get(fingerprint)!;
  };
  const reviewRequired = actions.reviewRequired(id);
  const busy = actions.replaceEvidence.isPending || actions.match.isPending || actions.verify.isPending;

  // Selfie selections are only valid for the latest match run, so a new run drops them.
  const dropSelfieSelections = () => setSelection((current) => new Map([...current].filter(([, item]) => item.source !== "SELFIE")));

  async function runMatch(version: number) {
    dropSelfieSelections();
    setMatch({ kind: "running" });
    const fingerprint = `match:${version}`;
    try {
      const result = await actions.match.mutateAsync({ id, command: { commandId: commandFor(fingerprint), expectedVersion: version } });
      commands.current.delete(fingerprint);
      setMatch({ kind: "result", result });
    } catch (error) {
      const status = error instanceof ApiError ? error.status : undefined;
      setMatch({ kind: "error", message: status === 503
        ? "Selfie matching is unavailable right now. Use bib search or browse the event photos below."
        : status === 422 ? "No face was found in the selfie. Try a clearer photo, or use bib search and browsing."
        : status === 404 ? "Upload a selfie before matching."
        : errorText(error) });
    }
  }

  async function search() {
    setSearchError(null);
    if (bib.trim() !== bibFilter) { setBibFilter(bib.trim()); setOffset(0); }
    if (!file) return;
    const invalid = validateImageFile(file) ?? (file.size > MAX_EVIDENCE_BYTES ? "Selfie must be under 5 MB." : null);
    if (invalid) { setSearchError(invalid); return; }
    const fingerprint = `evidence:${request.version}:${file.name}:${file.size}:${file.lastModified}`;
    try {
      const updated = await actions.replaceEvidence.mutateAsync({ id, file, commandId: commandFor(fingerprint), expectedVersion: request.version });
      commands.current.delete(fingerprint);
      setFile(null);
      await runMatch(updated.version);
    } catch (error) {
      setSearchError(errorText(error));
    }
  }

  function toggle(photo: RemovalPhotoDto, source: RemovalMatchSource) {
    setSelection((current) => {
      const next = new Map(current);
      if (next.has(photo.photo.id)) next.delete(photo.photo.id);
      else next.set(photo.photo.id, { photo, source, note: "" });
      return next;
    });
  }

  const selected = [...selection.values()];
  const missingNotes = selected.some((item) => item.source === "MANUAL" && !item.note.trim());
  const canVerify = selected.length > 0 && selected.length <= 100 && method.trim() && note.trim() && !missingNotes && !busy && !refreshing && !reviewRequired;

  async function verify() {
    if (!canVerify) return;
    setVerifyError(null);
    const body = {
      expectedVersion: request.version, verificationMethod: method.trim(), verificationNote: note.trim(),
      selectedPhotos: selected.map((item) => item.source === "MANUAL"
        ? { photoId: item.photo.photo.id, source: item.source, note: item.note.trim() }
        : { photoId: item.photo.photo.id, source: item.source }),
    };
    const fingerprint = `verify:${JSON.stringify(body)}`;
    try {
      await actions.verify.mutateAsync({ id, command: { ...body, commandId: commandFor(fingerprint) } });
      commands.current.delete(fingerprint);
    } catch (error) {
      setVerifyError(errorText(error));
    }
  }

  const tile = (items: RemovalPhotoDto[], source: RemovalMatchSource) => items.map((item, index) => (
    <PhotoTile key={item.photo.id} item={item} checked={selection.has(item.photo.id)} disabled={busy}
      onToggle={() => toggle(item, source)} onPreview={() => setPreview({ items, index })} position={`${index + 1} of ${items.length}`} />
  ));
  const galleryItems = gallery.data?.items ?? [];
  const galleryTotal = gallery.data?.total ?? 0;

  return (
    <div className="mt-8 space-y-12">
      <ReviewCurrentState request={request} actions={actions} refreshing={refreshing} refresh={refresh}
        onReviewed={() => { setMatch({ kind: "idle" }); dropSelfieSelections(); }} />

      <section aria-labelledby="removal-find" className="space-y-4">
        <h2 id="removal-find" className="font-display text-xl font-extrabold text-ink">Find the runner&apos;s photos</h2>
        {request.evidencePresent && <EvidencePreview requestId={id} />}
        {request.evidencePresent && file && <p className="text-sm text-slate">Searching with a new selfie replaces the one on file and clears earlier matches.</p>}
        {/* A runner case's evidence is its own live-check image: it is never replaced here (R10). */}
        {isRunner(request)
          ? !request.evidencePresent && <p className="text-sm text-amber-700">No live-check image was saved for this request. Pick the runner&apos;s photos by hand below, or cancel the case.</p>
          : <FindPhotosControls file={file} bib={bib} onFileChange={setFile} onBibChange={setBib} onSearch={() => void search()}
              busy={busy || reviewRequired} error={searchError} allowPersonalLibrary={false} />}
        {request.evidencePresent && !file && (
          <button type="button" className={cn(BTN_SECONDARY, BTN_SIZE.sm)} disabled={busy || reviewRequired || refreshing} onClick={() => void runMatch(request.version)}>
            Match stored selfie
          </button>
        )}
      </section>

      {match.kind === "running" && <p role="status" className="text-sm text-slate">Matching the selfie across every photographer in this event…</p>}
      {match.kind === "error" && <p role="alert" className="text-sm text-ink">{match.message}</p>}
      {match.kind === "result" && (
        <section aria-label="Selfie matches" className="space-y-4">
          <h2 className="font-display text-xl font-extrabold text-ink">Selfie matches</h2>
          <p className="text-sm text-slate">
            Candidates only — nothing is selected until you tick it. Showing up to <span className="tnum">{match.result.candidateLimit}</span> matches.
            {match.result.possiblyTruncated && " There may be more matches than shown; check bib results and browse too."}
            {match.result.candidatesPurged && " Earlier candidates were cleared because the selfie changed."}
          </p>
          {match.result.items.length === 0
            ? <p className="text-sm text-slate">No matching photos in this event.</p>
            : <ul className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">{tile(match.result.items, "SELFIE")}</ul>}
        </section>
      )}

      <section aria-labelledby="removal-browse" className="space-y-4">
        <h2 id="removal-browse" className="font-display text-xl font-extrabold text-ink">{bibFilter ? <>Bib <span className="tnum">{bibFilter}</span></> : "All event photos"}</h2>
        <p className="text-sm text-slate">
          Every photographer in this event{bibFilter ? "" : ". Photos added from here need a note explaining why"}. <span className="tnum">{galleryTotal}</span> photos.
          {bibFilter && <button type="button" className={cn(BTN_GHOST, "ml-2 underline")} onClick={() => { setBib(""); setBibFilter(""); setOffset(0); }}>Clear bib</button>}
        </p>
        {gallery.isPending && <p role="status" className="text-sm text-slate">Loading photos…</p>}
        {gallery.isError && <p role="alert" className="text-sm text-ink">Couldn&apos;t load photos. <button type="button" className={cn(BTN_GHOST, "underline")} onClick={() => void gallery.refetch()}>Try again</button></p>}
        {gallery.isSuccess && galleryItems.length === 0 && <p className="text-sm text-slate">No photos found.</p>}
        <ul className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">{tile(galleryItems, bibFilter ? "BIB" : "MANUAL")}</ul>
        {galleryTotal > PAGE && (
          <div className="flex items-center gap-2">
            <button type="button" className={cn(BTN_GHOST, BTN_SIZE.sm)} disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>← Previous</button>
            <span className="font-mono tnum text-sm text-slate">{offset + 1}–{Math.min(offset + PAGE, galleryTotal)} of {galleryTotal}</span>
            <button type="button" className={cn(BTN_GHOST, BTN_SIZE.sm)} disabled={offset + PAGE >= galleryTotal} onClick={() => setOffset(offset + PAGE)}>Next →</button>
          </div>
        )}
      </section>

      {selected.length > 0 && (
        <section aria-label="Confirm verification" className="space-y-5 border-t border-line pt-8">
          <h2 className="font-display text-xl font-extrabold text-ink">Confirm verification</h2>
          <p className="text-sm text-ink"><span className="tnum">{plural(selected.length, "photo")}</span> will be hidden from sale and search while owners review.</p>
          {groupByOwner(selected, (item) => [item.photo.photographerId, item.photo.photographerName]).map(([key, owner, items]) => (
            <div key={key} className="space-y-2">
              <h3 className="font-display font-bold text-ink">{owner}</h3>
              {key === ""
                ? <p className="text-sm text-amber-700">No photographer will be asked to review these photos — they have no owner on record.</p>
                : <p className="text-sm text-slate">Will receive a review request for <span className="tnum">{plural(items.length, "photo")}</span>.</p>}
              <ul className="space-y-2">
                {items.map((item) => (
                  <li key={item.photo.photo.id} className="text-sm text-ink">
                    <span className="font-mono tnum">{item.photo.photo.id}</span> · {item.source}
                    {item.source === "MANUAL" && (
                      <div className="mt-1">
                        <label htmlFor={`note-${item.photo.photo.id}`} className="sr-only">Note for {item.photo.photo.id}</label>
                        <input id={`note-${item.photo.photo.id}`} type="text" maxLength={2000} placeholder="Why this photo is the runner (required)"
                          aria-invalid={!item.note.trim()} value={item.note} disabled={busy} className={REMOVAL_INPUT}
                          onChange={(e) => { const value = e.target.value; setSelection((current) => new Map(current).set(item.photo.photo.id, { ...item, note: value })); }} />
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}
          <div className="space-y-2">
            <AdminFieldLabel htmlFor="removal-method">Verification method</AdminFieldLabel>
            <input id="removal-method" type="text" maxLength={2000} value={method} disabled={busy} onChange={(e) => setMethod(e.target.value)} className={REMOVAL_INPUT} />
          </div>
          <div className="space-y-2">
            <AdminFieldLabel htmlFor="removal-note">Verification note</AdminFieldLabel>
            <textarea id="removal-note" rows={3} maxLength={2000} value={note} disabled={busy} onChange={(e) => setNote(e.target.value)} className={REMOVAL_INPUT} />
          </div>
          {selected.length > 100 && <p className="text-sm text-amber-700">Select at most <span className="tnum">100</span> photos per verification.</p>}
          <FieldError message={verifyError} id="removal-verify-error" />
          <button type="button" disabled={!canVerify} onClick={() => void verify()} className={cn(BTN_PRIMARY, BTN_SIZE.md)}>
            {actions.verify.isPending ? "Verifying…" : `Verify ${plural(selected.length, "photo")}`}
          </button>
        </section>
      )}

      {isRunner(request) && !request.evidencePresent && <CancelCase request={request} disabled={busy || refreshing || reviewRequired} />}

      {preview && (
        <PhotoPreviewCard mode="review" eventName={request.event.name} index={preview.index + 1} total={preview.items.length}
          photo={{ ...preview.items[preview.index].photo, alt: preview.items[preview.index].photo.alt ?? undefined, photographerHandle: null }}
          onClose={() => setPreview(null)}
          onPrev={preview.index > 0 ? () => setPreview({ ...preview, index: preview.index - 1 }) : undefined}
          onNext={preview.index < preview.items.length - 1 ? () => setPreview({ ...preview, index: preview.index + 1 }) : undefined} />
      )}
    </div>
  );
}

// Groups by owner ID (names can repeat); a null ID is the unassigned group.
export function groupByOwner<T>(items: T[], owner: (item: T) => [string | null, string | null]): [string, string, T[]][] {
  const groups = new Map<string, [string, string, T[]]>();
  for (const item of items) {
    const [ownerId, name] = owner(item);
    const key = ownerId ?? "";
    const group = groups.get(key) ?? [key, ownerOf(ownerId, name), []];
    group[2].push(item);
    groups.set(key, group);
  }
  return [...groups.values()].sort(([a, nameA], [b, nameB]) => !a ? 1 : !b ? -1 : nameA.localeCompare(nameB));
}

function PhotoTile({ item, checked, disabled, onToggle, onPreview, position }: {
  item: RemovalPhotoDto; checked: boolean; disabled: boolean; onToggle: () => void; onPreview: () => void; position: string;
}) {
  const { photo } = item;
  return (
    <li className={cn("rounded-xl border p-2 space-y-2", checked ? "border-ink bg-fresh-tint/30" : "border-line")}>
      <button type="button" onClick={onPreview} aria-label={`Preview photo ${position}`} className="block w-full aspect-[3/4] overflow-hidden rounded-lg bg-bone-deep">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        {photo.imageUrl && <img src={photo.imageUrl} alt={photo.alt ?? ""} loading="lazy" className="h-full w-full object-cover" />}
      </button>
      <label className="flex items-start gap-2 text-sm text-ink cursor-pointer">
        <input type="checkbox" value={photo.id} checked={checked} disabled={disabled} onChange={onToggle}
          aria-label={`Include ${photo.bib ? `bib ${photo.bib}` : "untagged photo"} · ${photo.id}`} className="mt-1" />
        <span className="min-w-0">
          <span className={cn("block truncate", !item.photographerId && "text-amber-700")}>{ownerOf(item.photographerId, item.photographerName)}</span>
          <span className="block font-mono tnum text-slate">{photo.bib ?? "No bib"}{item.matchScore != null && <> · {Math.round(item.matchScore * 100)}%</>}</span>
          {(item.trashed || item.removalState !== "NONE" || !item.bytesAvailable) && (
            <span className="block text-amber-700">{item.removalState === "REMOVED" ? "Removed" : item.removalState === "REVIEW" ? "Under review" : item.trashed ? "In trash" : "File unavailable"}</span>
          )}
        </span>
      </label>
    </li>
  );
}

// Evidence bytes stay in memory: an object URL revoked when the image changes or unmounts.
function EvidencePreview({ requestId }: { requestId: string }) {
  const evidence = useRemovalEvidence(requestId);
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!evidence.data) return;
    const next = URL.createObjectURL(evidence.data);
    setUrl(next);
    return () => { URL.revokeObjectURL(next); setUrl(null); };
  }, [evidence.data]);
  return (
    <div className="flex items-center gap-4">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      {url ? <img src={url} alt="Runner selfie evidence" className="size-20 rounded-xl object-cover" /> : <span className="size-20 rounded-xl bg-bone-deep" aria-hidden="true" />}
      <div>
        <p className="font-display font-bold text-ink">Selfie on file</p>
        <p className="text-sm text-slate">Private case evidence. It is not added to anyone&apos;s selfie library.</p>
        {evidence.isError && <p className="text-sm text-amber-700">Preview unavailable.</p>}
      </div>
    </div>
  );
}

// A retry of the same action after a failure reuses its command ID; any changed action gets a new one.
function useCommandIds() {
  const commands = useRef(new Map<string, string>());
  return {
    commandFor: (fingerprint: string) => {
      if (!commands.current.has(fingerprint)) commands.current.set(fingerprint, crypto.randomUUID());
      return commands.current.get(fingerprint)!;
    },
    done: (fingerprint: string) => commands.current.delete(fingerprint),
  };
}

// R10: a runner case with no saved image can only be matched by hand or cancelled.
function CancelCase({ request, disabled }: { request: PhotoRemovalRequestDto; disabled: boolean }) {
  const actions = useAdminRemovalActions();
  const { confirm } = useConfirmation();
  const { commandFor, done } = useCommandIds();
  const [reason, setReason] = useState("");
  const [error, setError] = useState<string | null>(null);
  const busy = disabled || actions.cancel.isPending;
  async function cancel() {
    if (busy || !reason.trim()) return;
    if (!(await confirm({
      title: "Cancel this request?",
      message: "Nothing is hidden or removed. The runner sees the request as rejected.",
      confirmLabel: "Cancel case", danger: true,
    }))) return;
    setError(null);
    const body = { expectedVersion: request.version, reason: reason.trim() };
    const fingerprint = `cancel:${JSON.stringify(body)}`;
    try {
      await actions.cancel.mutateAsync({ id: request.id, command: { ...body, commandId: commandFor(fingerprint) } });
      done(fingerprint);
    } catch (failure) {
      setError(errorText(failure));
    }
  }
  return (
    <section aria-label="Cancel case" className="space-y-3 border-t border-line pt-8">
      <AdminFieldLabel htmlFor="removal-cancel-reason">Reason for cancelling</AdminFieldLabel>
      <input id="removal-cancel-reason" type="text" maxLength={2000} value={reason} disabled={busy} onChange={(e) => setReason(e.target.value)} className={REMOVAL_INPUT} />
      <FieldError message={error} id="removal-cancel-error" />
      <button type="button" disabled={busy || !reason.trim()} onClick={() => void cancel()} className={cn(BTN_DANGER, BTN_SIZE.sm)}>
        {actions.cancel.isPending ? "Cancelling…" : "Cancel case"}
      </button>
    </section>
  );
}

type SuggestionAction = "promote" | "dismiss" | "demote";

// RD-6 suggestion review on an open runner case. Promote hides and verifies; dismiss leaves the photo public;
// demote makes an auto-hidden photo public again as a suggestion. Decisions stay in RemovalRequestDecision.
function RunnerSuggestions({ request, refreshing, refresh }: { request: PhotoRemovalRequestDto; refreshing: boolean; refresh: RefreshCase }) {
  const actions = useAdminRemovalActions();
  const { commandFor, done } = useCommandIds();
  const [error, setError] = useState<string | null>(null);
  const hidden = request.items.filter((item) => item.autoHidden && item.state === "VERIFIED");
  const suggestions = request.items.filter((item) => item.state === "CANDIDATE");
  const busy = actions.reviewSuggestions.isPending || refreshing || actions.reviewRequired(request.id) || request.status !== "OPEN";

  async function review(action: SuggestionAction, itemId: string) {
    if (busy) return;
    setError(null);
    const body = { expectedVersion: request.version, promote: [] as string[], dismiss: [] as string[], demote: [] as string[], [action]: [itemId] };
    const fingerprint = `suggestions:${JSON.stringify(body)}`;
    try {
      await actions.reviewSuggestions.mutateAsync({ id: request.id, command: { ...body, commandId: commandFor(fingerprint) } });
      done(fingerprint);
    } catch (failure) {
      setError(errorText(failure));
    }
  }

  const row = (item: PhotoRemovalItemDto, buttons: ReactNode) => (
    <li key={item.id} className="flex flex-wrap items-center gap-3 rounded-xl border border-line p-3 text-sm">
      <span className="size-16 shrink-0 overflow-hidden rounded-lg bg-bone-deep">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        {item.photo?.photo.imageUrl && <img src={item.photo.photo.imageUrl} alt="" loading="lazy" className="h-full w-full object-cover" />}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block font-mono tnum text-ink truncate">{item.photoIdSnapshot}</span>
        <span className="block text-slate truncate">{ownerOf(item.ownerIdSnapshot, item.ownerNameSnapshot)}</span>
        {/* Unknown face counts count as other people (the count is a lower bound). */}
        {(item.otherPeopleVisible ?? true) && <span className="mt-1 inline-block"><AdminStatusPill label="Other people visible" tone="amber" /></span>}
      </span>
      <span className="flex w-full flex-wrap gap-2 sm:w-auto">{buttons}</span>
    </li>
  );

  return (
    <div className="mt-8 space-y-10">
      <ReviewCurrentState request={request} actions={actions} refreshing={refreshing} refresh={refresh} />
      <section aria-labelledby="runner-hidden" className="space-y-3">
        <h2 id="runner-hidden" className="font-display text-xl font-extrabold text-ink">Hidden automatically</h2>
        <p className="text-sm text-slate">Hidden from sale and search after the runner&apos;s live check. Nothing is deleted until you decide below.</p>
        {hidden.length === 0 ? <p className="text-sm text-slate">No photos were hidden automatically.</p> : (
          <ul className="space-y-2">
            {hidden.map((item) => row(item, item.adminDecision == null && (
              <button type="button" disabled={busy} onClick={() => void review("demote", item.id)} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>
                Make public again, keep as suggestion<span className="sr-only"> for {item.photoIdSnapshot}</span>
              </button>
            )))}
          </ul>
        )}
      </section>
      <section aria-labelledby="runner-suggestions" className="space-y-3">
        <h2 id="runner-suggestions" className="font-display text-xl font-extrabold text-ink">Suggestions (still public)</h2>
        <p className="text-sm text-slate">Possible matches that were not hidden. Promote hides a photo and sends it to its photographer for review; dismiss leaves it public.</p>
        {suggestions.length === 0 ? <p className="text-sm text-slate">No suggestions left.</p> : (
          <ul className="space-y-2">
            {suggestions.map((item) => row(item, <>
              <button type="button" disabled={busy} onClick={() => void review("promote", item.id)} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>
                Promote<span className="sr-only"> {item.photoIdSnapshot}</span>
              </button>
              <button type="button" disabled={busy} onClick={() => void review("dismiss", item.id)} className={cn(BTN_GHOST, BTN_SIZE.sm)}>
                Dismiss<span className="sr-only"> {item.photoIdSnapshot}</span>
              </button>
            </>))}
          </ul>
        )}
      </section>
      <FieldError message={error} id="runner-suggestions-error" />
    </div>
  );
}

// Per-owner responses and decisions live in RemovalRequestDecision; this keeps the verification record.
function VerifiedSummary({ request }: { request: PhotoRemovalRequestDto }) {
  return (
    <section aria-label="Verification" className="mt-8 space-y-1 text-sm">
      <p className="text-ink">Verified by: {request.verificationMethod ?? "—"}</p>
      <p className="text-slate">{request.verificationNote}</p>
      {request.verifiedAt && <p className="font-mono tnum text-slate">{formatLongDate(request.verifiedAt)}</p>}
    </section>
  );
}
