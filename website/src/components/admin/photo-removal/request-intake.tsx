"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { useAdminEventsInfinite } from "@/hooks/use-admin-events";
import { useAdminRemovalActions } from "@/hooks/use-photo-removal";
import { AdminFieldHint, AdminTextInput, AdminTextarea } from "@/components/admin/admin-form-fields";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { FieldError } from "@/components/ui/field-error";
import { LoadError } from "@/components/ui/load-error";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Skeleton } from "@/components/ui/skeleton";
import { BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import type { AdminEventRow } from "@/lib/api-admin";
import { ROUTES } from "@/lib/constants";
import { formatRaceDate, sentenceCaseDate } from "@/lib/format";
import { cn } from "@/lib/utils";

// The backend can't read any other shape, and its reply then names no field.
const RUNNER_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const RUNNER_ID_ERROR = "Runner account ID must be the 36-character account ID, like 1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed. Leave it empty if you don't have it.";
// The same box as AdminTextInput, for the event search (a hidden label and an Enter guard).
const REMOVAL_INPUT = "w-full rounded-2xl border border-line bg-surface px-4 py-3 font-sans text-sm text-ink focus:outline-none focus:ring-2 focus:ring-fresh focus:border-fresh";

// Support intake only: contact and reason live in component state and go
// straight to the backend; nothing here is written to storage or the URL.
export function RemovalRequestIntake() {
  const router = useRouter();
  const { create } = useAdminRemovalActions();
  const [search, setSearch] = useState("");
  const [q, setQ] = useState("");
  useEffect(() => { const timer = setTimeout(() => setQ(search.trim()), 300); return () => clearTimeout(timer); }, [search]);
  // AX-37 (RM5): the server searches every event by name and place; the backend has no review-status filter.
  const events = useAdminEventsInfinite(undefined, q);
  // Kept so a search that no longer lists the chosen event still shows it.
  const [picked, setPicked] = useState<AdminEventRow | null>(null);
  const [draft, setDraft] = useState({ eventId: "", supportReference: "", requesterContact: "", requesterUserId: "", reason: "", adminNote: "" });
  const [error, setError] = useState<string | null>(null);
  // The same draft retried after a failure keeps its command ID so the server can deduplicate it.
  const attempt = useRef<{ fingerprint: string; commandId: string } | null>(null);
  const set = (field: keyof typeof draft) => (value: string) => setDraft((current) => ({ ...current, [field]: value }));
  const missing = [
    !draft.eventId && "an event", !draft.supportReference.trim() && "support reference",
    !draft.requesterContact.trim() && "requester contact", !draft.reason.trim() && "reason",
  ].filter(Boolean);
  const ready = missing.length === 0;

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!ready || create.isPending) return;
    const runnerId = draft.requesterUserId.trim();
    if (runnerId && !RUNNER_ID.test(runnerId)) { setError(RUNNER_ID_ERROR); return; }
    const fingerprint = JSON.stringify(draft);
    if (attempt.current?.fingerprint !== fingerprint) attempt.current = { fingerprint, commandId: crypto.randomUUID() };
    setError(null);
    try {
      const created = await create.mutateAsync({
        eventId: draft.eventId, supportReference: draft.supportReference.trim(), requesterContact: draft.requesterContact.trim(),
        requesterUserId: draft.requesterUserId.trim() || null, reason: draft.reason.trim(), adminNote: draft.adminNote.trim() || null,
        commandId: attempt.current.commandId,
      });
      router.push(`${ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS}/${encodeURIComponent(created.id)}`);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Couldn't create the request. Try again.");
    }
  }

  // Pending and sent-back events never went live, so they hold no photos to remove.
  const live = events.items.filter((row) => row.reviewStatus === "approved" || row.reviewStatus === "change_pending");
  const rows = picked && !live.some((row) => row.id === picked.id) ? [picked, ...live] : live;
  return (
    <form onSubmit={submit} className="mt-8 space-y-10 max-w-2xl">
      <fieldset className="min-w-0 space-y-4" disabled={create.isPending}>
        <legend className="font-display text-xl font-extrabold text-ink">1. Event</legend>
        <p className="text-sm text-slate">One event per request. Only approved events are listed; public and unlisted ones can both be chosen.</p>
        <label htmlFor="removal-event-search" className="sr-only">Search events</label>
        {/* Enter here would submit the whole case form. */}
        <input id="removal-event-search" type="search" value={search} onChange={(e) => setSearch(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") e.preventDefault(); }} placeholder="Search by event name or place" className={REMOVAL_INPUT} />
        {events.isLoading && (
          <div role="status" aria-busy="true" aria-label="Loading events" className="space-y-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-14 w-full" />)}</div>
        )}
        {!!events.error && <LoadError what="events" onRetry={() => void events.refetch()} />}
        {!events.isLoading && !events.error && live.length === 0 && (
          <p className="text-sm text-slate">{events.hasNextPage ? "No approved events among those loaded so far." : q ? "No approved event matches this search." : "No approved events yet."}</p>
        )}
        <ul className="divide-y divide-line border-y border-line">
          {rows.map((row) => (
            <li key={row.id}>
              <label className={cn("flex items-center gap-3 py-3 px-2 cursor-pointer hover:bg-bone-deep/50", draft.eventId === row.id && "bg-fresh-tint/30")}>
                <input type="radio" name="eventId" value={row.id} checked={draft.eventId === row.id} onChange={() => { setPicked(row); set("eventId")(row.id); }} />
                <span className="min-w-0 flex-1">
                  <span className="block font-display font-bold text-ink truncate">{row.name}</span>
                  <span className="block font-mono tnum text-sm text-slate">{sentenceCaseDate(formatRaceDate(row.date))} · {row.city}</span>
                </span>
                <AdminStatusPill label={row.visibility === "unlisted" ? "UNLISTED" : "PUBLIC"} tone={row.visibility === "unlisted" ? "amber" : "slate"} />
              </label>
            </li>
          ))}
        </ul>
        <LoadMoreButton shown={events.items.length} total={events.total} increment={50} isLoading={events.isFetchingNextPage} onLoadMore={events.fetchNextPage}
          countLabel={`Looked through ${events.items.length.toLocaleString()} of ${events.total.toLocaleString()} events`} className="mt-4 md:mt-6" />
      </fieldset>

      <fieldset className="space-y-5" disabled={create.isPending}>
        <legend className="font-display text-xl font-extrabold text-ink">2. Request</legend>
        <p className="text-sm text-slate">Admin-only. Photographers see a generic privacy reason, never these details.</p>
        <AdminTextInput id="removal-support-reference" label="Support reference" value={draft.supportReference} onChange={set("supportReference")} maxLength={1000}
          hint="The ticket or message number from your support inbox." hintTone="calm" />
        <AdminTextInput id="removal-requester-contact" label="Requester contact" value={draft.requesterContact} onChange={set("requesterContact")} maxLength={1000}
          hint="The email or phone number the person wrote from." hintTone="calm" />
        <AdminTextInput id="removal-requester-user" label="Runner account ID (optional)" value={draft.requesterUserId} onChange={set("requesterUserId")} maxLength={64}
          hint="The runner's 36-character account ID, not their email or name." hintTone="calm" error={error === RUNNER_ID_ERROR ? error : null} />
        <div className="space-y-2">
          <AdminTextarea id="removal-reason" label="Reason" maxLength={2000} value={draft.reason} onChange={set("reason")} describedBy="removal-reason-hint" />
          <AdminFieldHint id="removal-reason-hint" tone="calm">What the person asked for, in their own words. Only admins see this.</AdminFieldHint>
        </div>
        <AdminTextarea id="removal-admin-note" label="Admin note (optional)" maxLength={2000} value={draft.adminNote} onChange={set("adminNote")} />
      </fieldset>

      {/* The runner ID error shows under its own field. */}
      <FieldError message={error === RUNNER_ID_ERROR ? null : error} id="removal-intake-error" />
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <button type="submit" disabled={!ready || create.isPending} aria-describedby={ready ? undefined : "removal-intake-missing"} className={cn(BTN_PRIMARY, BTN_SIZE.md)}>
          {create.isPending ? "Creating…" : "Create request"}
        </button>
        {!ready && <p id="removal-intake-missing" className="text-sm text-slate">Still needed: {missing.join(", ")}.</p>}
      </div>
    </form>
  );
}
