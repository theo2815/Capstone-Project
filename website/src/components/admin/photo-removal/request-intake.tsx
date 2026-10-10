"use client";

import { useRef, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { useAdminEvents } from "@/hooks/use-admin-events";
import { useAdminRemovalActions } from "@/hooks/use-photo-removal";
import { AdminFieldLabel } from "@/components/admin/admin-form-fields";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { FieldError } from "@/components/ui/field-error";
import { BTN_GHOST, BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ROUTES } from "@/lib/constants";
import { formatRaceDate } from "@/lib/format";
import { cn } from "@/lib/utils";

const EVENT_PAGE = 20;
export const REMOVAL_INPUT = "w-full rounded-2xl border border-line bg-surface px-4 py-3 font-sans text-sm text-ink focus:outline-none focus:border-fresh";

// Support intake only: contact and reason live in component state and go
// straight to the backend; nothing here is written to storage or the URL.
export function RemovalRequestIntake() {
  const router = useRouter();
  const { create } = useAdminRemovalActions();
  const [offset, setOffset] = useState(0);
  const events = useAdminEvents({ offset, limit: EVENT_PAGE });
  const [draft, setDraft] = useState({ eventId: "", supportReference: "", requesterContact: "", requesterUserId: "", reason: "", adminNote: "" });
  const [error, setError] = useState<string | null>(null);
  // The same draft retried after a failure keeps its command ID so the server can deduplicate it.
  const attempt = useRef<{ fingerprint: string; commandId: string } | null>(null);
  const set = (field: keyof typeof draft) => (value: string) => setDraft((current) => ({ ...current, [field]: value }));
  const ready = draft.eventId && draft.supportReference.trim() && draft.requesterContact.trim() && draft.reason.trim();

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!ready || create.isPending) return;
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
      setError(failure instanceof Error ? failure.message : "Couldn't create the case. Try again.");
    }
  }

  const rows = events.data ?? [];
  return (
    <form onSubmit={submit} className="mt-8 space-y-10 max-w-2xl">
      <fieldset className="min-w-0 space-y-4" disabled={create.isPending}>
        <legend className="font-display text-xl font-extrabold text-ink">1. Event</legend>
        <p className="text-sm text-slate">One event per case. Public and unlisted events can both be chosen.</p>
        {events.isPending && <p role="status" className="text-sm text-slate">Loading events…</p>}
        {events.isError && <p role="alert" className="text-sm text-error">Couldn&apos;t load events. <button type="button" className={cn(BTN_GHOST, "underline")} onClick={() => void events.refetch()}>Try again</button></p>}
        {events.isSuccess && rows.length === 0 && <p className="text-sm text-slate">No events on this page.</p>}
        <ul className="divide-y divide-line border-y border-line">
          {rows.map((row) => (
            <li key={row.id}>
              <label className={cn("flex items-center gap-3 py-3 px-2 cursor-pointer hover:bg-bone-deep/50", draft.eventId === row.id && "bg-fresh-tint/30")}>
                <input type="radio" name="eventId" value={row.id} checked={draft.eventId === row.id} onChange={() => set("eventId")(row.id)} />
                <span className="min-w-0 flex-1">
                  <span className="block font-display font-bold text-ink truncate">{row.name}</span>
                  <span className="block font-mono tnum text-sm text-slate">{formatRaceDate(row.date)} · {row.city}</span>
                </span>
                <AdminStatusPill label={row.visibility === "unlisted" ? "UNLISTED" : "PUBLIC"} tone={row.visibility === "unlisted" ? "amber" : "slate"} />
              </label>
            </li>
          ))}
        </ul>
        <div className="flex gap-2">
          <button type="button" className={cn(BTN_GHOST, BTN_SIZE.sm)} disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - EVENT_PAGE))}>← Previous</button>
          <button type="button" className={cn(BTN_GHOST, BTN_SIZE.sm)} disabled={rows.length < EVENT_PAGE} onClick={() => setOffset(offset + EVENT_PAGE)}>Next →</button>
        </div>
      </fieldset>

      <fieldset className="space-y-5" disabled={create.isPending}>
        <legend className="font-display text-xl font-extrabold text-ink">2. Request</legend>
        <p className="text-sm text-slate">Admin-only. Photographers see a generic privacy reason, never these details.</p>
        <Field id="removal-support-reference" label="Support reference" value={draft.supportReference} onChange={set("supportReference")} max={1000} />
        <Field id="removal-requester-contact" label="Requester contact" value={draft.requesterContact} onChange={set("requesterContact")} max={1000} />
        <Field id="removal-requester-user" label="Runner account ID (optional)" value={draft.requesterUserId} onChange={set("requesterUserId")} max={64} />
        <Field id="removal-reason" label="Reason" value={draft.reason} onChange={set("reason")} max={2000} multiline />
        <Field id="removal-admin-note" label="Admin note (optional)" value={draft.adminNote} onChange={set("adminNote")} max={2000} multiline />
      </fieldset>

      <FieldError message={error} id="removal-intake-error" />
      <button type="submit" disabled={!ready || create.isPending} className={cn(BTN_PRIMARY, BTN_SIZE.md)}>
        {create.isPending ? "Creating…" : "Create case"}
      </button>
    </form>
  );
}

function Field({ id, label, value, onChange, max, multiline = false }: {
  id: string; label: string; value: string; onChange: (value: string) => void; max: number; multiline?: boolean;
}) {
  return (
    <div className="space-y-2">
      <AdminFieldLabel htmlFor={id}>{label}</AdminFieldLabel>
      {multiline
        ? <textarea id={id} rows={3} maxLength={max} value={value} onChange={(e) => onChange(e.target.value)} className={REMOVAL_INPUT} />
        : <input id={id} type="text" autoComplete="off" maxLength={max} value={value} onChange={(e) => onChange(e.target.value)} className={REMOVAL_INPUT} />}
    </div>
  );
}
