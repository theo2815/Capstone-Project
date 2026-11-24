"use client";

import { useState } from "react";
import { AdminConfirmModal, AdminFormModal } from "@/components/admin/admin-form-modal";
import { AdminTextarea } from "@/components/admin/admin-form-fields";
import { AdminStatusTabs } from "@/components/admin/admin-status-tabs";
import { Kicker } from "@/components/ui/kicker";
import { Skeleton } from "@/components/ui/skeleton";
import { LoadError } from "@/components/ui/load-error";
import { LogoPlaceholder } from "@/components/ui/logo-placeholder";
import {
  BTN_PRIMARY,
  BTN_SECONDARY,
  BTN_SIZE,
} from "@/components/ui/button-styles";
import Link from "next/link";
import { useAdminEvents, useAdminEventActions } from "@/hooks/use-admin-events";
import { useRaceTypes } from "@/hooks/use-race-types";
import { useRegions } from "@/hooks/use-regions";
import { useToast } from "@/hooks/use-toast";
import { useUrlState } from "@/hooks/use-url-state";
import { isRaceEdit, type AdminEventRow, type RaceEditChange } from "@/lib/api-admin";
import type { RaceCandidate } from "@/lib/api-photographer";
import { formatLongDate, sentenceCaseDate } from "@/lib/format";
import { getRegion } from "@/lib/ph-regions";
import { describePricing, raceCheckReasonLine } from "@/lib/photographer-events";
import { cn } from "@/lib/utils";

// Event requests queue (V46) — photographer-owned events waiting on an
// admin: a new submission (the event is a draft nobody can see) or a pricing
// change parked on a live event (the gallery keeps its current settings until
// approved). Approve / Send back land on the same two BE endpoints; the
// service decides which state it is in. Price edits before approval go
// through the existing /admin/events board.

export function EventRequestsQueue() {
  const query = useAdminEvents({ review: "queue" });
  const rows = query.data;
  const { approve, reject, sameRace, isPending, pendingAction } = useAdminEventActions();
  const { showToast } = useToast();
  const [rejecting, setRejecting] = useState<AdminEventRow | null>(null);
  const [reason, setReason] = useState("");
  // "Same race" merges and deletes a request, so it asks first.
  // Only the id is kept: confirm reads the row's current revision, so a change made meanwhile is not a conflict.
  const [joiningState, setJoining] = useState<{ id: string; target: RaceCandidate } | null>(null);
  const joiningRow = joiningState && rows?.find((r) => r.id === joiningState.id);
  const joining = joiningState && joiningRow ? { row: joiningRow, target: joiningState.target } : null;
  // ER2: Approve asks first; like the join, only the id is kept so confirm sends the current revision.
  const [approvingId, setApprovingId] = useState<string | null>(null);
  const approving = (approvingId && rows?.find((r) => r.id === approvingId)) || null;
  const [kindRaw, setKind] = useUrlState<string>("kind", "all");
  const kind: RequestKind = kindRaw === "new" || kindRaw === "changes" ? kindRaw : "all";
  async function run(action: () => Promise<boolean>, message: string) {
    try {
      const ok = await action();
      if (ok) showToast({ kind: "success", message });
      return ok;
    } catch {
      // The hook surfaces backend errors and reconciles conflicting reviews.
      return false;
    }
  }

  async function submitReject() {
    const row = rejecting;
    const text = reason.trim();
    if (!row || !text) return;
    const ok = await run(
      () => reject(row.id, text, row.reviewRevision),
      row.reviewStatus === "change_pending"
        ? `Change declined — ${row.name} keeps its current ${isRaceEdit(row.pendingChange) ? "details" : "pricing"}.`
        : `${row.name} sent back to the photographer.`,
    );
    if (ok) {
      setRejecting(null);
      setReason("");
    }
  }

  async function confirmJoin() {
    if (!joining) return;
    const { row, target } = joining;
    const ok = await run(
      () => sameRace(row.id, target.id, row.reviewRevision),
      `${row.submittedByName ?? "The photographer"} joined ${target.name}. The duplicate request is removed.`,
    );
    if (ok) setJoining(null);
  }

  async function confirmApprove() {
    const row = approving;
    if (!row) return;
    const ok = await run(
      () => approve(row.id, row.reviewRevision),
      row.reviewStatus === "change_pending"
        ? `Change applied to ${row.name}.`
        : `${row.name} is live. Uploads are open.`,
    );
    if (ok) setApprovingId(null);
  }

  const changes = rows?.filter((r) => r.reviewStatus === "change_pending").length ?? 0;
  const shown = rows?.filter((r) => kind === "all" || (r.reviewStatus === "change_pending") === (kind === "changes"));

  return (
    // ER4: the page header (or the Inbox chip) already names the queue; the h2 stays for screen readers.
    <section id="event-requests" aria-labelledby="event-requests-heading">
      <h2 id="event-requests-heading" className="sr-only">Event requests</h2>
      <p className="font-sans text-sm text-slate mt-6 md:mt-8">
        Photographer-owned events · new submissions and pricing changes
      </p>
      {query.isError && <LoadError what="event requests" onRetry={() => void query.refetch()} className="text-sm text-error" />}
      {query.isPending ? (
        <div aria-busy="true" className="space-y-4">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-28 w-full rounded-2xl" />
          ))}
        </div>
      ) : !rows ? null : rows.length === 0 ? (
        <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
          <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
            No event requests waiting. You&apos;re caught up.
          </p>
          <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
            New photographer events and pricing change requests land here.
          </p>
        </div>
      ) : (
        <>
          <AdminStatusTabs
            label="Request kind"
            tabs={[
              { key: "all", label: "All", count: rows.length },
              { key: "new", label: "New", count: rows.length - changes },
              { key: "changes", label: "Changes", count: changes },
            ]}
            value={kind}
            onChange={setKind}
          />
          {shown?.length === 0 ? (
            <p className="font-sans text-sm text-slate-soft pt-6">Nothing of this kind is waiting.</p>
          ) : (
            <ul className="divide-y divide-line border-b border-line">
              {shown?.map((row) => (
                <RequestRow
                  key={row.id}
                  row={row}
                  busy={isPending(row.id)}
                  pendingLabel={pendingAction(row.id)}
                  onApprove={() => setApprovingId(row.id)}
                  onReject={() => setRejecting(row)}
                  onSameRace={(target) => setJoining({ id: row.id, target })}
                />
              ))}
            </ul>
          )}
        </>
      )}

      <AdminConfirmModal
        open={approving !== null}
        onClose={() => setApprovingId(null)}
        onConfirm={() => void confirmApprove()}
        title={approving ? approveTitle(approving) : ""}
        confirmLabel={approving && isDuplicate(approving) ? "Approve as a new race" : "Approve"}
        submitting={!!approving && isPending(approving.id)}
        pendingLabel="Approving…"
      >
        {approving && <ApproveSummary row={approving} />}
      </AdminConfirmModal>

      <AdminConfirmModal
        open={joining !== null}
        onClose={() => setJoining(null)}
        onConfirm={() => void confirmJoin()}
        title="Join the existing race?"
        confirmLabel="Join race"
        submitting={!!joining && isPending(joining.row.id)}
        pendingLabel="Joining…"
      >
        {joining && (
          <p className="font-sans text-sm text-ink-soft">
            {joining.row.submittedByName ?? "The photographer"} will cover{" "}
            <strong className="text-ink">{joining.target.name}</strong>, and
            their request &ldquo;{joining.row.name}&rdquo; is removed.
          </p>
        )}
      </AdminConfirmModal>

      <AdminFormModal
        tone="danger"
        open={rejecting !== null}
        onClose={() => {
          setRejecting(null);
          setReason("");
        }}
        onSubmit={submitReject}
        title={
          rejecting?.reviewStatus === "change_pending"
            ? "Decline the change"
            : "Send back for fixes"
        }
        intro={
          rejecting
            ? `${rejecting.createdByName ?? rejecting.submittedByName ?? "The photographer"} sees your reason in their inbox. Write what they should change.`
            : undefined
        }
        submitLabel={
          rejecting?.reviewStatus === "change_pending" ? "Decline" : "Send back"
        }
        submitting={!!rejecting && isPending(rejecting.id)}
        pendingLabel={rejecting?.reviewStatus === "change_pending" ? "Declining…" : "Sending back…"}
        submitDisabled={reason.trim().length === 0}
      >
        <AdminTextarea
          id="event-reject-reason"
          label="Reason (max 500)"
          value={reason}
          onChange={setReason}
          maxLength={500}
          placeholder="What should they change before resubmitting?"
          hint="Seen by: the photographer."
        />
      </AdminFormModal>
    </section>
  );
}

type RequestKind = "all" | "new" | "changes";

// The same test RequestRow uses to show the duplicate's "Same race" choice.
const isDuplicate = (row: AdminEventRow) =>
  row.reviewStatus !== "change_pending" && !!row.eventType && !!row.possibleDuplicateOf;

const visibilityLabel = (row: AdminEventRow) =>
  row.visibility === "unlisted" ? "Unlisted · link only" : "Public · link and profile";

function approveTitle(row: AdminEventRow) {
  if (row.reviewStatus !== "change_pending") return isDuplicate(row) ? "Approve as a new race?" : "Approve this event?";
  return isRaceEdit(row.pendingChange) ? "Apply this race edit?" : "Apply this pricing change?";
}

// What the approval makes public, or what it changes, in the admin's words before the click.
function ApproveSummary({ row }: { row: AdminEventRow }) {
  const edit = isRaceEdit(row.pendingChange) ? row.pendingChange : null;
  const price = row.pendingChange && !isRaceEdit(row.pendingChange) ? row.pendingChange : null;
  const current = describePricing({
    pricingMode: row.pricingMode,
    pricePerPhoto: row.pricePerPhoto ?? 0,
    watermarkPolicy: row.watermarkPolicy,
  });
  if (row.reviewStatus === "change_pending") {
    return (
      <p className="font-sans text-sm text-ink-soft tnum">
        {edit ? (
          <>
            <strong className="text-ink">{row.name}</strong> takes the new{" "}
            {[
              ...EDIT_FIELDS.filter(([f]) => edit.fields[f] !== undefined).map(([, label]) => label.toLowerCase()),
              ...(edit.coverS3Key ? ["cover photo"] : []),
            ].join(", ")}.
          </>
        ) : price ? (
          <>
            <strong className="text-ink">{row.name}</strong> changes from {current} to{" "}
            <strong className="text-ink">{describePricing(price)}</strong>.
          </>
        ) : (
          <>The requested change replaces what <strong className="text-ink">{row.name}</strong> shows now.</>
        )}
      </p>
    );
  }
  return (
    <div className="space-y-2 font-sans text-sm text-ink-soft tnum">
      <p>
        <strong className="text-ink">{row.name}</strong> goes live and uploads open.
        {isDuplicate(row) && <> It stays its own race, apart from {row.possibleDuplicateOf!.name}.</>}
      </p>
      <p>
        Visibility: {visibilityLabel(row)}. Pricing: {current}.
      </p>
    </div>
  );
}

// ER3: a change request waits from its request time, a new submission from its last resend, else createdAt
// (both absent on an older backend).
function waitingLine(row: AdminEventRow) {
  const at = row.reviewStatus === "change_pending" ? row.pendingChange?.requestedAt : (row.resubmittedAt ?? row.createdAt);
  if (!at) return null;
  const days = Math.floor((Date.now() - new Date(at).getTime()) / 86_400_000);
  if (Number.isNaN(days) || days < 0) return null;
  return days === 0 ? "Waiting since today" : `Waiting ${days} ${days === 1 ? "day" : "days"}`;
}

function RequestRow({
  row,
  busy,
  pendingLabel,
  onApprove,
  onReject,
  onSameRace,
}: {
  row: AdminEventRow;
  busy: boolean;
  pendingLabel: string | null;
  onApprove: () => void;
  onReject: () => void;
  onSameRace: (target: RaceCandidate) => void;
}) {
  const isChange = row.reviewStatus === "change_pending";
  const heldEdit = isRaceEdit(row.pendingChange) ? row.pendingChange : null;
  const duplicate = isDuplicate(row) ? row.possibleDuplicateOf! : null;
  const waiting = waitingLine(row);
  const current = describePricing({
    pricingMode: row.pricingMode,
    pricePerPhoto: row.pricePerPhoto ?? 0,
    watermarkPolicy: row.watermarkPolicy,
  });
  // A shared race has no owner; its submitter (AA-R3) stands in.
  const handle = row.createdByHandle ?? row.submittedByHandle;
  const name = row.createdByName ?? row.submittedByName;
  const owner = handle ? `@${handle}` : (name ?? "Photographer");
  const photographer =
    [name, handle && `@${handle}`].filter(Boolean).join(" · ") || "Unknown";
  const kind = isChange
    ? heldEdit ? "Race edit · live race" : "Pricing change · live event"
    : row.eventType ? "New race" : "New event";

  return (
    // Side by side only from lg: at md the admin sidebar leaves too little room.
    <li className="py-5 md:py-6 flex flex-col lg:flex-row lg:items-start gap-4 lg:gap-8">
      <div className="flex-1 min-w-0">
        <div className="mb-4 w-full sm:w-56 aspect-[16/9] rounded-xl overflow-hidden border border-line bg-bone-deep">
          {row.bannerUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={row.bannerUrl}
              alt={`Cover photo for ${row.name}`}
              loading="lazy"
              decoding="async"
              className="w-full h-full object-cover"
            />
          ) : (
            <LogoPlaceholder size="md" label="No cover photo yet" className="h-full" />
          )}
        </div>
        <p className="font-sans text-sm text-slate-soft tnum">
          {kind} · {owner}{waiting && <> · {waiting}</>}
        </p>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2 break-words">
          {row.name}
        </p>
        <p className="font-sans text-sm text-ink-soft mt-1 tnum">
          {sentenceCaseDate(formatLongDate(row.date, true))} · {row.location}
        </p>
        {isChange && row.pendingChange && !isRaceEdit(row.pendingChange) && (
          <p className="font-sans text-sm text-ink mt-3 tnum">
            <span className="text-slate">Now</span> {current}
            <span className="text-slate-soft mx-2">→</span>
            <span className="text-slate">Requested</span>{" "}
            {describePricing(row.pendingChange)}
          </p>
        )}
        {heldEdit && <HeldRaceEdit row={row} change={heldEdit} />}
        {!!row.checkReasons?.length && (
          <ul className="mt-3 space-y-1 list-disc pl-5 font-sans text-sm text-ink">
            {row.checkReasons.map((code) => <li key={code}>{raceCheckReasonLine(code)}</li>)}
          </ul>
        )}
        {duplicate && <RaceRef label="Possible duplicate of" race={duplicate} />}
        {!row.eventType && row.lookalike && <RaceRef label="Looks like a shared race:" race={row.lookalike} />}
        <dl className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-y-3 gap-x-8">
          <Detail label="Photographer" value={photographer} />
          <Detail label="Organizer" value={row.organizerName || "Not given"} />
          <Detail
            label="Visibility"
            value={visibilityLabel(row)}
          />
          {!isChange && <Detail label="Pricing" value={current} />}
          <div className="sm:col-span-2">
            <Kicker as="dt" tone="soft">Race day notes</Kicker>
            <dd className="mt-1 font-sans text-sm text-ink whitespace-pre-line break-words max-w-prose">
              {row.description || "None given"}
            </dd>
          </div>
        </dl>
      </div>
      {/* Capped so a duplicate's three buttons wrap instead of squeezing the details to one word a line. */}
      <div className="flex flex-wrap gap-2 shrink-0 lg:max-w-64 lg:justify-end">
        <button
          type="button"
          onClick={onApprove}
          disabled={busy}
          aria-busy={busy}
          className={cn(BTN_PRIMARY, BTN_SIZE.sm)}
        >
          {pendingLabel === "Approving…" ? pendingLabel : duplicate ? "Approve as a new race" : "Approve"}
        </button>
        {duplicate && (
          <button
            type="button"
            onClick={() => onSameRace(duplicate)}
            disabled={busy}
            aria-busy={busy}
            className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
          >
            {pendingLabel === "Joining…" ? pendingLabel : "Same race as an existing one…"}
          </button>
        )}
        <button
          type="button"
          onClick={onReject}
          disabled={busy}
          className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
        >
          {isChange ? "Decline" : "Send back"}
        </button>
      </div>
    </li>
  );
}

const EDIT_FIELDS: ReadonlyArray<[keyof RaceEditChange["fields"], string]> = [
  ["title", "Title"], ["date", "Date"], ["location", "Location"], ["regionCode", "Region"],
  ["provinceCode", "Province"], ["eventType", "Race type"], ["organizerName", "Organizer"],
  ["description", "Race day notes"],
];

// AA-R5 held edit: each changed field beside its live value. The race keeps
// the live values until an admin approves.
function HeldRaceEdit({ row, change }: { row: AdminEventRow; change: RaceEditChange }) {
  const { types } = useRaceTypes();
  const { regions } = useRegions();
  const live: Record<keyof RaceEditChange["fields"], string | null | undefined> = {
    title: row.name, date: row.date, location: row.location, regionCode: row.regionCode,
    provinceCode: row.provinceCode, eventType: row.eventType, organizerName: row.organizerName,
    description: row.description,
  };
  // Codes only on the wire; labels come from the backend's reference data.
  const show = (field: keyof RaceEditChange["fields"], value: string | null | undefined) => {
    if (!value) return "None";
    if (field === "eventType") return types.find((t) => t.code === value)?.label ?? value;
    if (field === "regionCode") return getRegion(regions, value)?.name ?? value;
    if (field === "provinceCode") {
      const region = getRegion(regions, change.fields.regionCode ?? row.regionCode ?? "");
      return region?.provinces.find((p) => p.code === value)?.name ?? value;
    }
    return value;
  };
  return (
    <dl className="mt-3 space-y-2">
      {EDIT_FIELDS.filter(([field]) => change.fields[field] !== undefined).map(([field, label]) => (
        <div key={field} className="min-w-0">
          <Kicker as="dt" tone="soft">{label}</Kicker>
          <dd className="mt-1 font-sans text-sm text-ink tnum break-words whitespace-pre-line">
            <span className="text-slate">{show(field, live[field])}</span>
            <span className="text-slate-soft mx-2">→</span>
            {show(field, change.fields[field])}
          </dd>
        </div>
      ))}
      {change.coverS3Key && (
        <div>
          <Kicker as="dt" tone="soft">Cover</Kicker>
          <dd className="mt-1 font-sans text-sm text-ink">
            {row.heldCoverUrl ? (
              <div className="w-full sm:w-56 aspect-[16/9] rounded-xl overflow-hidden border border-line bg-bone-deep">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={row.heldCoverUrl}
                  alt={`New cover photo for ${row.name}`}
                  loading="lazy"
                  decoding="async"
                  className="w-full h-full object-cover"
                />
              </div>
            ) : (
              "A new cover photo"
            )}
          </dd>
        </div>
      )}
    </dl>
  );
}

// A race the server matched. Only a live race has a public page to open.
function RaceRef({ label, race }: { label: string; race: RaceCandidate }) {
  return (
    <p className="font-sans text-sm text-ink mt-3 tnum break-words">
      <span className="text-slate">{label}</span>{" "}
      {race.reviewStatus === "approved" ? (
        <Link href={`/events/${race.slug}`} target="_blank" rel="noreferrer" className="underline decoration-line-strong underline-offset-4 hover:decoration-fresh">
          {race.name}
        </Link>
      ) : (
        <>{race.name} (waiting for review)</>
      )}{" "}
      · {sentenceCaseDate(formatLongDate(race.date, true))} · {race.location} · {race.strength === "strong" ? "strong match" : "possible match"}
    </p>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <Kicker as="dt" tone="soft">{label}</Kicker>
      <dd className="mt-1 font-sans text-sm text-ink tnum break-words">{value}</dd>
    </div>
  );
}
