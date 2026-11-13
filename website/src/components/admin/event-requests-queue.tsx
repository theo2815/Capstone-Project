"use client";

import { useState } from "react";
import { Slab } from "@/components/profile-shell";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import { AdminTextarea } from "@/components/admin/admin-form-fields";
import { Kicker } from "@/components/ui/kicker";
import { Skeleton } from "@/components/ui/skeleton";
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
import { isRaceEdit, type AdminEventRow, type RaceEditChange } from "@/lib/api-admin";
import type { RaceCandidate } from "@/lib/api-photographer";
import { formatLongDate } from "@/lib/format";
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

  const waiting = rows?.length ?? 0;

  return (
    <Slab
      id="event-requests"
      number="01"
      title="Event requests"
      caption="Photographer-owned events · new submissions and pricing changes"
      trailing={rows ? `${waiting} waiting` : undefined}
    >
      {query.isError && (
        <div role="alert" className="space-y-3">
          <p className="font-sans text-sm text-error">Couldn&apos;t load event requests. Please try again.</p>
          <button type="button" onClick={() => void query.refetch()} disabled={query.isFetching} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>Retry</button>
        </div>
      )}
      {query.isPending ? (
        <div className="space-y-4">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-28 w-full rounded-2xl" />
          ))}
        </div>
      ) : !rows ? null : rows.length === 0 ? (
        <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
          <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
            Nothing waiting.
          </p>
          <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
            New photographer events and pricing change requests land here.
          </p>
        </div>
      ) : (
        <ul className="divide-y divide-line border-y border-line">
          {rows.map((row) => (
            <RequestRow
              key={row.id}
              row={row}
              busy={isPending(row.id)}
              pendingLabel={pendingAction(row.id)}
              onApprove={() =>
                run(
                  () => approve(row.id, row.reviewRevision),
                  row.reviewStatus === "change_pending"
                    ? `Change applied to ${row.name}.`
                    : `${row.name} is live. Uploads are open.`,
                )
              }
              onReject={() => setRejecting(row)}
              onSameRace={(target) =>
                run(
                  () => sameRace(row.id, target.id, row.reviewRevision),
                  `${row.submittedByName ?? "The photographer"} joined ${target.name}. The duplicate request is removed.`,
                )
              }
            />
          ))}
        </ul>
      )}

      <AdminFormModal
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
            ? `${rejecting.createdByName ?? rejecting.submittedByName ?? "The photographer"} sees your reason in their inbox.`
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
        />
      </AdminFormModal>
    </Slab>
  );
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
  const duplicate = !isChange && row.eventType ? row.possibleDuplicateOf : null;
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
            <p className="h-full flex items-center justify-center font-sans text-sm text-slate">
              No cover photo
            </p>
          )}
        </div>
        <Kicker as="p" tone="soft" tnum>
          {kind} · {owner}
        </Kicker>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2 break-words">
          {row.name}
        </p>
        <p className="font-sans text-sm text-ink-soft mt-1 tnum">
          {formatLongDate(row.date, true)} · {row.location}
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
            value={row.visibility === "unlisted" ? "Unlisted · link only" : "Public · link and profile"}
          />
          {!isChange && <Detail label="Pricing" value={current} />}
          <div className="sm:col-span-2">
            <dt className={DETAIL_LABEL}>Race day notes</dt>
            <dd className="mt-1 font-sans text-sm text-ink whitespace-pre-line break-words max-w-prose">
              {row.description || "None given"}
            </dd>
          </div>
        </dl>
      </div>
      <div className="flex flex-wrap gap-2 shrink-0">
        <button
          type="button"
          onClick={onApprove}
          disabled={busy}
          aria-busy={busy}
          className={cn(BTN_PRIMARY, BTN_SIZE.sm)}
        >
          {pendingLabel === "Approving…" ? pendingLabel : duplicate ? "Different race" : "Approve"}
        </button>
        {duplicate && (
          <button
            type="button"
            onClick={() => onSameRace(duplicate)}
            disabled={busy}
            aria-busy={busy}
            className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
          >
            {pendingLabel === "Joining…" ? pendingLabel : "Same race → join"}
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

const DETAIL_LABEL =
  "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft";

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
          <dt className={DETAIL_LABEL}>{label}</dt>
          <dd className="mt-1 font-sans text-sm text-ink tnum break-words whitespace-pre-line">
            <span className="text-slate">{show(field, live[field])}</span>
            <span className="text-slate-soft mx-2">→</span>
            {show(field, change.fields[field])}
          </dd>
        </div>
      ))}
      {change.coverS3Key && (
        <div>
          <dt className={DETAIL_LABEL}>Cover</dt>
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
      · {formatLongDate(race.date, true)} · {race.location} · {race.strength === "strong" ? "strong match" : "possible match"}
    </p>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className={DETAIL_LABEL}>{label}</dt>
      <dd className="mt-1 font-sans text-sm text-ink tnum break-words">{value}</dd>
    </div>
  );
}
