"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { AdminDetailDrawer } from "@/components/admin/admin-detail-drawer";
import { Slab } from "@/components/profile-shell/slab";
import { BTN_DANGER, BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { FieldError } from "@/components/ui/field-error";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import {
  useAdminSellerDetailsActions,
  useAdminSellerDetailsList,
  useAdminSellerDetailsRecord,
} from "@/hooks/use-admin-seller-details";
import { EMPTY_ADMIN_USERS, useAdminPhotographerSettings, useAdminUsers } from "@/hooks/use-admin-users";
import { useToast } from "@/hooks/use-toast";
import { ApiError } from "@/lib/api";
import { SELLER_ID_TYPES, type AdminSellerDetails, type AdminSellerFilter, type RepeatMatch } from "@/lib/api-seller-details";
import { captureAuthSession } from "@/lib/auth-session";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { cn } from "@/lib/utils";

const KICKER = "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px]";
// Keyed by reviewState; `status` keeps the old values for old builds.
const STATUS_LABEL: Record<string, string> = { manual_review: "Needs review", auto_approved: "Auto-approved", approved: "Approved", rejected: "Rejected" };
const REASON_LABEL: Record<string, string> = {
  PASSED: "Passed",
  LIVENESS_LOW: "Live check score below the bar",
  FACE_MISMATCH: "Live face does not match the ID photo",
  NO_FACE_ON_ID: "No face found on the ID photo",
  AI_DISABLED: "Automatic checks were off",
  AI_ERROR: "Automatic check failed",
  DECLINED: "Photographer asked for an admin check",
  ID_NUMBER_MISMATCH: "Typed ID number differs from the ID photo",
  ID_NUMBER_UNREADABLE: "Could not read the ID number off the photo",
  PAYOUT_NAME_MISMATCH: "Payout name differs from the legal name",
  REPEAT_ACCOUNT_MATCH: "Same ID number, payout number or phone as another account",
  PREVIOUSLY_REJECTED: "Sent again after an admin rejected it",
  ID_FORMAT_UNCHECKED: "ID number format not checked for this ID type",
  ID_NAME_MISMATCH: "Legal name not found on the ID",
  REPEATED_FAILED_CHECKS: "Several failed automatic checks before this one",
};
const MATCH_LABEL: Record<string, string> = { ID_NUMBER: "ID number", PAYOUT_NUMBER: "payout number", PHONE: "phone" };
const FILTERS = [["manual_review", "Needs review"], ["auto_approved", "Auto-approved"], [null, "All"]] as const;
const FILTER_COUNT = { manual_review: "waiting", auto_approved: "auto-approved", all: "in all" } as const;
const FILTER_EMPTY = { manual_review: "No seller details waiting.", auto_approved: "No auto-approved seller details.", all: "No seller details yet." } as const;
const LINK_LIFETIME_MS = 5 * 60_000;
const idLabel = (value: string) => SELLER_ID_TYPES.find((t) => t.value === value)?.label ?? value;
const statusLabel = (reviewState: string) => STATUS_LABEL[reviewState] ?? reviewState;
const score = (value: number | null) => (value === null ? "Not run" : `${value.toFixed(1)}%`);

// SC-2b: admins review photographers' seller details (ID photos, a selfie
// holding the ID) on the verifications page, separate from verification itself.
// AA-I16: manual-review cases first (the server sorts), with the automatic results.
export function SellerDetailsQueue() {
  const [status, setStatus] = useState<AdminSellerFilter>("manual_review");
  const [limit, setLimit] = useState<number>(PAGE_SIZE.ADMIN_INCREMENT);
  const [openId, setOpenId] = useState<string | null>(null);
  const list = useAdminSellerDetailsList(status, limit);
  const rows = list.data?.items ?? [];
  const total = list.data?.total ?? 0;
  const filter = (next: AdminSellerFilter) => { setStatus(next); setLimit(PAGE_SIZE.ADMIN_INCREMENT); };

  return (
    <Slab id="seller-details" number="04" title="Seller details" caption="ID checks before uploads"
      trailing={<span className="tnum">{total} {FILTER_COUNT[status ?? "all"]}</span>}>
      <div role="group" aria-label="Filter seller details" className="mb-6 flex flex-wrap gap-2">
        {FILTERS.map(([value, label]) => (
          <button key={label} type="button" aria-pressed={status === value} onClick={() => filter(value)}
            className={cn(BTN_SECONDARY, BTN_SIZE.sm, status === value && "bg-ink text-surface")}>
            {label}
          </button>
        ))}
      </div>
      {list.isPending ? (
        <p className="font-sans text-sm text-slate-soft">Loading…</p>
      ) : list.isError ? (
        <div role="alert" className="font-sans text-sm text-ink">
          <p>Couldn&apos;t load seller details.</p>
          <button type="button" onClick={() => void list.refetch()} className={cn(BTN_SECONDARY, BTN_SIZE.sm, "mt-3")}>Try again</button>
        </div>
      ) : rows.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">{FILTER_EMPTY[status ?? "all"]}</p>
      ) : (
        <>
          <ul className="space-y-3">
            {rows.map((row) => (
              <li key={row.userId}>
                <button type="button" onClick={() => setOpenId(row.userId)}
                  className="w-full flex items-center justify-between gap-4 border-b border-line pb-3 text-left hover:bg-bone-deep/50">
                  <span className="min-w-0">
                    <span className="block font-display text-base text-ink truncate">{row.legalName}</span>
                    <span className={cn(KICKER, "block text-slate-soft mt-1")}>
                      {idLabel(row.idType)} · <span className="tnum">{formatLongDate(row.submittedAt, true)}</span>
                    </span>
                  </span>
                  <span className={cn(KICKER, "shrink-0 text-slate")}>{statusLabel(row.reviewState)}</span>
                </button>
              </li>
            ))}
          </ul>
          <LoadMoreButton shown={rows.length} total={total} increment={PAGE_SIZE.ADMIN_INCREMENT}
            isLoading={list.isFetching} onLoadMore={() => setLimit((n) => n + PAGE_SIZE.ADMIN_INCREMENT)} />
        </>
      )}
      {openId && <SellerDetailsDrawer userId={openId} onClose={() => setOpenId(null)} />}
    </Slab>
  );
}

function SellerDetailsDrawer({ userId, onClose }: { userId: string; onClose: () => void }) {
  const record = useAdminSellerDetailsRecord(userId);
  const settings = useAdminPhotographerSettings(userId);
  const { approve, reject, isPending } = useAdminSellerDetailsActions();
  const { showToast } = useToast();
  const [reason, setReason] = useState("");
  const [reasonError, setReasonError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const data = record.data;
  const expiresAt = record.dataUpdatedAt + LINK_LIFETIME_MS;

  useEffect(() => {
    if (!record.dataUpdatedAt) return;
    const timer = setTimeout(() => setNow(Date.now()), Math.max(0, expiresAt - Date.now()));
    return () => clearTimeout(timer);
  }, [record.dataUpdatedAt, expiresAt]);

  const decide = async (kind: "approve" | "reject") => {
    if (!data) return;
    setActionError(null);
    const trimmed = reason.trim();
    if (kind === "reject" && (!trimmed || trimmed.length > 500)) {
      setReasonError(trimmed ? "Keep the reason under 500 characters." : "Write the reason the photographer will see.");
      return;
    }
    const isCurrent = captureAuthSession();
    try {
      if (kind === "approve") await approve.mutateAsync({ userId, revision: data.revision });
      else await reject.mutateAsync({ userId, revision: data.revision, reason: trimmed });
      if (!isCurrent()) return;
      showToast({ kind: "success", message: kind === "approve" ? "Seller details approved." : "Seller details rejected. The photographer was told why." });
      onClose();
    } catch (error) {
      if (!isCurrent()) return;
      setActionError(error instanceof ApiError && error.status === 409
        ? "These details changed. Reload before deciding."
        : error instanceof ApiError ? error.message : "Couldn't save the decision. Try again.");
    }
  };

  const canApprove = data?.reviewState === "manual_review";
  const canReject = canApprove || data?.reviewState === "auto_approved" || data?.reviewState === "approved";
  return (
    <AdminDetailDrawer open onClose={() => { if (!isPending) onClose(); }} escDisabled={isPending}
      kicker="Seller details" title={data?.legalName ?? "Seller details"}
      rightHeader={data && <span className={cn(KICKER, "inline-flex rounded-full border border-line text-slate px-3 py-0.5")}>{statusLabel(data.reviewState)}</span>}
      subtitle={data && <p className={cn(KICKER, "text-slate")}>{data.email}</p>}
      actions={data && (
        <>
          {canReject && (
            <button type="button" disabled={isPending} onClick={() => void decide("reject")} className={cn(BTN_DANGER, BTN_SIZE.sm)}>Reject</button>
          )}
          {canApprove && (
            <button type="button" disabled={isPending} onClick={() => void decide("approve")} className={cn(BTN_PRIMARY, BTN_SIZE.sm)}>Approve</button>
          )}
        </>
      )}>
      {record.isPending ? (
        <p className="font-sans text-sm text-slate-soft">Loading…</p>
      ) : record.isError || !data ? (
        <div role="alert" className="font-sans text-sm text-ink">
          <p>Couldn&apos;t open these seller details.</p>
          <button type="button" onClick={() => void record.refetch()} className={cn(BTN_SECONDARY, BTN_SIZE.sm, "mt-3")}>Try again</button>
        </div>
      ) : (
        <div className="space-y-10">
          {actionError && (
            <div role="alert" className="rounded-xl border border-error/40 bg-error/5 px-5 py-4 font-sans text-sm text-ink">
              <p>{actionError}</p>
              <button type="button" onClick={() => { setActionError(null); void record.refetch(); }} className="mt-2 underline underline-offset-4">Reload</button>
            </div>
          )}
          <Details data={data} payoutNames={settings.isError ? "Couldn't load payout accounts" : settings.data?.payouts.map((p) => p.accountName) ?? null} />
          {data.repeatMatches.length > 0 && <RepeatMatches matches={data.repeatMatches} />}
          <Photos data={data} expired={now >= expiresAt} onReload={() => void record.refetch()} reloading={record.isFetching} />
          <Checklist />
          {canReject && (
            <div className="flex flex-col gap-2">
              <label htmlFor="seller-reject-reason" className="font-sans text-sm text-slate">Reason for rejecting</label>
              <textarea id="seller-reject-reason" rows={3} maxLength={500} value={reason} disabled={isPending}
                aria-invalid={reasonError ? true : undefined} aria-describedby={reasonError ? "seller-reject-reason-error" : undefined}
                onChange={(e) => { setReason(e.target.value); setReasonError(null); }}
                className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink resize-none" />
              <p className="font-sans text-sm text-slate-soft">Sent to the photographer. Their new uploads pause until they send new details.</p>
              <FieldError message={reasonError} id="seller-reject-reason-error" density="tight" />
            </div>
          )}
        </div>
      )}
    </AdminDetailDrawer>
  );
}

function Details({ data, payoutNames }: { data: AdminSellerDetails; payoutNames: string[] | string | null }) {
  const rows: [string, string][] = [
    ["Legal name", data.legalName],
    ["ID type", idLabel(data.idType)],
    ["Address", data.address],
    ["Phone", data.phone],
    ["Payout account names", payoutNames === null ? "Loading…" : typeof payoutNames === "string" ? payoutNames : payoutNames.join(", ") || "No payout accounts"],
    ["Submitted", formatLongDate(data.submittedAt)],
    ["Live check score", score(data.livenessConfidence)],
    ["Face match score", score(data.faceSimilarity)],
    ["Reasons", data.reasonCodes.length > 0
      ? data.reasonCodes.map((code) => REASON_LABEL[code] ?? code).join("; ")
      : data.autoCheckedAt ? "None" : "Sent before automatic checks"],
    ["Live-selfie consent", data.biometricConsentAt ? formatLongDate(data.biometricConsentAt) : "Not given"],
    ["ID number on file", data.hasIdNumber ? "Yes" : "No"],
  ];
  if (data.failedAutoChecks) rows.push(["Failed automatic tries", String(data.failedAutoChecks)]);
  if (data.rejectionReason) rows.push(["Last rejection", data.rejectionReason]);
  return (
    <dl className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-6 gap-y-2 font-sans text-sm">
      {rows.map(([label, value]) => (
        <div key={label} className="contents">
          <dt className="text-slate">{label}</dt>
          <dd className="text-ink break-words tnum">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

// The admin page for one photographer is keyed by handle, which a match does not carry:
// it comes from the admin users list (already cached on this page). No handle = plain email.
function RepeatMatches({ matches }: { matches: RepeatMatch[] }) {
  const users = useAdminUsers().data ?? EMPTY_ADMIN_USERS;
  return (
    <section aria-label="Other accounts" className="font-sans text-sm text-ink">
      <p className="font-semibold">Other accounts with the same details</p>
      <ul className="mt-3 space-y-2">
        {matches.map((match) => {
          const handle = users.find((u) => u.userId === match.userId)?.handle;
          return (
            <li key={match.userId} className="flex flex-wrap items-baseline gap-x-2 gap-y-1 break-words">
              {handle
                ? <Link href={`${ROUTES.ADMIN_PHOTOGRAPHERS}/${encodeURIComponent(handle)}`} className="underline underline-offset-4">{match.email}</Link>
                : <span>{match.email}</span>}
              <span className="text-slate">· {match.matchedOn.map((kind) => MATCH_LABEL[kind] ?? kind).join(", ")}</span>
              {match.suspended && <span className={cn(KICKER, "text-error")}>Suspended</span>}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function Photos({ data, expired, onReload, reloading }: { data: AdminSellerDetails; expired: boolean; onReload: () => void; reloading: boolean }) {
  const files = data.files;
  const shots = files ? ([
    ["Live selfie (reference)", files.livenessReference],
    ["Front of the ID", files.idFront],
    ["Back of the ID", files.idBack],
    ["Photo holding the ID", files.selfie],
    ["BIR Form 2303", files.birForm],
  ] as [string, string | null][]).filter((shot): shot is [string, string] => !!shot[1]) : [];
  return (
    <section aria-label="Photos">
      {expired && (
        <div className="mb-4 flex flex-wrap items-center gap-3 font-sans text-sm text-slate">
          <p>These photo links have expired.</p>
          <button type="button" disabled={reloading} onClick={onReload} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>Reload photos</button>
        </div>
      )}
      <div className="grid gap-6">
        {shots.map(([label, src]) => (
          <figure key={label}>
            {/* eslint-disable-next-line @next/next/no-img-element -- signed storage URL outside Next image-domain config. */}
            <img src={src} alt={label} draggable={false} className="w-full rounded-xl border border-line bg-bone-deep object-contain" />
            <figcaption className={cn(KICKER, "mt-2 text-slate-soft")}>{label}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}

function Checklist() {
  return (
    <section aria-label="Checklist" className="font-sans text-sm text-ink">
      <p className="font-semibold">Before you approve</p>
      <ul className="mt-3 list-disc space-y-2 pl-5">
        <li>The ID is not expired, and the photos show a real card (not a screen) with no edited-looking text or pasted-on photo.</li>
        <li>The live selfie and the photo holding the ID show the person on the ID photo, and the held card is the same card.</li>
        <li>PhilSys National ID: scan the QR code in the back photo with PSA&apos;s PhilSys Check (verify.philsys.gov.ph). It should say &quot;Verified&quot; and show the same photo.</li>
        <li>PRC ID: look the licence up at verification.prc.gov.ph.</li>
      </ul>
    </section>
  );
}
