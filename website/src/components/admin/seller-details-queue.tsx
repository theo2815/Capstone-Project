"use client";

import Link from "next/link";
import { KICKER_SIZE_CLASS } from "@/components/ui/kicker";
import { useEffect, useState } from "react";
import { AdminDetailDrawer } from "@/components/admin/admin-detail-drawer";
import { AdminConfirmModal } from "@/components/admin/admin-form-modal";
import { AdminTextarea } from "@/components/admin/admin-form-fields";
import { AdminStatusPill, type AdminStatusPillTone } from "@/components/admin/admin-status-pill";
import { AdminStatusTabs } from "@/components/admin/admin-status-tabs";
import { Slab } from "@/components/profile-shell/slab";
import { BTN_DANGER, BTN_GHOST, BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Modal } from "@/components/ui/modal";
import { Skeleton } from "@/components/ui/skeleton";
import { LoadError } from "@/components/ui/load-error";
import { useAdminKpis } from "@/hooks/use-admin-data";
import {
  useAdminSellerDetailsActions,
  useAdminSellerDetailsList,
  useAdminSellerDetailsRecord,
} from "@/hooks/use-admin-seller-details";
import { EMPTY_ADMIN_USERS, useAdminPhotographerSettings, useAdminUsers } from "@/hooks/use-admin-users";
import { useToast } from "@/hooks/use-toast";
import { useUrlState } from "@/hooks/use-url-state";
import { ApiError } from "@/lib/api";
import { SELLER_ID_TYPES, type AdminSellerDetails, type AdminSellerFilter, type RepeatMatch } from "@/lib/api-seller-details";
import { captureAuthSession } from "@/lib/auth-session";
import { ROUTES } from "@/lib/constants";
import { formatLongDate, sentenceCaseDate } from "@/lib/format";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { cn } from "@/lib/utils";

const KICKER = `font-mono uppercase ${KICKER_SIZE_CLASS.sm}`;
// Keyed by reviewState; `status` keeps the old values for old builds.
const STATUS_LABEL: Record<string, string> = { manual_review: "Needs review", auto_approved: "Auto-approved", approved: "Approved", rejected: "Rejected" };
const REASON_LABEL: Record<string, string> = {
  PASSED: "Passed",
  LIVENESS_LOW: "Live check score below the pass mark",
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
const STATUS_TONE: Record<string, AdminStatusPillTone> = { manual_review: "amber", rejected: "error" };
const FILTERS = [["manual_review", "Needs review"], ["auto_approved", "Auto-approved"], ["all", "All"]] as const;
type FilterKey = (typeof FILTERS)[number][0];
const FILTER_COUNT = { manual_review: "waiting", auto_approved: "auto-approved", all: "in all" } as const;
const FILTER_EMPTY = { manual_review: "No seller details waiting. You're caught up.", auto_approved: "No auto-approved seller details.", all: "No seller details yet." } as const;
const LINK_LIFETIME_MS = 5 * 60_000;
const REASON_ID = "seller-reject-reason";
const idLabel = (value: string) => SELLER_ID_TYPES.find((t) => t.value === value)?.label ?? value;
const statusLabel = (reviewState: string) => STATUS_LABEL[reviewState] ?? reviewState;
const StatusPill = ({ reviewState, className }: { reviewState: string; className?: string }) => (
  <AdminStatusPill label={statusLabel(reviewState)} tone={STATUS_TONE[reviewState] ?? "slate"} className={className} />
);
// The photographer detail page shows the same seller-details status (AX-27).
export const SellerStatusPill = StatusPill;
// The pass mark is the server's (FT-29); an older backend sends none, so the bare score shows.
const score = (value: number | null, passMark?: number | null) =>
  value === null ? "Not run" : passMark == null ? `${value.toFixed(1)}%` : `${value.toFixed(1)}% (needs ${passMark}%)`;
// C-10: the row names why a human is needed; these three never are the reason.
const NOT_A_ROW_REASON = new Set(["PASSED", "AI_DISABLED", "ID_FORMAT_UNCHECKED"]);
const rowReason = (codes: string[]) => {
  const code = codes.find((c) => REASON_LABEL[c] && !NOT_A_ROW_REASON.has(c));
  return code ? REASON_LABEL[code] : null;
};

// SC-2b: admins review photographers' seller details (ID photos, a selfie
// holding the ID) on the verifications page, separate from verification itself.
// AA-I16: manual-review cases first (the server sorts), with the automatic results.
export function SellerDetailsQueue({ number = "04" }: { number?: string } = {}) {
  const [rawFilter, setFilter] = useUrlState<string>("sd", "manual_review");
  const key: FilterKey = FILTERS.find(([k]) => k === rawFilter)?.[0] ?? "manual_review";
  const status: AdminSellerFilter = key === "all" ? null : key;
  // Paging belongs to one filter, so a tab click or Back/Forward starts on the first page.
  const [paging, setPaging] = useState<{ key: FilterKey; limit: number }>({ key, limit: PAGE_SIZE.ADMIN_INCREMENT });
  const limit = paging.key === key ? paging.limit : PAGE_SIZE.ADMIN_INCREMENT;
  // C-11: the open row lives in the URL, so a reload reopens it and a link can point at it.
  const [openId, setOpenId] = useUrlState<string>("sd-row", "");
  const list = useAdminSellerDetailsList(status, limit);
  // Absent until the backend sends it (AX-8): the tab then shows no count rather than 0.
  const waiting = useAdminKpis()?.pendingSellerDetails;
  const rows = list.data?.items ?? [];
  const total = list.data?.total ?? 0;

  return (
    <Slab id="seller-details" number={number} title="Seller details" caption="ID checks before uploads"
      trailing={<span className="tnum">{total} {FILTER_COUNT[key]}</span>}>
      <AdminStatusTabs label="Filter seller details" value={key} onChange={setFilter}
        tabs={FILTERS.map(([k, label]) => ({ key: k, label, count: k === "manual_review" ? waiting : undefined }))} />
      <div className="pt-6">
      {list.isPending ? (
        <div aria-busy="true" className="space-y-3">
          {Array.from({ length: 3 }, (_, i) => <Skeleton key={i} className="h-14 w-full rounded-xl" />)}
        </div>
      ) : list.isError ? (
        <LoadError what="seller details" onRetry={() => void list.refetch()} className="text-sm text-error" />
      ) : rows.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">{FILTER_EMPTY[key]}</p>
      ) : (
        <>
          <ul className="space-y-3">
            {rows.map((row) => {
              const reason = row.reviewState === "manual_review" ? rowReason(row.reasonCodes ?? []) : null;
              return (
              <li key={row.userId}>
                <button type="button" onClick={() => setOpenId(row.userId)}
                  className="w-full flex items-center justify-between gap-4 border-b border-line pb-3 text-left hover:bg-bone-deep/50">
                  <span className="min-w-0">
                    <span className="block font-display text-base text-ink truncate">{row.legalName}</span>
                    <span className="block font-sans text-sm text-slate-soft mt-1">
                      {idLabel(row.idType)} · <span className="tnum">{sentenceCaseDate(formatLongDate(row.submittedAt, true))}</span>
                    </span>
                    {reason && <span className="block font-sans text-sm text-ink mt-1">{reason}</span>}
                  </span>
                  <StatusPill reviewState={row.reviewState} className="shrink-0" />
                </button>
              </li>
              );
            })}
          </ul>
          <LoadMoreButton shown={rows.length} total={total} increment={PAGE_SIZE.ADMIN_INCREMENT}
            isLoading={list.isFetching} onLoadMore={() => setPaging({ key, limit: limit + PAGE_SIZE.ADMIN_INCREMENT })} />
        </>
      )}
      </div>
      {openId && <SellerDetailsDrawer userId={openId} onClose={() => setOpenId("")} />}
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
  const [zoom, setZoom] = useState<{ label: string; src: string } | null>(null);
  const [confirmReject, setConfirmReject] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const data = record.data;
  const expiresAt = record.dataUpdatedAt + LINK_LIFETIME_MS;

  useEffect(() => {
    if (!record.dataUpdatedAt) return;
    const timer = setTimeout(() => setNow(Date.now()), Math.max(0, expiresAt - Date.now()));
    return () => clearTimeout(timer);
  }, [record.dataUpdatedAt, expiresAt]);

  const decide = async (kind: "approve" | "reject", confirmed = false) => {
    if (!data) return;
    setActionError(null);
    const trimmed = reason.trim();
    if (kind === "reject" && !trimmed) {
      setReasonError("Write the reason the photographer will see.");
      // Reject sits in the sticky footer, far below the box on a long record.
      const box = document.getElementById(REASON_ID);
      box?.scrollIntoView({ block: "center" });
      box?.focus();
      return;
    }
    // C-12: rejecting a seller who is already approved pauses a working photographer, so it asks first.
    if (kind === "reject" && !confirmed && (data.reviewState === "approved" || data.reviewState === "auto_approved")) {
      setConfirmReject(true);
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
    } finally {
      // A failure shows in the drawer, with the typed reason kept.
      setConfirmReject(false);
    }
  };

  const canApprove = data?.reviewState === "manual_review";
  const canReject = canApprove || data?.reviewState === "auto_approved" || data?.reviewState === "approved";
  return (
    <AdminDetailDrawer open onClose={() => { if (!isPending) onClose(); }} escDisabled={isPending || !!zoom || confirmReject}
      kicker="Seller details" title={data?.legalName ?? "Seller details"}
      rightHeader={data && <StatusPill reviewState={data.reviewState} />}
      subtitle={data && <p className="font-mono text-[13px] text-slate break-all">{data.email}</p>}
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
        <div aria-busy="true" className="space-y-3">{Array.from({ length: 2 }, (_, i) => <Skeleton key={i} className="h-14 w-full rounded-xl" />)}</div>
      ) : record.isError || !data ? (
        <LoadError what="seller details" onRetry={() => void record.refetch()} className="text-sm text-error" />
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
          <Photos data={data} expired={now >= expiresAt} onReload={() => void record.refetch()} reloading={record.isFetching}
            onOpen={(label, src) => setZoom({ label, src })} />
          <Checklist />
          {canReject && (
            <div className="flex flex-col gap-2">
              <p className="font-sans text-sm text-ink">Approve: the ID check is passed. If their profile is complete it goes live and they are told. Reject: they are told your reason and their new uploads pause until they send details again. A rejection cannot be undone here.</p>
              <AdminTextarea id={REASON_ID} label="Reason for rejecting" required maxLength={500} value={reason} error={reasonError}
                onChange={(next) => { setReason(next); setReasonError(null); }} />
              <p className="font-sans text-sm text-slate-soft">Sent to the photographer. Their new uploads pause until they send new details.</p>
            </div>
          )}
        </div>
      )}
      {/* Drawer is z-50, Modal z-60: the photo opens on top; the drawer's Esc waits while it is open. */}
      <Modal isOpen={!!zoom} onClose={() => setZoom(null)} title={zoom?.label} className="w-auto max-w-[calc(100vw-3rem)] overflow-auto">
        {/* eslint-disable-next-line @next/next/no-img-element -- signed storage URL outside Next image-domain config. */}
        {zoom && <img src={zoom.src} alt={zoom.label} draggable={false} className="max-w-none" />}
      </Modal>
      <AdminConfirmModal open={confirmReject} submitting={isPending} onClose={() => setConfirmReject(false)}
        onConfirm={() => void decide("reject", true)} title="Reject an approved ID check?" confirmLabel="Reject" tone="danger">
        <p className="font-sans text-sm text-ink-soft">This photographer is already approved. Rejecting pauses their new uploads until they send details again, and they are told your reason. This cannot be undone here.</p>
      </AdminConfirmModal>
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
    ["Submitted", sentenceCaseDate(formatLongDate(data.submittedAt))],
    ["Live check score", score(data.livenessConfidence, data.livenessPassMark)],
    ["Face match score", score(data.faceSimilarity, data.faceMatchPassMark)],
    ["Reasons", data.reasonCodes.length > 0
      ? data.reasonCodes.map((code) => REASON_LABEL[code] ?? code).join("; ")
      : data.autoCheckedAt ? "None" : "Sent before automatic checks"],
    ["Live-selfie consent", data.biometricConsentAt ? sentenceCaseDate(formatLongDate(data.biometricConsentAt)) : "Not given"],
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

function Photos({ data, expired, onReload, reloading, onOpen }: {
  data: AdminSellerDetails; expired: boolean; onReload: () => void; reloading: boolean; onOpen: (label: string, src: string) => void;
}) {
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
            <figcaption className="mt-2 flex flex-wrap items-center justify-between gap-2">
              <span className={cn(KICKER, "text-slate-soft")}>{label}</span>
              <button type="button" disabled={expired} onClick={() => onOpen(label, src)} className={cn(BTN_GHOST, BTN_SIZE.sm)}>
                Open full size<span className="sr-only">: {label}</span>
              </button>
            </figcaption>
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
        <li>PhilSys National ID: scan the QR code in the back photo with PSA&apos;s PhilSys Check (<a href="https://verify.philsys.gov.ph" target="_blank" rel="noopener noreferrer" className="underline underline-offset-4">verify.philsys.gov.ph</a>). It should say &quot;Verified&quot; and show the same photo.</li>
        <li>PRC ID: look the licence up at <a href="https://verification.prc.gov.ph" target="_blank" rel="noopener noreferrer" className="underline underline-offset-4">verification.prc.gov.ph</a>.</li>
      </ul>
    </section>
  );
}
