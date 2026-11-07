"use client";

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
import { useAdminPhotographerSettings } from "@/hooks/use-admin-users";
import { useToast } from "@/hooks/use-toast";
import { ApiError } from "@/lib/api";
import { SELLER_ID_TYPES, type AdminSellerDetails } from "@/lib/api-seller-details";
import { captureAuthSession } from "@/lib/auth-session";
import { formatLongDate } from "@/lib/format";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { cn } from "@/lib/utils";

const KICKER = "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px]";
const STATUS_LABEL = { submitted: "Waiting", approved: "Approved", rejected: "Rejected" } as const;
const LINK_LIFETIME_MS = 5 * 60_000;
const idLabel = (value: string) => SELLER_ID_TYPES.find((t) => t.value === value)?.label ?? value;

// SC-2b: admins review photographers' seller details (ID photos, a selfie
// holding the ID) on the verifications page, separate from verification itself.
export function SellerDetailsQueue() {
  const [status, setStatus] = useState<"submitted" | null>("submitted");
  const [limit, setLimit] = useState<number>(PAGE_SIZE.ADMIN_INCREMENT);
  const [openId, setOpenId] = useState<string | null>(null);
  const list = useAdminSellerDetailsList(status, limit);
  const rows = list.data?.items ?? [];
  const total = list.data?.total ?? 0;
  const filter = (next: "submitted" | null) => { setStatus(next); setLimit(PAGE_SIZE.ADMIN_INCREMENT); };

  return (
    <Slab id="seller-details" number="04" title="Seller details" caption="ID checks before uploads"
      trailing={<span className="tnum">{total} {status ? "waiting" : "in all"}</span>}>
      <div role="group" aria-label="Filter seller details" className="mb-6 flex gap-2">
        {([["submitted", "Waiting"], [null, "All"]] as const).map(([value, label]) => (
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
        <p className="font-sans text-sm text-slate-soft">{status ? "No seller details waiting." : "No seller details yet."}</p>
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
                  <span className={cn(KICKER, "shrink-0 text-slate")}>{STATUS_LABEL[row.status]}</span>
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

  const canReject = data?.status === "submitted" || data?.status === "approved";
  return (
    <AdminDetailDrawer open onClose={() => { if (!isPending) onClose(); }} escDisabled={isPending}
      kicker="Seller details" title={data?.legalName ?? "Seller details"}
      rightHeader={data && <span className={cn(KICKER, "inline-flex rounded-full border border-line text-slate px-3 py-0.5")}>{STATUS_LABEL[data.status]}</span>}
      subtitle={data && <p className={cn(KICKER, "text-slate")}>{data.email}</p>}
      actions={data && (
        <>
          {canReject && (
            <button type="button" disabled={isPending} onClick={() => void decide("reject")} className={cn(BTN_DANGER, BTN_SIZE.sm)}>Reject</button>
          )}
          {data.status === "submitted" && (
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
  ];
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

function Photos({ data, expired, onReload, reloading }: { data: AdminSellerDetails; expired: boolean; onReload: () => void; reloading: boolean }) {
  const files = data.files;
  const shots = files ? ([
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
        <li>The typed legal name, the name on the ID and the payout account names match.</li>
        <li>The ID is not expired, and the photos show a real card (not a screen) with no edited-looking text or pasted-on photo.</li>
        <li>The face in the selfie matches the ID photo, and the ID in the selfie is the same card.</li>
        <li>PhilSys National ID: scan the QR code in the back photo with PSA&apos;s PhilSys Check (verify.philsys.gov.ph). It should say &quot;Verified&quot; and show the same photo.</li>
        <li>PRC ID: look the licence up at verification.prc.gov.ph.</li>
      </ul>
    </section>
  );
}
