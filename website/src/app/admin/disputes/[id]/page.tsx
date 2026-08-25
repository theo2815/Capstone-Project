"use client";

import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { Slab } from "@/components/profile-shell";
import { Kicker } from "@/components/ui/kicker";
import { useAdminDisputeView } from "@/lib/admin-dispute-view";
import { AdminDisputeActionAside } from "@/components/admin/admin-dispute-action-aside";
import { AdminDisputeActivityList } from "@/components/admin/admin-dispute-activity-list";
import { AdminStatusPill, type AdminStatusPillTone } from "@/components/admin/admin-status-pill";
import {
  DISPUTE_REASON_LABEL,
  DISPUTE_RESOLUTION_LABEL,
  type DisputeStatus,
} from "@/lib/admin-disputes";
import { getEventById } from "@/lib/event-catalog";
import { syntheticCoverGradient } from "@/lib/admin-photographer-view";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import { formatPrice } from "@/lib/utils";

const STATUS_LABEL: Record<DisputeStatus, string> = {
  open: "Open",
  resolved: "Resolved",
  denied: "Denied",
  escalated: "Escalated",
};

function statusTone(status: DisputeStatus): AdminStatusPillTone {
  switch (status) {
    case "open":
      return "amber";
    case "resolved":
      return "fresh";
    case "denied":
      return "ink";
    case "escalated":
      return "slate";
  }
}

const PAYMENT_LABEL: Record<string, string> = {
  gcash: "GCash",
  maya: "Maya",
  paymaya: "PayMaya",
  card: "Card",
  grabpay: "GrabPay",
};

// Phase 2b — dispute detail page. Mirrors the photographer-detail
// layout: hero + 5 content slabs on the left, sticky action aside on
// lg right column. Activity slab pulls from useAdminDisputeStore.log
// via <AdminDisputeActivityList>.
export default function AdminDisputeDetailPage() {
  const params = useParams<{ id: string }>();
  const disputeId = params.id;
  const view = useAdminDisputeView(disputeId);

  if (!view) notFound();

  const { dispute, photographerName, decisions } = view;
  const event = getEventById(dispute.eventId);
  const eventName = event?.name ?? `Event ${dispute.eventId}`;
  const cover = syntheticCoverGradient(dispute.id);
  const paymentLabel =
    PAYMENT_LABEL[dispute.orderSnapshot.paymentMethod] ??
    dispute.orderSnapshot.paymentMethod;

  return (
    <>
      <BackLink />

      <div className="grid lg:grid-cols-[1fr_18rem] lg:gap-12 lg:items-start">
        <div className="min-w-0">
          <Hero
            disputeId={dispute.id}
            runnerHandle={dispute.runnerHandle}
            photographerHandle={dispute.photographerHandle}
            status={dispute.status}
            reportedAt={dispute.reportedAt}
            eventName={eventName}
            amountDisputed={dispute.orderSnapshot.total}
            cover={cover}
          />

          <Slab id="claim" number="01" title="Claim" caption="Runner's case">
            <div className="space-y-3">
              <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
                Reason: {DISPUTE_REASON_LABEL[dispute.reason]}
              </p>
              <p className="font-sans text-sm md:text-base text-ink-soft whitespace-pre-line">
                {dispute.note}
              </p>
            </div>
          </Slab>

          <Slab id="order" number="02" title="Order" caption="Payment record">
            <dl className="grid grid-cols-1 sm:grid-cols-3 gap-y-4 gap-x-8">
              <FieldRow label="Order id" value={dispute.orderId} mono />
              <FieldRow
                label="Total"
                value={formatPrice(dispute.orderSnapshot.total)}
                mono
              />
              <FieldRow label="Method" value={paymentLabel} />
              <FieldRow
                label="Paid at"
                value={formatLongDate(dispute.orderSnapshot.paidAt)}
                mono
              />
            </dl>
          </Slab>

          <Slab id="photo" number="03" title="Photo" caption="Disputed frame">
            <div className="grid grid-cols-1 sm:grid-cols-[12rem_1fr] gap-6 items-start">
              <div
                className="aspect-[4/3] rounded-2xl border border-line"
                style={{
                  background: `linear-gradient(135deg, ${cover.from}, ${cover.to})`,
                }}
                aria-label={dispute.photoSnapshot.alt}
              />
              <div className="space-y-3">
                <p className="font-sans text-sm text-ink-soft">
                  {dispute.photoSnapshot.alt}
                </p>
                <dl className="grid grid-cols-2 gap-y-3 gap-x-8">
                  <FieldRow
                    label="Bib"
                    value={dispute.photoSnapshot.bib ?? "—"}
                    mono
                  />
                  <FieldRow
                    label="Km mark"
                    value={
                      dispute.photoSnapshot.kmMark === null
                        ? "—"
                        : `${dispute.photoSnapshot.kmMark} km`
                    }
                    mono
                  />
                  <FieldRow label="Photo id" value={dispute.photoId} mono />
                  <FieldRow label="Event" value={eventName} />
                </dl>
              </div>
            </div>
          </Slab>

          <Slab id="parties" number="04" title="Parties" caption="Both sides">
            <div className="space-y-4">
              <div className="rounded-2xl border border-line bg-bone p-5">
                <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
                  Runner
                </p>
                <p className="font-display text-lg text-ink mt-2">
                  @{dispute.runnerHandle}
                </p>
              </div>
              <div className="rounded-2xl border border-line bg-bone p-5">
                <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
                  Photographer
                </p>
                <p className="font-display text-lg text-ink mt-2">
                  {photographerName}
                </p>
                <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate mt-1">
                  @{dispute.photographerHandle}
                </p>
              </div>
            </div>
          </Slab>

          <Slab id="activity" number="05" title="Activity" caption="Decision log">
            {dispute.status !== "open" && decisions.length === 0 && (
              <ResolutionNote
                resolution={dispute.resolution}
                refundAmount={dispute.refundAmount}
                resolvedAt={dispute.resolvedAt}
              />
            )}
            <AdminDisputeActivityList
              disputeId={dispute.id}
              activity={dispute.activity}
            />
          </Slab>
        </div>

        <AdminDisputeActionAside
          dispute={dispute}
          runnerHandle={dispute.runnerHandle}
        />
      </div>
    </>
  );
}

function BackLink() {
  return (
    <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] mb-6">
      <Link
        href={ROUTES.ADMIN_DISPUTES}
        className="text-slate hover:text-ink transition-colors"
      >
        ← All disputes
      </Link>
    </p>
  );
}

function Hero({
  disputeId,
  runnerHandle,
  photographerHandle,
  status,
  reportedAt,
  eventName,
  amountDisputed,
  cover,
}: {
  disputeId: string;
  runnerHandle: string;
  photographerHandle: string;
  status: DisputeStatus;
  reportedAt: string;
  eventName: string;
  amountDisputed: number;
  cover: { from: string; to: string };
}) {
  return (
    <section className="pb-8 md:pb-10 border-b border-line">
      <div
        className="aspect-[16/7] rounded-2xl border border-line"
        style={{
          background: `linear-gradient(135deg, ${cover.from}, ${cover.to})`,
        }}
        aria-hidden
      />
      <div className="mt-6 md:mt-8 flex items-start justify-between gap-6 flex-wrap">
        <div className="min-w-0">
          <Kicker as="p">
            Dispute · <span className="tnum">{disputeId}</span>
          </Kicker>
          <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
            @{runnerHandle}
            <span className="text-slate-soft"> vs </span>
            @{photographerHandle}
          </h1>
          <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft mt-3 tnum">
            {eventName}
            <span className="text-slate-soft"> · </span>
            {formatPrice(amountDisputed)}
            <span className="text-slate-soft"> · </span>
            Reported {formatLongDate(reportedAt)}
          </p>
        </div>
        <AdminStatusPill
          label={STATUS_LABEL[status]}
          tone={statusTone(status)}
        />
      </div>
    </section>
  );
}

function FieldRow({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <dt className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
        {label}
      </dt>
      <dd
        className={`mt-1 ${
          mono
            ? "font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink tnum"
            : "font-sans text-sm text-ink"
        }`}
      >
        {value}
      </dd>
    </div>
  );
}

function ResolutionNote({
  resolution,
  refundAmount,
  resolvedAt,
}: {
  resolution: import("@/lib/admin-disputes").DisputeResolution | null;
  refundAmount: number | null;
  resolvedAt: string | null;
}) {
  if (!resolution) return null;
  return (
    <div className="rounded-xl border border-line bg-bone-deep p-4 mb-4">
      <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
        Closed before live decision logging started
      </p>
      <p className="font-sans text-sm text-ink-soft mt-2">
        {DISPUTE_RESOLUTION_LABEL[resolution]}
        {refundAmount !== null && ` · ${formatPrice(refundAmount)} refunded`}
        {resolvedAt && ` · ${formatLongDate(resolvedAt)}`}
      </p>
    </div>
  );
}
