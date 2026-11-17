"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { Kicker } from "@/components/ui/kicker";
import { useConfirmation } from "@/hooks/use-confirmation";
import { useRunnerRemovalActions, useRunnerRemovalRequest } from "@/hooks/use-photo-removal";
import { ApiError } from "@/lib/api";
import { ROUTES } from "@/lib/constants";
import { RUNNER_REMOVAL_STATUS_NOTE, runnerStatusLabel } from "@/lib/runner-removal-status";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/store/auth-store";
import type { RunnerRemovalCounts } from "@/types/photo-removal";
import { EmailRoute } from "./request-form";
import { RequestTimeline } from "./request-timeline";

/** Other roles never get runner data (the queries only run for RUNNER): they see the email route. */
export function RunnerOnly({ children }: { children: React.ReactNode }) {
  const role = useAuthStore((s) => s.user?.role);
  if (!role) return null;
  if (role === "RUNNER") return <>{children}</>;
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-line bg-bone-deep px-5 py-4">
      <p className="font-display text-lg font-bold text-ink">Runner accounts only</p>
      <p className="font-sans text-sm text-slate">Online deletion requests are for runner accounts.</p>
      <EmailRoute />
    </div>
  );
}

const photos = (n: number) => `${n} ${n === 1 ? "photo" : "photos"}`;
export const countsLine = (c: RunnerRemovalCounts) =>
  `${photos(c.hidden)} hidden · ${c.suggested} more waiting for review · ${c.removed} deleted`;

// RD-12: one request's status, counts and timeline. No images and no photo ids (Global Constraints).
export function RequestStatus({ requestId }: { requestId: string }) {
  const query = useRunnerRemovalRequest(requestId);
  const { withdraw, reviewCurrentState, reviewRequired } = useRunnerRemovalActions();
  const { confirm } = useConfirmation();
  const key = useRef<string | null>(null); // one key per withdraw, reused when a send is retried
  const [notice, setNotice] = useState<string | null>(null);
  const data = query.data;
  const conflict = reviewRequired(requestId);

  // After a 409 the hook has re-read the request; showing it is the runner's acknowledgement.
  useEffect(() => {
    if (conflict && data && !query.isFetching) reviewCurrentState(requestId, data.version);
  }, [conflict, data, query.isFetching, requestId, reviewCurrentState]);

  if (query.isPending) return <p role="status" className="font-sans text-sm text-slate">Loading your request…</p>;
  if (!data) {
    const missing = query.error instanceof ApiError && query.error.status === 404;
    return (
      <div role="alert" className="flex flex-col items-start gap-3 font-sans text-sm text-error">
        <p>{missing ? "We couldn't find this request." : "We couldn't load this request."}</p>
        {!missing && <button type="button" onClick={() => void query.refetch()} className={cn(BTN_GHOST, BTN_SIZE.sm)}>Retry</button>}
      </div>
    );
  }

  async function handleWithdraw(id: string, expectedVersion: number) {
    const ok = await confirm({
      title: "Withdraw this request?",
      message: "Photos hidden for this request become public again unless another request still covers them. You can send a new request for this event afterwards.",
      confirmLabel: "Withdraw",
      cancelLabel: "Keep request",
      danger: true,
    });
    if (!ok) return;
    setNotice(null);
    key.current ??= crypto.randomUUID();
    try {
      await withdraw.mutateAsync({ id, expectedVersion, idempotencyKey: key.current });
      key.current = null;
    } catch (error) {
      const changed = error instanceof ApiError && error.status === 409;
      if (changed) key.current = null;
      setNotice(changed ? "This request changed while you were looking. Here is where it stands now." : "We couldn't withdraw your request. Try again.");
    }
  }

  const note = RUNNER_REMOVAL_STATUS_NOTE[data.status];
  return (
    <article className="flex flex-col gap-8">
      <header className="flex flex-col gap-3">
        <Kicker as="p">Photo deletion request</Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold leading-tight">{data.eventName}</h1>
        <p className="flex items-center gap-2 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink">
          <span aria-hidden="true" className="size-1.5 rounded-full bg-ink shrink-0" />
          <span>{runnerStatusLabel(data.status)}</span>
        </p>
        {note && <p className="font-sans text-sm text-slate">{note}</p>}
        {data.status === "NO_MATCHES_FOUND" && <EmailRoute />}
      </header>
      <p className="font-mono tnum text-sm text-ink">{countsLine(data.counts)}</p>
      <RequestTimeline steps={data.timeline} />
      <p className="max-w-xl font-sans text-sm text-slate leading-relaxed">
        Photos uploaded after your request are not included. You can send a new request once this one closes.
      </p>
      {notice && <p role="alert" className="font-sans text-sm text-error">{notice}</p>}
      <div className="flex flex-wrap items-center gap-4">
        {data.canWithdraw && (
          <button type="button" disabled={withdraw.isPending} onClick={() => void handleWithdraw(data.id, data.version)}
            className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>
            Withdraw request
          </button>
        )}
        <Link href={ROUTES.ACCOUNT_PHOTO_DELETION} className="font-mono text-sm text-ink hover:text-fresh transition-colors">
          All your requests →
        </Link>
      </div>
    </article>
  );
}
