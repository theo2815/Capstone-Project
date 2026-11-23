"use client";

import { useRef, useState } from "react";
import { useAdminRemovalActions, useAdminRemovalRequest, useRemovalHistory } from "@/hooks/use-photo-removal";
import { AdminFieldLabel, AdminTextarea } from "@/components/admin/admin-form-fields";
import { FieldError } from "@/components/ui/field-error";
import { BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { removalActionLabel, removalActorLabel, removalStateLabel } from "@/lib/admin-removal-labels";
import { formatLongDate } from "@/lib/format";
import { cn } from "@/lib/utils";
import { REMOVAL_INPUT } from "./request-intake";
import { ReviewCurrentState } from "./request-review";
import type { PhotoRemovalRequestDto } from "@/types/photo-removal";

const PAGE = 50;
const errorText = (error: unknown) => error instanceof Error ? error.message : "Something went wrong. Try again.";

// Server-recorded, append-only timeline (oldest first). Metadata is never shown: it can carry storage internals.
export function RemovalRequestHistory({ requestId }: { requestId: string }) {
  const history = useRemovalHistory(requestId, PAGE);
  const detail = useAdminRemovalRequest(requestId);
  const entries = history.items;
  if (!detail.data) return null;
  return (
    <section aria-labelledby="removal-history" className="mt-12 space-y-6 border-t border-line pt-8">
      <h2 id="removal-history" className="font-display text-xl font-extrabold text-ink">Case history</h2>
      <SupportUpdate request={detail.data} refreshing={detail.isFetching} refresh={detail.refetch} />
      {history.isLoading && <p role="status" className="text-sm text-slate">Loading history…</p>}
      {!!history.error && <p role="alert" className="text-sm text-ink">Couldn&apos;t load the history. <button type="button" className={cn(BTN_GHOST, "underline")} onClick={() => void history.refetch()}>Try again</button></p>}
      <ol className="space-y-3">
        {entries.map((entry) => (
          <li key={entry.id} className="border-l-2 border-line pl-4 text-sm space-y-0.5">
            <p className="font-mono tnum text-slate">{formatLongDate(entry.occurredAt)} · {removalActorLabel(entry.actorType)}{entry.actorIdSnapshot && <> · {entry.actorIdSnapshot}</>}</p>
            <p className="text-ink">
              {removalActionLabel(entry.action)}
              {(entry.fromState || entry.toState) && <> · {entry.fromState ? removalStateLabel(entry.fromState) : "—"} → {entry.toState ? removalStateLabel(entry.toState) : "—"}</>}
              {entry.photoIdSnapshot && <> · <span className="font-mono tnum">{entry.photoIdSnapshot}</span></>}
            </p>
            {entry.reason && <p className="text-slate break-words">{entry.reason}</p>}
          </li>
        ))}
      </ol>
      <LoadMoreButton shown={entries.length} total={history.total} increment={PAGE} onLoadMore={history.fetchNextPage} isLoading={history.isFetchingNextPage} />
    </section>
  );
}

// Records a runner-support outcome or a guest buyer-notice follow-up. It sends no email.
function SupportUpdate({ request, refreshing, refresh }: {
  request: PhotoRemovalRequestDto; refreshing: boolean; refresh: () => Promise<{ data?: PhotoRemovalRequestDto; isError: boolean }>;
}) {
  const actions = useAdminRemovalActions();
  const [note, setNote] = useState("");
  const [noticeId, setNoticeId] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const commands = useRef(new Map<string, string>());
  const pending = actions.supportUpdate.isPending;
  const canSubmit = !!note.trim() && !pending && !refreshing && !actions.reviewRequired(request.id);

  async function submit() {
    if (!canSubmit) return;
    setError(null); setSaved(false);
    const body = { expectedVersion: request.version, note: note.trim(), noticeId: noticeId || null };
    const fingerprint = JSON.stringify(body);
    if (!commands.current.has(fingerprint)) commands.current.set(fingerprint, crypto.randomUUID());
    try {
      await actions.supportUpdate.mutateAsync({ id: request.id, command: { ...body, commandId: commands.current.get(fingerprint)! } });
      commands.current.delete(fingerprint);
      setNote(""); setNoticeId(""); setSaved(true);
    } catch (failure) {
      setError(errorText(failure));
    }
  }

  return (
    <div className="space-y-3">
      <ReviewCurrentState request={request} actions={actions} refreshing={refreshing} refresh={refresh} />
      {request.guestBuyerRemovalNotices.length > 0 && (
        <div className="space-y-1 text-sm">
          <h3 className="font-display font-bold text-ink">Guest buyer notices</h3>
          <p className="text-slate">QuickPitik cannot message guest buyers. Contact each one yourself, then note it below.</p>
          <ul className="space-y-1 text-slate">
            {request.guestBuyerRemovalNotices.map((notice) => (
              <li key={notice.noticeId} className="break-words">
                <span className="tnum">{formatLongDate(notice.occurredAt)}</span> · {removalStateLabel(notice.status)} · {notice.recipientEmail} · order <span className="font-mono tnum">{notice.orderId}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <AdminTextarea id="removal-support-note" label="Note what support told the requester or a guest buyer" maxLength={2000} value={note} disabled={pending}
        placeholder="What was said, and to whom" describedBy="removal-support-note-help" onChange={setNote} />
      <p id="removal-support-note-help" className="text-sm text-slate">This only adds a note to the history. It does not message anyone.</p>
      {request.guestBuyerRemovalNotices.length > 0 && (
        <div className="space-y-2">
          <AdminFieldLabel htmlFor="removal-support-notice">Guest buyer notice (optional)</AdminFieldLabel>
          <select id="removal-support-notice" value={noticeId} disabled={pending} onChange={(e) => setNoticeId(e.target.value)} className={REMOVAL_INPUT}>
            <option value="">Runner support (no notice)</option>
            {request.guestBuyerRemovalNotices.map((notice) => (
              <option key={notice.noticeId} value={notice.noticeId}>{notice.recipientEmail} · order {notice.orderId}</option>
            ))}
          </select>
        </div>
      )}
      <FieldError message={error} id="removal-support-error" />
      {saved && <p role="status" className="text-sm text-ink">Support update recorded.</p>}
      <button type="button" disabled={!canSubmit} onClick={() => void submit()} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>
        {pending ? "Recording…" : "Record update"}
      </button>
    </div>
  );
}
