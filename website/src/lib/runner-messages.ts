// Runner-facing inbox messages — written BE-side by RunnerMessagesService
// in the same TX as every admin decision targeting a runner (dispute
// outcome, suspension, admin DM). FE reads via GET /me/runner/messages.
// Wire forms below MUST stay aligned with the V20 CHECK constraint (V73
// added photo_removed) + RunnerMessageKind enum in the BE.

export type RunnerMessageKind =
  | "dispute_resolved"
  | "dispute_denied"
  | "dispute_escalated"
  | "admin_message"
  | "account_suspended"
  | "account_unsuspended"
  | "photo_removed"
  | "photo_removal_update";

// FE shape mirrors the BE RunnerMessageDto (see
// backend/dto/runner/RunnerMessageDtos.kt). `orderId` deep-links the
// dispute-outcome notifications to /orders?expand={orderId}. `title` is
// non-null only for admin_message; every other kind has title=null and
// the FE derives the label from RUNNER_MESSAGE_KIND_LABEL[kind].
export interface RunnerMessage {
  id: string;
  kind: RunnerMessageKind;
  title: string | null;
  body: string;
  orderId: string | null;
  // RD-12: set on photo_removal_update rows (the runner's own case). A buyer's
  // photo_removed row may carry another person's case id: never link it.
  photoRemovalRequestId?: string | null;
  sourceDecisionId: string | null;
  createdAt: string;
  readAt: string | null;
}

export const RUNNER_MESSAGE_KIND_LABEL: Record<RunnerMessageKind, string> = {
  dispute_resolved: "Refund approved",
  dispute_denied: "Refund declined",
  dispute_escalated: "Refund in review",
  admin_message: "Message from admin",
  account_suspended: "Account suspended",
  account_unsuspended: "Account reinstated",
  photo_removed: "Photo removed",
  photo_removal_update: "Photo deletion",
};

// Resolve the title shown in the inbox row header. admin_message carries
// the admin-supplied subject in `title`; every other kind falls back to
// the kind label.
export function resolveRunnerMessageTitle(message: RunnerMessage): string {
  // ?? "Update": an older tab meeting a kind added later shows a neutral label.
  return message.title?.trim() || (RUNNER_MESSAGE_KIND_LABEL[message.kind] ?? "Update");
}

export function getRunnerUnreadCount(
  messages: ReadonlyArray<RunnerMessage>,
): number {
  let n = 0;
  for (const m of messages) {
    if (m.readAt === null) n += 1;
  }
  return n;
}
