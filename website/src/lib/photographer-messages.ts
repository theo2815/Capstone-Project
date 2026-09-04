// Photographer-facing inbox messages — written BE-side by
// AdminDecisionLogService.pushMessage in the same TX as every admin
// decision (V10 + V15 migrations). FE reads via
// GET /me/photographer/messages. Wire forms below MUST stay aligned with
// the V15 CHECK constraint + PhotographerMessageKind enum in the BE.

export type PhotographerMessageKind =
  | "verification_approved"
  | "verification_rejected"
  | "verification_reset"
  | "event_approved"
  | "event_rejected"
  | "event_change_approved"
  | "event_change_rejected"
  | "suspended"
  | "unsuspended"
  | "force_edit"
  | "dispute_resolved"
  | "dispute_denied"
  | "dispute_escalated"
  | "payout_approved"
  | "payout_held"
  | "payout_paid"
  | "payout_report_acknowledged"
  | "payout_report_resolved"
  | "admin_message";

// FE shape mirrors the BE PhotographerMessageDto (see
// backend/dto/photographer/PhotographerMessageDtos.kt). `title` is non-null
// only for admin_message (admin-supplied subject); every other kind has
// title=null and the FE derives the label from MESSAGE_KIND_LABEL[kind].
export interface PhotographerMessage {
  id: string;
  kind: PhotographerMessageKind;
  title: string | null;
  body: string;
  sourceDecisionId: string | null;
  createdAt: string;
  readAt: string | null;
}

export const MESSAGE_KIND_LABEL: Record<PhotographerMessageKind, string> = {
  verification_approved: "Verification approved",
  verification_rejected: "Verification needs changes",
  verification_reset: "Verification reset",
  event_approved: "Event approved",
  event_rejected: "Event sent back",
  event_change_approved: "Event change approved",
  event_change_rejected: "Event change declined",
  suspended: "Account suspended",
  unsuspended: "Account reinstated",
  force_edit: "Profile updated by admin",
  dispute_resolved: "Refund resolved",
  dispute_denied: "Refund denied",
  dispute_escalated: "Refund escalated",
  payout_approved: "Payout approved",
  payout_held: "Payout held",
  payout_paid: "Payout paid",
  payout_report_acknowledged: "Report acknowledged",
  payout_report_resolved: "Report resolved",
  admin_message: "Message from admin",
};

// Resolve the title shown in the inbox row header. admin_message carries
// the admin-supplied subject in `title`; every other kind falls back to
// the kind label so admin actions don't have to invent per-kind titles.
export function resolveMessageTitle(message: PhotographerMessage): string {
  return message.title?.trim() || MESSAGE_KIND_LABEL[message.kind];
}

export function getUnreadCount(
  messages: ReadonlyArray<PhotographerMessage>,
): number {
  let n = 0;
  for (const m of messages) {
    if (m.readAt === null) n += 1;
  }
  return n;
}
