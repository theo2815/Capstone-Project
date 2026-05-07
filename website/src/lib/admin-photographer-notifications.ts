// Helpers that translate admin-side decisions on a photographer into the
// inbox message payload pushed via usePhotographerMessageStore.addMessage.
// Centralized so /admin/photographers/[handle] (admin-action-aside) and
// the verifications drawer + bulk bar (verifications-queue) emit the same
// copy and the same CTA.
//
// TODO(backend): Phase F replaces these with server-side broadcasts —
// admin POSTs the decision, backend writes the photographer's inbox row
// and pushes via WebSocket. Frontend keeps these helpers as the local
// preview/optimistic surface.

import { ROUTES } from "@/lib/constants";
import type { AddMessagePayload } from "@/store/photographer-message-store";
import { usePhotographerMessageStore } from "@/store/photographer-message-store";

interface NotifyDecisionInput {
  photographerId: string;
  brandName: string;
  reason?: string;
}

export function notifyVerificationApproved(input: NotifyDecisionInput): void {
  push({
    photographerId: input.photographerId,
    kind: "verification_approved",
    title: "You’re verified — start uploading.",
    body: `Admin approved ${input.brandName}. Your dashboard now shows the upload tools and your public profile is live.`,
    cta: { href: ROUTES.DASHBOARD, label: "Open dashboard" },
  });
}

export function notifyVerificationRejected(input: NotifyDecisionInput): void {
  const reason = input.reason?.trim();
  const tail = reason ? ` Note from admin: “${reason}”.` : "";
  push({
    photographerId: input.photographerId,
    kind: "verification_rejected",
    title: "Verification needs changes.",
    body: `Admin sent ${input.brandName} back to draft.${tail} Update your settings and resubmit when you’re ready.`,
    cta: { href: ROUTES.DASHBOARD_SETTINGS, label: "Fix settings" },
  });
}

export function notifyAccountSuspended(input: NotifyDecisionInput): void {
  const reason = input.reason?.trim();
  const tail = reason ? ` Reason: “${reason}”.` : "";
  push({
    photographerId: input.photographerId,
    kind: "account_suspended",
    title: "Your account has been suspended.",
    body: `${input.brandName} is paused on QuickPitik.${tail} Sales are on hold until admin lifts the suspension.`,
    cta: { href: ROUTES.DASHBOARD, label: "Open dashboard" },
  });
}

export function notifyAccountUnsuspended(input: NotifyDecisionInput): void {
  push({
    photographerId: input.photographerId,
    kind: "account_unsuspended",
    title: "Your account has been reinstated.",
    body: `${input.brandName} is active again on QuickPitik. Sales resume immediately.`,
    cta: { href: ROUTES.DASHBOARD, label: "Open dashboard" },
  });
}

interface NotifyAdminMessageInput {
  photographerId: string;
  subject: string;
  body: string;
}

// Free-form admin → photographer message. The subject becomes the message
// title (truncation handled by the inbox row); body is passed through. No
// CTA — admin-side messaging is read-only on the photographer surface for
// v1. Phase F adds reply threading + a Compose modal on the photographer
// side; until then this is one-way.
export function notifyAdminMessage(input: NotifyAdminMessageInput): void {
  push({
    photographerId: input.photographerId,
    kind: "admin_message",
    title: input.subject.trim(),
    body: input.body.trim(),
  });
}

function push(payload: AddMessagePayload): void {
  usePhotographerMessageStore.getState().addMessage(payload);
}
