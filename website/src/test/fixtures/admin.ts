import type { Dispute } from "@/lib/admin-disputes";
import type { PayoutReport } from "@/lib/admin-payout-reports";
import type { User } from "@/types/user";

export const admin: User = {
  id: "admin-a", role: "ADMIN", name: "Admin", email: "admin@example.test", createdAt: "2026-09-10T00:00:00Z",
};

export const dispute: Dispute = {
  id: "dispute-1", orderId: "order-1", photoId: "photo-1", eventId: "event-1",
  runnerHandle: "runner", photographerHandle: "photographer", reason: "low_quality",
  note: "The photo is blurry.", status: "open", reportedAt: "2026-09-10T00:00:00Z",
  resolvedAt: null, refundAmount: null, resolution: null,
  orderSnapshot: { total: 100, paymentMethod: "gcash", paidAt: "2026-09-09T00:00:00Z" },
  photoSnapshot: { alt: "Runner", bib: "123", kmMark: 5 },
};
export const report: PayoutReport = {
  id: "report-1", payoutCycleId: "cycle-1", photographerId: "photographer-1",
  photographerName: "Photographer", handle: "photographer", reason: "not_received",
  note: "No funds received.", status: "open", reportedAt: "2026-09-10T00:00:00Z",
  acknowledgedAt: null, acknowledgeReply: null, resolvedAt: null, resolutionNote: null,
};

