import type { Dispute } from "@/lib/admin-disputes";
import type { PayoutReport } from "@/lib/admin-payout-reports";
import type { AdminPayoutCycle } from "@/lib/admin-payouts";
import type { User } from "@/types/user";

export const admin: User = {
  id: "admin-a", role: "ADMIN", name: "Admin", email: "admin@example.test", createdAt: "2026-09-10T00:00:00Z",
};

export const dispute: Dispute = {
  id: "dispute-1", orderId: "order-1", photoId: "photo-1", eventId: "event-1",
  runnerHandle: "runner", photographerHandle: "photographer", reason: "low_quality",
  note: "The photo is blurry.", status: "open", reportedAt: "2026-09-10T00:00:00Z",
  resolvedAt: null, refundAmount: null, itemChargeAmount: 100, resolution: null,
  orderSnapshot: { total: 100, paymentMethod: "gcash", paidAt: "2026-09-09T00:00:00Z" },
  photoSnapshot: { alt: "Runner", bib: "123", kmMark: 5 },
};
export const report: PayoutReport = {
  id: "report-1", payoutCycleId: "cycle-1", photographerId: "photographer-1",
  photographerName: "Photographer", handle: "photographer", reason: "not_received",
  note: "No funds received.", status: "open", reportedAt: "2026-09-10T00:00:00Z",
  acknowledgedAt: null, acknowledgeReply: null, resolvedAt: null, resolutionNote: null,
};

export const payoutCycle: AdminPayoutCycle = {
  id: "cycle-1", photographerId: "photographer-1", photographerName: "Photographer",
  brandName: null, handle: "photographer", weekOf: "2026-09-07", amount: 25000,
  itemCount: 100, method: "gcash", provider: "manual", status: "pending_review",
  submittedAt: "2026-09-10T00:00:00Z", reviewedAt: null, paidAt: null,
  paymentReference: null, holdReason: null, rejectedReason: null, failedReason: null,
  approvedBy: null, secondApprovedBy: null,
  payoutAccount: { method: "gcash", accountNumber: "09000000000", accountName: "Test Photographer", qr: null },
};

export const adminEvent: import("@/lib/api-admin").AdminEventRow = {
  id: "event-1", slug: "cebu-race", name: "Cebu Race", date: "2026-09-10",
  city: "Cebu", location: "Cebu", state: "live", status: "ACTIVE",
  photoCount: 10, participantCount: 20, pricePerPhoto: 80,
  createdByHandle: null, createdByName: null, visibility: "public", pricingMode: "paid",
  watermarkPolicy: "platform", reviewStatus: "approved", reviewNote: null, pendingChange: null,
};

export const adminUser: import("@/lib/admin-user-registry").AdminUserRow = {
  userId: "photographer-1", role: "PHOTOGRAPHER", name: "Test Photographer", email: "photographer@example.test",
  avatarUrl: null, brandName: "Test Studio", handle: "test-studio", region: "Cebu", city: "Cebu",
  createdAt: "2026-09-10T00:00:00Z", verificationStatus: "pending", suspendedAt: null, suspensionReason: null,
  settingsSnapshot: { hasCover: true, hasBrandName: true, hasWatermark: true, hasHandle: true, hasRegion: true, socialCount: 1, payoutCount: 1 },
};
export const adminUserSettings: import("@/lib/api-admin").AdminPhotographerSettingsResponse = {
  userId: adminUser.userId, handle: adminUser.handle!, brandName: adminUser.brandName!, brandColor: "none", bio: "Photographer bio",
  avatarUrl: null, region: null, cover: null, watermark: null, socials: [], payouts: [],
};
