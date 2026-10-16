import type { PhotoRemovalItemDto, PhotoRemovalRequestDto, PhotoRemovalReviewDto, RemovalPhotoDto } from "@/types/photo-removal";

export const draftCase: PhotoRemovalRequestDto = {
  id: "case-1", event: { id: "event-unlisted", name: "Private Fun Run" }, status: "DRAFT", version: 3,
  createdAt: "2026-10-08T00:00:00Z", updatedAt: "2026-10-08T00:00:00Z", resolvedAt: null,
  phaseCounts: { candidate: 0, verified: 0, excluded: 0, removalAuthorized: 0, removed: 0, retained: 0, retainedRestricted: 0, pendingOwnerReviews: 0 },
  requesterUserId: null, requesterContact: "runner@example.test", supportReference: "TICKET-42", requesterReason: "Runner asked",
  publicToOwnerReason: "Privacy removal request", creatorId: "admin-a", verifierId: null,
  verificationMethod: null, verificationNote: null, verifiedAt: null,
  evidencePresent: true, evidenceContentType: "image/png", evidenceGeneration: 1, evidenceCleanupState: "PRESENT",
  outstandingEvidenceCount: 0, items: [], reviews: [], guestBuyerRemovalNotices: [],
};

export function removalPhoto(id: string, photographerName: string | null, extra: Partial<RemovalPhotoDto> = {}): RemovalPhotoDto {
  return {
    photo: {
      id, bib: `B-${id}`, km: null, time: "07:00", tone: 1, span: "default", price: 80, imageUrl: `https://cdn.example.test/${id}.jpg`,
      alt: null, cleanUrl: null, photographerHandle: null, photographerName, couponCode: null, couponPercentOff: null,
      couponPrice: null, free: false, downloadUrl: null, metadataAccess: "none",
    },
    photographerId: photographerName ? `owner-${photographerName}` : null, photographerName,
    itemId: null, removalState: "NONE", trashed: false, bytesAvailable: true, matchScore: null, matchSource: null, ...extra,
  };
}

export function removalItem(id: string, owner: string | null, extra: Partial<PhotoRemovalItemDto> = {}): PhotoRemovalItemDto {
  return {
    id, photoIdSnapshot: `photo-${id}`, ownerIdSnapshot: owner && `owner-${owner}`, ownerNameSnapshot: owner,
    reviewId: owner && `review-${owner}`, state: "VERIFIED", matchSource: "SELFIE", matchScore: 0.9, verificationActorId: "admin-a",
    verifiedAt: "2026-10-08T02:00:00Z", verificationReason: null, ownerResponse: null, ownerResponseReason: null, ownerRespondedAt: null,
    adminDecision: null, adminDecisionReason: null, adminDecisionActorId: null, adminDecidedAt: null, retentionReviewAt: null,
    cleanupJobId: null, cleanupState: null, photo: null, reviewOwnerId: owner && `owner-${owner}`, ownerReviewPending: !!owner, ...extra,
  };
}

export function ownerReview(owner: string, extra: Partial<PhotoRemovalReviewDto> = {}): PhotoRemovalReviewDto {
  return {
    id: `review-${owner}`, ownerIdSnapshot: `owner-${owner}`, notifiedMessageId: "message-1", notifiedAt: "2026-10-08T03:00:00Z",
    deadline: new Date(Date.now() + 7 * 24 * 3600 * 1000).toISOString(), overdue: false, overdueAt: null, unavailableReason: null, ...extra,
  };
}

export const openCase: PhotoRemovalRequestDto = {
  ...draftCase, status: "OPEN", version: 7, verificationMethod: "Ticket", verificationNote: "Checked", verifiedAt: "2026-10-08T02:00:00Z",
  evidenceCleanupState: "PRESENT",
};
