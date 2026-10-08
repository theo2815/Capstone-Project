import type { MockPhoto } from "./photo";
import type { PaginatedResponse } from "./api";

// Wire shapes mirror backend dto/removal/PhotoRemovalDtos.kt; authority is server-derived.
export type RemovalRequestStatus = "DRAFT" | "OPEN" | "RESOLVING" | "RESOLVED" | "CANCELLED";
export type RemovalItemState = "CANDIDATE" | "VERIFIED" | "EXCLUDED" | "REMOVAL_AUTHORIZED" | "REMOVED" | "RETAINED" | "RETAINED_RESTRICTED";
export type PhotographerRemovalResponse = "AGREE" | "OBJECT" | "ALREADY_DELETED";
export type RemovalDecision = "REMOVE" | "RETAIN" | "MORE_EVIDENCE" | "RETAIN_RESTRICTED";
export type RemovalState = "NONE" | "REVIEW" | "REMOVED";
export type RemovalMatchSource = "SELFIE" | "BIB" | "MANUAL";
export type RemovalActorType = "ADMIN" | "PHOTOGRAPHER" | "SYSTEM" | "SUPPORT";
export type RemovalEvidenceCleanupState = "NONE" | "PRESENT" | "PENDING" | "DELETED" | "RETRY_REQUIRED";
export type RemovalPage<T> = PaginatedResponse<T>;
export interface RemovalEventDto { id: string; name: string }
export interface RemovalPhaseCounts {
  candidate: number; verified: number; excluded: number; removalAuthorized: number; removed: number;
  retained: number; retainedRestricted: number; pendingOwnerReviews: number;
}
export type RemovalPhotoPresentation = Omit<MockPhoto, "alt" | "imageUrl" | "cleanUrl" | "photographerHandle" | "photographerName" | "couponCode" | "couponPercentOff" | "couponPrice" | "free" | "downloadUrl" | "metadataAccess"> & {
  alt: string | null; imageUrl: string | null; cleanUrl: string | null; photographerHandle: string | null;
  photographerName: string | null; couponCode: string | null; couponPercentOff: number | null;
  couponPrice: number | null; free: boolean; downloadUrl: string | null; metadataAccess: NonNullable<MockPhoto["metadataAccess"]>;
};
export interface RemovalPhotoDto {
  photo: RemovalPhotoPresentation; photographerId: string | null; photographerName: string | null;
  itemId: string | null; removalState: RemovalState; trashed: boolean; bytesAvailable: boolean;
  matchScore: number | null; matchSource: RemovalMatchSource | null;
}
export interface PhotoRemovalItemDto {
  id: string; photoIdSnapshot: string; ownerIdSnapshot: string | null; ownerNameSnapshot: string | null;
  reviewId: string | null; state: RemovalItemState; matchSource: RemovalMatchSource; matchScore: number | null;
  verificationActorId: string | null; verifiedAt: string | null; verificationReason: string | null;
  ownerResponse: PhotographerRemovalResponse | null; ownerResponseReason: string | null; ownerRespondedAt: string | null;
  adminDecision: RemovalDecision | null; adminDecisionReason: string | null; adminDecisionActorId: string | null;
  adminDecidedAt: string | null; retentionReviewAt: string | null; cleanupJobId: string | null; cleanupState: string | null;
  photo: RemovalPhotoDto | null; reviewOwnerId: string | null; ownerReviewPending: boolean;
}
export interface PhotoRemovalReviewDto {
  id: string; ownerIdSnapshot: string; notifiedMessageId: string | null; notifiedAt: string | null;
  deadline: string | null; overdue: boolean; overdueAt: string | null; unavailableReason: string | null;
}
export interface GuestBuyerRemovalNoticeDto {
  noticeId: string; orderId: string; photoId: string; occurredAt: string; status: string; recipientEmail: string;
}
export interface PhotoRemovalRequestDto {
  id: string; event: RemovalEventDto; status: RemovalRequestStatus; version: number; createdAt: string; updatedAt: string;
  resolvedAt: string | null; phaseCounts: RemovalPhaseCounts; requesterUserId: string | null; requesterContact: string;
  supportReference: string; requesterReason: string; publicToOwnerReason: string; creatorId: string; verifierId: string | null;
  verificationMethod: string | null; verificationNote: string | null; verifiedAt: string | null;
  evidencePresent: boolean; evidenceContentType: string | null; evidenceGeneration: number;
  evidenceCleanupState: RemovalEvidenceCleanupState; outstandingEvidenceCount: number;
  items: PhotoRemovalItemDto[]; reviews: PhotoRemovalReviewDto[]; guestBuyerRemovalNotices: GuestBuyerRemovalNoticeDto[];
}
export interface PhotographerRemovalItemDto {
  id: string; photoIdSnapshot: string; state: RemovalItemState; response: PhotographerRemovalResponse | null;
  responseReason: string | null; respondedAt: string | null; decision: RemovalDecision | null;
  decisionReason: string | null; decidedAt: string | null; retentionReviewAt: string | null; cleanupState: string | null;
  photo: RemovalPhotoPresentation | null; photoState: string | null; ownerReviewPending: boolean;
}
export interface RemovalResponseCounts { unanswered: number; agree: number; objected: number; alreadyDeleted: number }
export interface PhotographerRemovalReviewDto {
  id: string; notifiedAt: string | null; deadline: string | null; overdue: boolean; overdueAt: string | null; responses: RemovalResponseCounts;
}
export interface PhotographerRemovalRequestDto {
  id: string; event: RemovalEventDto; status: RemovalRequestStatus; version: number; createdAt: string; updatedAt: string;
  resolvedAt: string | null; sanitizedReason: string; items: PhotographerRemovalItemDto[]; reviews: PhotographerRemovalReviewDto[];
}
export interface RemovalHistoryDto {
  id: string; requestId: string; itemId: string | null; reviewId: string | null; photoIdSnapshot: string | null;
  actorType: RemovalActorType; actorIdSnapshot: string | null; action: string; fromState: string | null;
  toState: string | null; reason: string | null; occurredAt: string; commandId: string; metadata: Record<string, string>;
}
export interface RemovalMatchResult {
  items: RemovalPhotoDto[]; candidateLimit: number; possiblyTruncated: boolean;
  requestVersion: number; evidenceGeneration: number; candidatesPurged: boolean;
}
export interface MatchRemovalCommand { commandId: string; expectedVersion: number }
export interface CreateRemovalRequestCommand {
  eventId: string; supportReference: string; requesterContact: string; requesterUserId?: string | null;
  reason: string; commandId: string; adminNote?: string | null;
}
export interface CancelRemovalCommand extends MatchRemovalCommand { reason: string }
export interface VerifyRemovalSelection { photoId: string; source: RemovalMatchSource; note?: string | null }
export interface VerifyRemovalCommand extends MatchRemovalCommand {
  verificationMethod: string; verificationNote: string; selectedPhotos: VerifyRemovalSelection[];
}
export interface RespondToRemovalCommand extends MatchRemovalCommand {
  itemIds: string[]; response: PhotographerRemovalResponse; reason?: string | null;
}
export interface RemovalItemDecision { itemId: string; decision: RemovalDecision; reason: string; retentionReviewAt?: string | null }
export interface DecideRemovalCommand extends MatchRemovalCommand { decisions: RemovalItemDecision[]; urgentReason?: string | null }
export interface RetryRemovalCleanupCommand extends MatchRemovalCommand { itemIds: string[] }
export interface RemovalSupportUpdateCommand extends MatchRemovalCommand { note: string; noticeId?: string | null }
export interface RemovalRequestFilters { offset?: number; limit?: number; eventId?: string; status?: RemovalRequestStatus; needsAction?: boolean }
export interface RemovalPhotoFilters { offset?: number; limit?: number; bib?: string }
export interface RemovalPageFilters { offset?: number; limit?: number }
