import type { RunnerRemovalStatus, RunnerTimelineStep } from "@/types/photo-removal";

// Display labels for the runner's deletion status codes (RD-12); codes stay codes on the client.
export const RUNNER_REMOVAL_STATUS_LABEL: Record<RunnerRemovalStatus, string> = {
  PROCESSING: "Processing",
  PENDING_ADMIN_REVIEW: "Pending admin review",
  ADDITIONAL_VERIFICATION_REQUIRED: "Additional verification required",
  NO_MATCHES_FOUND: "No matches found",
  APPROVED: "Approved: deleting",
  REJECTED: "Rejected",
  COMPLETED: "Completed",
  WITHDRAWN: "Withdrawn",
};

// The extra line some statuses carry (NO_MATCHES_FOUND shows the email route instead).
export const RUNNER_REMOVAL_STATUS_NOTE: Partial<Record<RunnerRemovalStatus, string>> = {
  ADDITIONAL_VERIFICATION_REQUIRED: "Our team will email you.",
  REJECTED: "You can reply by email within 14 days.",
};

// Default copy; the owner may reword.
export const RUNNER_TIMELINE_STEP_LABEL: Record<RunnerTimelineStep, string> = {
  SUBMITTED: "Request sent",
  PHOTOS_HIDDEN: "Photos hidden",
  SENT_FOR_REVIEW: "Sent for review",
  MORE_INFO_NEEDED: "More information needed",
  APPROVED: "Deletion approved",
  KEPT: "Kept",
  WITHDRAWN: "Withdrawn",
  CLOSED: "Closed",
};

/** An older tab meeting a code added later shows a neutral word, never a raw code. */
export const runnerStatusLabel = (status: string) => RUNNER_REMOVAL_STATUS_LABEL[status as RunnerRemovalStatus] ?? "Update";
export const runnerTimelineLabel = (step: string) => RUNNER_TIMELINE_STEP_LABEL[step as RunnerTimelineStep] ?? "Update";
