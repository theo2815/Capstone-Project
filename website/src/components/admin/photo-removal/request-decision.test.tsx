import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as removalApi from "@/lib/api-photo-removal";
import { ApiError } from "@/lib/api";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import type { PhotoRemovalRequestDto } from "@/types/photo-removal";
import { RemovalRequestDecision } from "./request-decision";
import { openCase, ownerReview, removalItem } from "./removal-fixtures";

vi.mock("@/lib/api-photo-removal");
const confirm = vi.fn();
vi.mock("@/hooks/use-confirmation", () => ({ useConfirmation: () => ({ confirm }) }));

const mixedCase: PhotoRemovalRequestDto = {
  ...openCase,
  items: [
    removalItem("a1", "Photographer A", { ownerResponse: "AGREE", ownerRespondedAt: "2026-10-09T00:00:00Z", ownerReviewPending: false }),
    removalItem("b1", "Photographer B", { ownerResponse: "OBJECT", ownerResponseReason: "Model release on file", ownerRespondedAt: "2026-10-09T01:00:00Z", ownerReviewPending: false }),
  ],
  reviews: [ownerReview("Photographer A"), ownerReview("Photographer B")],
};
let client: QueryClient;
let serverCase: PhotoRemovalRequestDto;

beforeEach(() => {
  useAuthStore.getState().startSession(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  serverCase = mixedCase;
  confirm.mockResolvedValue(true);
  vi.mocked(removalApi.getAdminRemovalRequest).mockImplementation(async () => serverCase);
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.resetAllMocks(); });

function renderDecision() {
  return render(<QueryClientProvider client={client}><RemovalRequestDecision requestId="case-1" /></QueryClientProvider>);
}
async function decideItem(user: ReturnType<typeof userEvent.setup>, photoId: string, decision: string, reason: string) {
  await user.selectOptions(await screen.findByLabelText(`Decision for ${photoId}`), decision);
  if (reason) await user.type(screen.getByLabelText(`Reason for ${photoId}`), reason);
}

it("mixedResponsesRequireIndependentDecisions", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.decideRemovalRequest).mockResolvedValue({ ...mixedCase, version: 8 });
  renderDecision();
  expect(await screen.findByText("Photographer A — agreed")).toBeVisible();
  expect(screen.getByText("Photographer B — objected")).toBeVisible();
  expect(screen.getByText(/Model release on file/)).toBeVisible();
  await decideItem(user, "photo-a1", "REMOVE", "Runner verified");
  await decideItem(user, "photo-b1", "REMOVE", "Runner verified");
  expect(screen.getByRole("button", { name: "Authorize removal" })).toBeDisabled();
  expect(screen.getByText(/urgent reason is required/i)).toBeVisible();

  await user.selectOptions(screen.getByLabelText("Decision for photo-b1"), "");
  await user.click(screen.getByRole("button", { name: "Authorize removal" }));
  await waitFor(() => expect(removalApi.decideRemovalRequest).toHaveBeenCalledTimes(1));
  const body = vi.mocked(removalApi.decideRemovalRequest).mock.calls[0][1];
  expect(body.decisions).toEqual([{ itemId: "a1", decision: "REMOVE", reason: "Runner verified" }]);
  expect(body.expectedVersion).toBe(7);
  expect(body.urgentReason).toBeUndefined();
});

it("overrideRequiresVisibleReason", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.decideRemovalRequest).mockResolvedValue({ ...mixedCase, version: 8 });
  renderDecision();
  await decideItem(user, "photo-b1", "REMOVE", "Runner verified");
  const authorize = screen.getByRole("button", { name: "Authorize removal" });
  expect(authorize).toBeDisabled();
  expect(screen.getByText(/urgent reason is required/i)).toBeVisible();
  await user.type(screen.getByLabelText("Urgent override reason"), "Safety risk to runner");
  expect(authorize).toBeEnabled();
  await user.click(authorize);
  expect(confirm).toHaveBeenCalledTimes(1);
  await waitFor(() => expect(removalApi.decideRemovalRequest).toHaveBeenCalledTimes(1));
  expect(vi.mocked(removalApi.decideRemovalRequest).mock.calls[0][1]).toMatchObject({
    decisions: [{ itemId: "b1", decision: "REMOVE", reason: "Runner verified" }], urgentReason: "Safety risk to runner",
  });
  expect(await screen.findByText("Decisions recorded.")).toBeInTheDocument();
});

it("does not require an urgent reason once the server marks the review overdue", async () => {
  const user = userEvent.setup();
  serverCase = { ...mixedCase, items: [removalItem("b1", "Photographer B")],
    reviews: [ownerReview("Photographer B", { overdue: true, overdueAt: "2026-10-15T03:00:00Z" })] };
  renderDecision();
  expect(await screen.findByText("Photographer B — overdue")).toBeVisible();
  await decideItem(user, "photo-b1", "REMOVE", "No response by the deadline");
  expect(screen.getByRole("button", { name: "Authorize removal" })).toBeEnabled();
  expect(screen.queryByText(/urgent reason is required/i)).not.toBeInTheDocument();
});

it("requires a review date before retaining a photo restricted", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.decideRemovalRequest).mockResolvedValue({ ...mixedCase, version: 8 });
  renderDecision();
  await decideItem(user, "photo-a1", "RETAIN_RESTRICTED", "Legal hold");
  const record = screen.getByRole("button", { name: "Record decision" });
  expect(record).toBeDisabled();
  await user.type(screen.getByLabelText("Retention review date for photo-a1"), "2026-12-01");
  await user.click(record);
  await waitFor(() => expect(removalApi.decideRemovalRequest).toHaveBeenCalledTimes(1));
  expect(confirm).not.toHaveBeenCalled();
  expect(vi.mocked(removalApi.decideRemovalRequest).mock.calls[0][1].decisions).toEqual([
    { itemId: "a1", decision: "RETAIN_RESTRICTED", reason: "Legal hold", retentionReviewAt: new Date("2026-12-01T00:00").toISOString() },
  ]);
});

it("blocks duplicate submits, keeps input after a failure and retries the same command", async () => {
  const user = userEvent.setup();
  let fail!: (error: Error) => void;
  vi.mocked(removalApi.decideRemovalRequest)
    .mockReturnValueOnce(new Promise((_, reject) => { fail = reject; }))
    .mockResolvedValueOnce({ ...mixedCase, version: 8 });
  renderDecision();
  await decideItem(user, "photo-a1", "RETAIN", "Not the runner");
  await user.click(screen.getByRole("button", { name: "Record decision" }));
  expect(screen.getByRole("button", { name: "Saving…" })).toBeDisabled();
  await act(async () => { fail(new Error("Gateway timeout")); });
  expect(await screen.findByText("Gateway timeout")).toBeInTheDocument();
  expect(screen.queryByText("Decisions recorded.")).not.toBeInTheDocument();
  expect(screen.getByLabelText("Reason for photo-a1")).toHaveValue("Not the runner");
  await user.click(screen.getByRole("button", { name: "Record decision" }));
  await waitFor(() => expect(removalApi.decideRemovalRequest).toHaveBeenCalledTimes(2));
  const [first, second] = vi.mocked(removalApi.decideRemovalRequest).mock.calls.map((call) => call[1]);
  expect(second.commandId).toBe(first.commandId);
});

it("cleanupFailureNeverRendersErased", async () => {
  const user = userEvent.setup();
  serverCase = { ...mixedCase, status: "RESOLVING", items: [
    removalItem("a1", "Photographer A", { state: "REMOVAL_AUTHORIZED", adminDecision: "REMOVE", cleanupState: "PENDING", cleanupJobId: "s3://bucket/secret-key" }),
    removalItem("b1", "Photographer B", { state: "REMOVED", adminDecision: "REMOVE", cleanupState: "RETRY_REQUIRED", cleanupJobId: "job-secret" }),
    removalItem("c1", "Photographer C", { state: "RETAINED_RESTRICTED", adminDecision: "RETAIN_RESTRICTED", retentionReviewAt: "2026-12-01T00:00:00Z" }),
  ] };
  vi.mocked(removalApi.retryRemovalCleanup).mockResolvedValue({ ...serverCase, version: 8 });
  renderDecision();
  expect(await screen.findByText("Removal authorized · cleanup pending")).toBeVisible();
  expect(screen.getByText("Removed · cleanup failed")).toBeVisible();
  expect(screen.getByText(/Retained restricted · still hidden, not erased/)).toBeVisible();
  expect(screen.queryByText(/erasure complete/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/secret/)).not.toBeInTheDocument();
  expect(screen.getAllByRole("button", { name: /Retry cleanup/ })).toHaveLength(1);
  await user.click(screen.getByRole("button", { name: "Retry cleanup for photo-b1" }));
  await waitFor(() => expect(removalApi.retryRemovalCleanup).toHaveBeenCalledTimes(1));
  expect(vi.mocked(removalApi.retryRemovalCleanup).mock.calls[0][1]).toMatchObject({ itemIds: ["b1"], expectedVersion: 7 });
});

it("shows erasure complete only when the server confirms it", async () => {
  serverCase = { ...mixedCase, status: "RESOLVED", items: [
    removalItem("a1", "Photographer A", { state: "REMOVED", adminDecision: "REMOVE", cleanupState: "COMPLETE" }),
  ] };
  renderDecision();
  expect(await screen.findByText("Removed · erasure complete")).toBeVisible();
  expect(screen.queryByRole("button", { name: /Retry cleanup/ })).not.toBeInTheDocument();
});

it("staleDecisionRefetchesWithoutPretendSuccess", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.decideRemovalRequest).mockImplementationOnce(async () => {
    serverCase = { ...mixedCase, version: 9 };
    throw new ApiError([{ code: "CONFLICT", message: "Case changed" }], 409);
  });
  renderDecision();
  await decideItem(user, "photo-a1", "RETAIN", "Not the runner");
  await user.click(screen.getByRole("button", { name: "Record decision" }));
  expect(await screen.findByRole("button", { name: "Review current state" })).toBeInTheDocument();
  expect(screen.queryByText("Decisions recorded.")).not.toBeInTheDocument();
  expect(removalApi.getAdminRemovalRequest).toHaveBeenCalledTimes(2);
  expect(screen.getByRole("button", { name: "Record decision" })).toBeDisabled();
  expect(screen.getByLabelText("Reason for photo-a1")).toHaveValue("Not the runner");

  vi.mocked(removalApi.decideRemovalRequest).mockResolvedValue({ ...mixedCase, version: 10 });
  await user.click(screen.getByRole("button", { name: "Review current state" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Record decision" })).toBeEnabled());
  await user.click(screen.getByRole("button", { name: "Record decision" }));
  await waitFor(() => expect(removalApi.decideRemovalRequest).toHaveBeenCalledTimes(2));
  expect(vi.mocked(removalApi.decideRemovalRequest).mock.calls[1][1].expectedVersion).toBe(9);
});

it("shows stored notice time, unavailable owners and evidence state from the server", async () => {
  serverCase = { ...mixedCase, evidencePresent: false, evidenceCleanupState: "RETRY_REQUIRED",
    items: [removalItem("a1", "Photographer A"), removalItem("b1", "Photographer B"), removalItem("u1", null)],
    reviews: [ownerReview("Photographer A"), ownerReview("Photographer B", { notifiedAt: null, notifiedMessageId: null, unavailableReason: "Owner account inactive" })] };
  renderDecision();
  expect(await screen.findByText("Photographer A — no response")).toBeVisible();
  expect(screen.getByText(/Notice stored/)).toBeVisible();
  expect(screen.queryByText(/\bread\b/i)).not.toBeInTheDocument();
  expect(screen.getByText("Photographer B — unavailable")).toBeVisible();
  expect(screen.getByText(/Owner unavailable: Owner account inactive/)).toBeVisible();
  expect(screen.getByText(/Unassigned owner/)).toBeVisible();
  expect(screen.getByText(/Evidence deletion blocked/)).toBeVisible();
  expect(screen.getByText(/cannot recall/i)).toBeVisible();
});

it("limits the retention review date to the future", async () => {
  const user = userEvent.setup();
  renderDecision();
  await decideItem(user, "photo-a1", "RETAIN_RESTRICTED", "Legal hold");
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  const expected = `${tomorrow.getFullYear()}-${String(tomorrow.getMonth() + 1).padStart(2, "0")}-${String(tomorrow.getDate()).padStart(2, "0")}`;
  expect(screen.getByLabelText("Retention review date for photo-a1")).toHaveAttribute("min", expected);
});

it("shows readable decisions and the deadline time from the server", async () => {
  serverCase = { ...mixedCase, items: [removalItem("a1", "Photographer A", { state: "RETAINED", adminDecision: "RETAIN", adminDecisionReason: "Not the runner" })],
    reviews: [ownerReview("Photographer A", { overdue: true, overdueAt: "2026-10-15T03:00:00Z" })] };
  renderDecision();
  expect(await screen.findByText(/Decision: Retain/)).toBeVisible();
  expect(screen.queryByText(/Decision: RETAIN/)).not.toBeInTheDocument();
  // 03:00 UTC is 11:00 AM in Manila: notice stored, deadline and overdue-since all need the time, not just the day.
  expect(screen.getAllByText(/11:00/)).toHaveLength(3);
});
