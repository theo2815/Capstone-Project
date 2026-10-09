import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as removalApi from "@/lib/api-photo-removal";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { PhotographerInboxModal } from "@/components/layout/photographer-inbox-modal";
import type { PhotographerMessage } from "@/lib/photographer-messages";
import type { PhotographerRemovalItemDto, PhotographerRemovalRequestDto, RemovalPage } from "@/types/photo-removal";
import { PhotographerPhotoRemovalRequest, PhotographerRemovalRequestList } from "./photo-removal-request";

vi.mock("@/lib/api-photo-removal");
const confirm = vi.fn();
vi.mock("@/hooks/use-confirmation", () => ({ useConfirmation: () => ({ confirm }) }));

const requestId = "case-1";
const requesterEmail = "runner@example.test";
const ownerA = { id: "owner-a", role: "PHOTOGRAPHER" as const, name: "Owner A", email: "a@example.test", createdAt: "2026-01-01T00:00:00Z" };
function ownItem(id: string, extra: Partial<PhotographerRemovalItemDto> = {}): PhotographerRemovalItemDto {
  return {
    id, photoIdSnapshot: `photo-${id}`, state: "VERIFIED", response: null, responseReason: null, respondedAt: null,
    decision: null, decisionReason: null, decidedAt: null, retentionReviewAt: null, cleanupState: null,
    photo: null, photoState: "hidden", ownerReviewPending: true, ...extra,
  };
}
const ownerACase: PhotographerRemovalRequestDto = {
  id: requestId, event: { id: "event-1", name: "Cebu City Marathon" }, status: "OPEN", version: 4,
  createdAt: "2026-10-08T00:00:00Z", updatedAt: "2026-10-08T03:00:00Z", resolvedAt: null,
  sanitizedReason: "Privacy removal request",
  items: [ownItem("a1"), ownItem("a2"), ownItem("a3", { response: "AGREE", respondedAt: "2026-10-09T00:00:00Z", ownerReviewPending: false })],
  reviews: [{ id: "review-a", notifiedAt: "2026-10-08T03:00:00Z", deadline: "2026-10-11T03:00:00Z", overdue: false, overdueAt: null,
    responses: { unanswered: 2, agree: 1, objected: 0, alreadyDeleted: 0 } }],
};
const notice: PhotographerMessage = {
  id: "message-1", kind: "photo_removal_request", title: "Photo removal review requested",
  body: "Privacy removal request. Please review these selected items: a1, a2. https://evil.example/login",
  sourceDecisionId: null, createdAt: "2026-10-08T03:00:00Z", readAt: null, photoRemovalRequestId: requestId,
};
let client: QueryClient;
let serverCase: PhotographerRemovalRequestDto;
let rows: PhotographerMessage[];

beforeEach(() => {
  useAuthStore.getState().startSession(ownerA);
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  serverCase = ownerACase;
  rows = [notice];
  confirm.mockResolvedValue(true);
  vi.mocked(removalApi.getPhotographerRemovalRequest).mockImplementation(async () => serverCase);
  vi.mocked(removalApi.listPhotographerRemovalRequests).mockImplementation(async () => ({ items: [serverCase], total: 1, offset: 0, limit: 20 }));
  vi.spyOn(api, "getWithTotal").mockImplementation(async () => ({ data: rows, total: rows.length }));
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); vi.resetAllMocks(); });

function wrap(ui: React.ReactNode) {
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}
const renderDetail = () => wrap(<PhotographerPhotoRemovalRequest requestId={requestId} />);
const respondBody = (call = 0) => vi.mocked(removalApi.respondToRemovalRequest).mock.calls[call][1];

it("noticeOpensScopedDashboardRequest", async () => {
  const user = userEvent.setup();
  const onClose = vi.fn();
  rows = [notice, { ...notice, id: "message-2", kind: "photo_removal_outcome", title: null, readAt: "2026-10-09T00:00:00Z", photoRemovalRequestId: null }];
  wrap(<PhotographerInboxModal isOpen onClose={onClose} />);
  expect(await screen.findByText("Removal review")).toBeVisible();
  expect(screen.getAllByText("Removal decision")[0]).toBeVisible();
  const viewRequestLink = screen.getByRole("link", { name: /View request/ });
  expect(viewRequestLink).toHaveAttribute("href", `/dashboard/photo-removal-requests/${requestId}`);
  // Only the message's request ID builds a link; nothing in the body becomes one.
  expect(screen.getAllByRole("link")).toHaveLength(1);
  vi.spyOn(api, "fetch").mockResolvedValue({ ...notice, readAt: "2026-10-09T01:00:00Z" });
  viewRequestLink.addEventListener("click", (event) => event.preventDefault()); // jsdom cannot navigate
  await user.click(viewRequestLink);
  expect(onClose).toHaveBeenCalled();
  // Opening the request marks the notice read through the backend, not locally.
  await waitFor(() => expect(api.fetch).toHaveBeenCalledWith("/me/photographer/messages/message-1/read", { method: "PATCH" }));
});

it("responsePayloadContainsOnlyDisplayedOwnerItems", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.respondToRemovalRequest).mockResolvedValue({ ...ownerACase, version: 5 });
  renderDetail();
  await user.click(await screen.findByRole("checkbox", { name: "Select all of mine" }));
  await user.click(screen.getByRole("radio", { name: /Agree/ }));
  await user.click(screen.getByRole("button", { name: "Send response" }));
  await waitFor(() => expect(removalApi.respondToRemovalRequest).toHaveBeenCalledTimes(1));
  const ownerAItemIds = ["a1", "a2"];
  const responseBody = respondBody();
  expect(responseBody.itemIds).toEqual(ownerAItemIds);
  expect(responseBody).toMatchObject({ expectedVersion: 4, response: "AGREE" });
  expect(vi.mocked(removalApi.respondToRemovalRequest).mock.calls[0][0]).toBe(requestId);
});

it("noRunnerEvidenceIsRendered", async () => {
  // Even if a response ever carried admin-only fields, the dashboard never renders them.
  serverCase = { ...ownerACase, requesterContact: requesterEmail, requesterReason: "Selfie attached", evidencePresent: true } as PhotographerRemovalRequestDto;
  renderDetail();
  expect(await screen.findByText("Privacy removal request")).toBeVisible();
  expect(screen.queryByText(requesterEmail)).not.toBeInTheDocument();
  expect(screen.queryByText("Selfie attached")).not.toBeInTheDocument();
  expect(screen.queryByAltText("Runner verification selfie")).not.toBeInTheDocument();
  expect(removalApi.getAdminRemovalRequest).not.toHaveBeenCalled();
  expect(removalApi.getRemovalEvidence).not.toHaveBeenCalled();
});

it("failedResponsePreservesInput", async () => {
  const user = userEvent.setup();
  const typedReasonAfterFailure = "I have a signed model release";
  vi.mocked(removalApi.respondToRemovalRequest).mockRejectedValueOnce(new Error("Gateway timeout"));
  renderDetail();
  await user.click(await screen.findByRole("checkbox", { name: "Select photo-a1" }));
  await user.click(screen.getByRole("radio", { name: /Object/ }));
  const objectionInput = screen.getByLabelText("Why you object");
  await user.type(objectionInput, typedReasonAfterFailure);
  await user.click(screen.getByRole("button", { name: "Send response" }));
  expect(await screen.findByText("Gateway timeout")).toBeInTheDocument();
  expect(objectionInput).toHaveValue(typedReasonAfterFailure);
  expect(screen.getByRole("checkbox", { name: "Select photo-a1" })).toBeChecked();
  expect(screen.queryByText(/Response sent/)).not.toBeInTheDocument();
});

it("removedInboxMessageDoesNotRemoveCase", async () => {
  const user = userEvent.setup();
  wrap(<><PhotographerRemovalRequestList /><PhotographerInboxModal isOpen onClose={() => {}} /></>);
  expect(await screen.findByRole("link", { name: /Cebu City Marathon/ })).toBeVisible();
  vi.spyOn(api, "delete").mockImplementation(async () => { rows = []; return { removed: true }; });
  await user.click(await screen.findByRole("button", { name: "Remove notification" }));
  expect(await screen.findByText(/No messages yet/)).toBeVisible();
  const requestListAfterInboxRemoval = client.getQueriesData<RemovalPage<PhotographerRemovalRequestDto>>({ queryKey: ["photo-removal", "photographer"] })
    .filter(([key]) => key.includes("list")).flatMap(([, page]) => page?.items ?? []);
  expect(requestListAfterInboxRemoval).toContainEqual(expect.objectContaining({ id: requestId }));
  expect(screen.getByRole("link", { name: /Cebu City Marathon/ })).toHaveAttribute("href", `/dashboard/photo-removal-requests/${requestId}`);
});

it("requires a reason before an objection can be sent", async () => {
  const user = userEvent.setup();
  renderDetail();
  await user.click(await screen.findByRole("checkbox", { name: "Select photo-a1" }));
  await user.click(screen.getByRole("radio", { name: /Object/ }));
  expect(screen.getByRole("button", { name: "Send response" })).toBeDisabled();
  await user.type(screen.getByLabelText("Why you object"), "   ");
  expect(screen.getByRole("button", { name: "Send response" })).toBeDisabled();
  await user.type(screen.getByLabelText("Why you object"), "Model release on file");
  expect(screen.getByRole("button", { name: "Send response" })).toBeEnabled();
});

it("blocks a duplicate submit and retries with the same command ID", async () => {
  const user = userEvent.setup();
  let fail!: (error: Error) => void;
  vi.mocked(removalApi.respondToRemovalRequest)
    .mockReturnValueOnce(new Promise((_, reject) => { fail = reject; }))
    .mockResolvedValueOnce({ ...ownerACase, version: 5 });
  renderDetail();
  await user.click(await screen.findByRole("checkbox", { name: "Select photo-a2" }));
  await user.click(screen.getByRole("radio", { name: /Already deleted/ }));
  await user.click(screen.getByRole("button", { name: "Send response" }));
  expect(screen.getByRole("button", { name: "Sending…" })).toBeDisabled();
  await act(async () => { fail(new Error("Network error")); });
  await user.click(await screen.findByRole("button", { name: "Send response" }));
  await waitFor(() => expect(removalApi.respondToRemovalRequest).toHaveBeenCalledTimes(2));
  expect(respondBody(1).commandId).toBe(respondBody(0).commandId);
  expect(respondBody(1)).toMatchObject({ itemIds: ["a2"], response: "ALREADY_DELETED" });
  expect(await screen.findByText(/Response sent/)).toBeInTheDocument();
});

it("a conflict requires reviewing the current request before a new response", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.respondToRemovalRequest).mockImplementationOnce(async () => {
    serverCase = { ...ownerACase, version: 6 };
    throw new ApiError([{ code: "CONFLICT", message: "Request changed" }], 409);
  });
  renderDetail();
  await user.click(await screen.findByRole("checkbox", { name: "Select photo-a1" }));
  await user.click(screen.getByRole("radio", { name: /Agree/ }));
  await user.click(screen.getByRole("button", { name: "Send response" }));
  expect(await screen.findByRole("button", { name: "Review current state" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Send response" })).toBeDisabled();
  vi.mocked(removalApi.respondToRemovalRequest).mockResolvedValue({ ...ownerACase, version: 7 });
  await user.click(screen.getByRole("button", { name: "Review current state" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Send response" })).toBeEnabled());
  await user.click(screen.getByRole("button", { name: "Send response" }));
  await waitFor(() => expect(removalApi.respondToRemovalRequest).toHaveBeenCalledTimes(2));
  expect(respondBody(1).expectedVersion).toBe(6);
  expect(respondBody(1).commandId).not.toBe(respondBody(0).commandId);
});

it("shows the deadline with its time, overdue state and read-only outcomes", async () => {
  serverCase = { ...ownerACase, status: "RESOLVING", items: [
    ownItem("a1", { state: "REMOVAL_AUTHORIZED", decision: "REMOVE", cleanupState: "PENDING", response: "OBJECT", responseReason: "Model release", respondedAt: "2026-10-09T00:00:00Z", ownerReviewPending: false }),
    ownItem("a2", { state: "REMOVED", decision: "REMOVE", cleanupState: "COMPLETE", ownerReviewPending: false }),
    ownItem("a3", { state: "RETAINED_RESTRICTED", decision: "RETAIN_RESTRICTED", retentionReviewAt: "2026-12-01T00:00:00Z", ownerReviewPending: false }),
    ownItem("a4", { state: "RETAINED", decision: "RETAIN", ownerReviewPending: false }),
  ], reviews: [{ ...ownerACase.reviews[0], overdue: true, overdueAt: "2026-10-11T03:00:00Z" }] };
  renderDetail();
  expect(await screen.findByText(/Removal authorized · cleanup still running/)).toBeVisible();
  expect(screen.getByText(/Removed · erased from QuickPitik/)).toBeVisible();
  expect(screen.getByText(/Kept restricted · still hidden, not erased/)).toBeVisible();
  expect(screen.getByText(/^Kept$/)).toBeVisible();
  expect(screen.getByText(/You objected/)).toBeVisible();
  expect(screen.getByText(/Model release/)).toBeVisible();
  expect(screen.getByText(/Overdue/)).toBeVisible();
  // 03:00 UTC is 11:00 AM in Manila: the deadline needs its time, not just the day.
  expect(screen.getAllByText(/11:00/).length).toBeGreaterThan(0);
  expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Send response" })).not.toBeInTheDocument();
});

it("shows a not-found message for a case outside this photographer's scope", async () => {
  vi.mocked(removalApi.getPhotographerRemovalRequest).mockRejectedValue(new ApiError([{ code: "NOT_FOUND", message: "Not found" }], 404));
  renderDetail();
  expect(await screen.findByText(/isn't available to your account/)).toBeVisible();
});

it("never reads a failed or unconfirmed cleanup as progress or erasure", async () => {
  serverCase = { ...ownerACase, status: "RESOLVING", items: [
    ownItem("a1", { state: "REMOVAL_AUTHORIZED", decision: "REMOVE", cleanupState: "RETRY_REQUIRED", ownerReviewPending: false }),
    ownItem("a2", { state: "REMOVED", decision: "REMOVE", cleanupState: "UNCONFIRMED", ownerReviewPending: false }),
    ownItem("a3", { state: "REMOVAL_AUTHORIZED", decision: "REMOVE", cleanupState: null, ownerReviewPending: false }),
  ] };
  renderDetail();
  expect(await screen.findByText("Removal authorized · cleanup delayed, QuickPitik is retrying")).toBeVisible();
  expect(screen.getByText("Removal authorized · cleanup not yet confirmed")).toBeVisible();
  expect(screen.getByText("Removal authorized · cleanup still running")).toBeVisible();
  expect(screen.queryByText(/erased/)).not.toBeInTheDocument();
});
