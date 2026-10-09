import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as removalApi from "@/lib/api-photo-removal";
import { ApiError } from "@/lib/api";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import type { PhotoRemovalRequestDto, RemovalMatchResult } from "@/types/photo-removal";
import { RemovalRequestReview } from "./request-review";
import { RemovalRequestDecision } from "./request-decision";
import { draftCase, removalPhoto } from "./removal-fixtures";

vi.mock("@/lib/api-photo-removal");

const photoA = removalPhoto("photo-a", "Photographer A", { matchSource: "SELFIE", matchScore: 0.98 });
const photoB = removalPhoto("photo-b", "Photographer B", { matchSource: "SELFIE", matchScore: 0.91 });
const photoC = removalPhoto("photo-c", "Photographer C", { matchSource: "SELFIE", matchScore: 0.88 });
const falsePositive = removalPhoto("photo-fp", "Photographer A", { matchSource: "SELFIE", matchScore: 0.71 });
const orphan = removalPhoto("photo-orphan", null);
const matchResult: RemovalMatchResult = {
  items: [photoA, photoB, photoC, falsePositive], candidateLimit: 50, possiblyTruncated: false,
  requestVersion: 4, evidenceGeneration: 1, candidatesPurged: false,
};
let client: QueryClient;
let serverCase: PhotoRemovalRequestDto;

beforeEach(() => {
  window.history.pushState({}, "", "/admin/photo-removal-requests/case-1");
  useAuthStore.getState().startSession(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  serverCase = draftCase;
  vi.mocked(removalApi.getAdminRemovalRequest).mockImplementation(async () => serverCase);
  vi.mocked(removalApi.getRemovalEvidence).mockResolvedValue(new Blob(["selfie"], { type: "image/png" }));
  vi.mocked(removalApi.listRemovalPhotos).mockResolvedValue({ items: [orphan], total: 1, offset: 0, limit: 24 });
  vi.mocked(removalApi.findRemovalMatches).mockImplementation(async () => { serverCase = { ...serverCase, version: 4 }; return matchResult; });
  vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:evidence");
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.resetAllMocks(); vi.restoreAllMocks(); });

function renderReview() {
  return render(<QueryClientProvider client={client}><RemovalRequestReview requestId="case-1" /></QueryClientProvider>);
}
const selectedPhotoIds = () => (screen.queryAllByRole("checkbox") as HTMLInputElement[]).filter((box) => box.checked).map((box) => box.value);
const confirmation = () => screen.getByRole("region", { name: "Confirm verification" });
const ownerSections = () => within(confirmation()).getAllByRole("heading", { level: 3 }).map((heading) => heading.textContent);

async function runMatch(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole("button", { name: "Match stored selfie" }));
  await screen.findByRole("checkbox", { name: /photo-a/ });
}
async function fillVerification(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText("Verification method"), "Matched selfie to ticket");
  await user.type(screen.getByLabelText("Verification note"), "Runner confirmed by email");
}

it("matchCandidatesAreNotAutoSelected", async () => {
  const user = userEvent.setup();
  renderReview();
  await runMatch(user);
  expect(selectedPhotoIds()).toEqual([]);
  expect(screen.getByText(/up to/i)).toHaveTextContent("50");
  expect(screen.getByText(/98%/)).toBeInTheDocument();
});

it("selectionListsThreeOwners", async () => {
  const user = userEvent.setup();
  renderReview();
  await runMatch(user);
  for (const id of ["photo-c", "photo-a", "photo-b"]) await user.click(screen.getByRole("checkbox", { name: new RegExp(id) }));
  expect(ownerSections()).toEqual(["Photographer A", "Photographer B", "Photographer C"]);
  expect(confirmation()).toHaveTextContent(/3 photos will be hidden/);
});

it("falsePositiveCanBeExcluded", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.verifyRemovalRequest).mockResolvedValue({ ...draftCase, status: "OPEN", version: 5 });
  renderReview();
  await runMatch(user);
  await user.click(screen.getByRole("checkbox", { name: /photo-a/ }));
  await user.click(screen.getByRole("checkbox", { name: /photo-fp/ }));
  await user.click(screen.getByRole("checkbox", { name: /photo-fp/ }));
  await fillVerification(user);
  await user.click(screen.getByRole("button", { name: "Verify 1 photo" }));
  await waitFor(() => expect(removalApi.verifyRemovalRequest).toHaveBeenCalledTimes(1));
  const verifyBody = vi.mocked(removalApi.verifyRemovalRequest).mock.calls[0][1];
  expect(verifyBody.selectedPhotos.map((p) => p.photoId)).not.toContain(falsePositive.photo.id);
  expect(verifyBody.selectedPhotos).toEqual([{ photoId: "photo-a", source: "SELFIE" }]);
  expect(verifyBody.expectedVersion).toBe(4);
});

it("refreshRestoresServerCaseWithoutPersistingSelfie", async () => {
  const user = userEvent.setup();
  serverCase = { ...draftCase, evidencePresent: false, evidenceContentType: null, evidenceGeneration: 0 };
  vi.mocked(removalApi.replaceRemovalEvidence).mockImplementation(async () => {
    serverCase = { ...draftCase, evidencePresent: true }; return serverCase;
  });
  const view = renderReview();
  const selfie = new File(["runner-face"], "runner-selfie.png", { type: "image/png" });
  await user.upload(await screen.findByLabelText(/Upload a selfie/), selfie);
  await user.click(screen.getByRole("button", { name: "Search" }));
  await screen.findByRole("checkbox", { name: /photo-a/ });
  expect(removalApi.replaceRemovalEvidence).toHaveBeenCalledWith("case-1", selfie, expect.any(String), 3);
  await user.click(screen.getByRole("checkbox", { name: /photo-a/ }));

  view.unmount();
  client.clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  renderReview();
  expect(await screen.findByText("Selfie on file")).toBeInTheDocument();
  expect(selectedPhotoIds()).toEqual([]);
  expect(localStorage.getItem("removal-selfie")).toBeNull();
  const stored = [localStorage, sessionStorage].flatMap((store) => Object.keys(store).map((key) => store.getItem(key) ?? ""));
  expect(stored.join("\n")).not.toMatch(/runner-selfie|runner-face|runner@example\.test|blob:/);
  expect(location.search).toBe("");
  expect(location.pathname).toBe("/admin/photo-removal-requests/case-1");
});

it("keeps photographers who share a display name in separate owner groups", async () => {
  const user = userEvent.setup();
  const twinOne = removalPhoto("photo-t1", "Photographer A", { photographerId: "owner-1", matchSource: "SELFIE", matchScore: 0.9 });
  const twinTwo = removalPhoto("photo-t2", "Photographer A", { photographerId: "owner-2", matchSource: "SELFIE", matchScore: 0.9 });
  vi.mocked(removalApi.findRemovalMatches).mockResolvedValue({ ...matchResult, items: [twinOne, twinTwo] });
  renderReview();
  await user.click(await screen.findByRole("button", { name: "Match stored selfie" }));
  await user.click(await screen.findByRole("checkbox", { name: /photo-t1/ }));
  await user.click(screen.getByRole("checkbox", { name: /photo-t2/ }));
  expect(ownerSections()).toEqual(["Photographer A", "Photographer A"]);
  expect(within(confirmation()).getAllByText((_, el) => el?.tagName === "P" && /review request for 1 photo\./.test(el.textContent ?? ""))).toHaveLength(2);
});

it("refreshes the case when the post-conflict read is not current yet", async () => {
  const user = userEvent.setup();
  let reads = 0;
  vi.mocked(removalApi.getAdminRemovalRequest).mockImplementation(async () => {
    reads += 1;
    if (reads === 2 || reads === 3) throw new Error("Backend unavailable");
    return serverCase;
  });
  vi.mocked(removalApi.findRemovalMatches).mockImplementationOnce(async () => {
    serverCase = { ...serverCase, version: 6 };
    throw new ApiError([{ code: "CONFLICT", message: "Case changed" }], 409);
  });
  renderReview();
  await user.click(await screen.findByRole("button", { name: "Match stored selfie" }));
  await user.click(await screen.findByRole("button", { name: "Review current state" }));
  expect(await screen.findByText(/Couldn.t refresh the case/)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Review current state" }));
  await waitFor(() => expect(screen.queryByRole("button", { name: "Review current state" })).not.toBeInTheDocument());
  await user.click(screen.getByRole("button", { name: "Match stored selfie" }));
  await waitFor(() => expect(vi.mocked(removalApi.findRemovalMatches).mock.calls[1][1].expectedVersion).toBe(6));
});

it("warns about unassigned owners and requires a note for manual additions", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.verifyRemovalRequest).mockResolvedValue({ ...draftCase, status: "OPEN", version: 4 });
  renderReview();
  await user.click(await screen.findByRole("checkbox", { name: /photo-orphan/ }));
  expect(within(confirmation()).getByText(/no photographer will be asked/i)).toBeInTheDocument();
  await fillVerification(user);
  const verify = screen.getByRole("button", { name: "Verify 1 photo" });
  expect(verify).toBeDisabled();
  await user.type(within(confirmation()).getByLabelText(/Note for photo-orphan/), "Runner pointed to this frame");
  await user.click(verify);
  await waitFor(() => expect(removalApi.verifyRemovalRequest).toHaveBeenCalledTimes(1));
  expect(vi.mocked(removalApi.verifyRemovalRequest).mock.calls[0][1].selectedPhotos)
    .toEqual([{ photoId: "photo-orphan", source: "MANUAL", note: "Runner pointed to this frame" }]);
});

it("keeps the selection after a failed verify, blocks duplicates and retries the same command", async () => {
  const user = userEvent.setup();
  let fail!: (error: Error) => void;
  vi.mocked(removalApi.verifyRemovalRequest)
    .mockReturnValueOnce(new Promise((_, reject) => { fail = reject; }))
    .mockImplementationOnce(async () => { serverCase = { ...serverCase, status: "OPEN", version: 5 }; return serverCase; });
  renderReview();
  await runMatch(user);
  await user.click(screen.getByRole("checkbox", { name: /photo-a/ }));
  await fillVerification(user);
  await user.click(screen.getByRole("button", { name: "Verify 1 photo" }));
  expect(screen.getByRole("button", { name: "Verifying…" })).toBeDisabled();
  await act(async () => { fail(new Error("Gateway timeout")); });
  expect(await screen.findByText("Gateway timeout")).toBeInTheDocument();
  expect(selectedPhotoIds()).toEqual(["photo-a"]);
  await user.click(screen.getByRole("button", { name: "Verify 1 photo" }));
  await waitFor(() => expect(removalApi.verifyRemovalRequest).toHaveBeenCalledTimes(2));
  const [first, second] = vi.mocked(removalApi.verifyRemovalRequest).mock.calls.map((call) => call[1]);
  expect(second.commandId).toBe(first.commandId);
});

it("shows matching as unavailable instead of presenting the gallery as matches", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.findRemovalMatches).mockRejectedValue(new ApiError([{ code: "AI_DISABLED", message: "Face search is disabled" }], 503));
  renderReview();
  await user.click(await screen.findByRole("button", { name: "Match stored selfie" }));
  expect(await screen.findByText(/Selfie matching is unavailable/)).toBeInTheDocument();
  expect(screen.queryByRole("region", { name: "Selfie matches" })).not.toBeInTheDocument();
});

it("requires reviewing the current case after a conflict before submitting again", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.findRemovalMatches).mockImplementationOnce(async () => {
    serverCase = { ...serverCase, version: 6 };
    throw new ApiError([{ code: "CONFLICT", message: "Case changed" }], 409);
  });
  renderReview();
  await user.click(await screen.findByRole("button", { name: "Match stored selfie" }));
  await user.click(await screen.findByRole("button", { name: "Review current state" }));
  await waitFor(() => expect(screen.queryByRole("button", { name: "Review current state" })).not.toBeInTheDocument());
  await user.click(screen.getByRole("button", { name: "Match stored selfie" }));
  await waitFor(() => expect(removalApi.findRemovalMatches).toHaveBeenCalledTimes(2));
  expect(vi.mocked(removalApi.findRemovalMatches).mock.calls[1][1].expectedVersion).toBe(6);
});

it("shows the verified case grouped by owner after refresh", async () => {
  serverCase = { ...draftCase, status: "OPEN", verificationMethod: "Ticket", verificationNote: "Checked", verifiedAt: "2026-10-08T02:00:00Z",
    items: [
      { id: "i1", photoIdSnapshot: "photo-a", ownerIdSnapshot: "owner-a", ownerNameSnapshot: "Photographer A", reviewId: "r1", state: "VERIFIED",
        matchSource: "SELFIE", matchScore: 0.98, verificationActorId: "admin-a", verifiedAt: "2026-10-08T02:00:00Z", verificationReason: null,
        ownerResponse: null, ownerResponseReason: null, ownerRespondedAt: null, adminDecision: null, adminDecisionReason: null,
        adminDecisionActorId: null, adminDecidedAt: null, retentionReviewAt: null, cleanupJobId: null, cleanupState: null, photo: null,
        reviewOwnerId: "owner-a", ownerReviewPending: true },
    ] };
  // Owner grouping now lives in the decision panel the case page renders beside the review.
  render(<QueryClientProvider client={client}><RemovalRequestReview requestId="case-1" /><RemovalRequestDecision requestId="case-1" /></QueryClientProvider>);
  expect(await screen.findByRole("heading", { level: 3, name: /^Photographer A — / })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Verify/ })).not.toBeInTheDocument();
  expect(removalApi.listRemovalPhotos).not.toHaveBeenCalled();
});
