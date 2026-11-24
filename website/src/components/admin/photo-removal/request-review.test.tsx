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
import { draftCase, openCase, ownerReview, removalItem, removalPhoto, runnerCase } from "./removal-fixtures";
import { formatLongDate } from "@/lib/format";

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
const confirmVerify = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(within(await screen.findByRole("dialog", { name: "Hide these photos and ask for a review?" })).getByRole("button", { name: "Hide and ask for review" }));
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
  // FT-11 (C-73): the button asks first; only the confirm hides the photos.
  const ask = await screen.findByRole("dialog", { name: "Hide these photos and ask for a review?" });
  expect(within(ask).getByText((_, el) => el?.tagName === "P").textContent).toBe("1 photo is hidden from sale and search right away. Each photographer who owns one gets a review request in their inbox and by email, with a deadline to answer. You can put a photo back later with Retain.");
  expect(removalApi.verifyRemovalRequest).not.toHaveBeenCalled();
  await user.click(within(ask).getByRole("button", { name: "Hide and ask for review" }));
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
  expect(await screen.findByText("Couldn't refresh the request. Try again.")).toBeInTheDocument();
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
  await user.type(within(confirmation()).getByLabelText(/Note for photo 1/), "Runner pointed to this frame");
  await user.click(verify);
  await confirmVerify(user);
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
  await confirmVerify(user);
  expect(screen.getByRole("button", { name: "Verifying…" })).toBeDisabled();
  await act(async () => { fail(new Error("Gateway timeout")); });
  expect(await screen.findByText("Gateway timeout")).toBeInTheDocument();
  expect(selectedPhotoIds()).toEqual(["photo-a"]);
  await user.click(screen.getByRole("button", { name: "Verify 1 photo" }));
  await confirmVerify(user);
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

it("labelsAnErasedPhotoRemovedNotInTrash", async () => {
  vi.mocked(removalApi.listRemovalPhotos).mockResolvedValue({
    items: [removalPhoto("photo-gone", "Photographer A", { removalState: "REMOVED", trashed: true, bytesAvailable: false })],
    total: 1, offset: 0, limit: 24,
  });
  renderReview();
  expect(await screen.findByText("Removed")).toBeInTheDocument();
  expect(screen.queryByText("In trash")).not.toBeInTheDocument();
});

it("names photo thumbnails by position, never by the raw photo ID", async () => {
  const user = userEvent.setup();
  renderReview();
  await runMatch(user);
  const labels = screen.getAllByRole("button", { name: /^Preview / }).map((button) => button.getAttribute("aria-label"));
  expect(labels.length).toBeGreaterThan(0);
  for (const label of labels) expect(label).toMatch(/^Preview photo \d+ of \d+$/);
  expect(labels).toContain("Preview photo 1 of 4");
});

// RD-13: runner cases.
const runnerOpen: PhotoRemovalRequestDto = {
  ...runnerCase,
  items: [
    removalItem("h1", "Photographer A", { autoHidden: true, verificationActorId: null, faceCount: 1, otherPeopleVisible: false }),
    removalItem("h2", "Photographer A", { verificationActorId: null, faceCount: 3, otherPeopleVisible: true, autoHidden: true }),
    removalItem("c1", "Photographer B", { state: "CANDIDATE", verificationActorId: null, verifiedAt: null, faceCount: 1, otherPeopleVisible: false }),
  ],
};

it("the review shows account age and email verified", async () => {
  serverCase = runnerOpen;
  renderReview();
  expect(await screen.findByText(`Account created ${formatLongDate("2026-09-01T00:00:00Z")} · Email verified yes`)).toBeVisible();
  serverCase = { ...runnerOpen, requesterEmailVerified: false };
  cleanup(); client.clear();
  renderReview();
  expect(await screen.findByText(/Email verified no/)).toBeVisible();
});

it("the band shows words, never a number", async () => {
  serverCase = { ...runnerOpen, livenessBand: "UNCERTAIN", riskFlags: ["LIVENESS_UNCERTAIN", "SOMETHING_NEW"] };
  renderReview();
  const band = await screen.findByText(/^Live check:/);
  expect(band).toHaveTextContent("Live check: uncertain");
  expect(band.textContent).not.toMatch(/\d/);
  expect(screen.getByText("The live check was uncertain")).toBeVisible();
  expect(screen.getByText("SOMETHING_NEW")).toBeVisible();
  expect(screen.queryByText("LIVENESS_UNCERTAIN")).toBeNull();
});

it("groups auto-hidden photos and public suggestions with an other-people badge", async () => {
  serverCase = runnerOpen;
  renderReview();
  const hidden = await screen.findByRole("region", { name: "Hidden automatically" });
  const suggestions = screen.getByRole("region", { name: "Possible matches (still public)" });
  expect(within(hidden).getAllByRole("listitem")).toHaveLength(2);
  expect(within(hidden).getAllByText("Other people visible")).toHaveLength(1);
  expect(within(suggestions).getAllByRole("listitem")).toHaveLength(1);
});

it("an auto-hidden photo shows its preview", async () => {
  serverCase = { ...runnerOpen, items: runnerOpen.items.map((item) => item.id === "h1" ? { ...item, photo: removalPhoto("photo-x", "Photographer A") } : item) };
  const { container } = renderReview();
  await screen.findByRole("region", { name: "Hidden automatically" });
  expect(container.querySelector('img[src="https://cdn.example.test/photo-x.jpg"]')).not.toBeNull();
});

it("promote sends the expected version to the suggestions endpoint", async () => {
  const user = userEvent.setup();
  serverCase = runnerOpen;
  vi.mocked(removalApi.reviewRemovalSuggestions).mockImplementation(async () => { serverCase = { ...runnerOpen, version: 8 }; return serverCase; });
  renderReview();
  await user.click(await screen.findByRole("button", { name: "Hide this photo… photo-c1" }));
  // AX-36 RM3: Hide this photo… hides a public photo, so it waits for a confirm.
  const dialog = await screen.findByRole("dialog", { name: "Hide this photo?" });
  expect(removalApi.reviewRemovalSuggestions).not.toHaveBeenCalled();
  await user.click(within(dialog).getByRole("button", { name: "Hide photo" }));
  expect(await screen.findByText("Photo hidden and sent to its photographer for review.")).toBeVisible();
  await waitFor(() => {
    expect(removalApi.getAdminRemovalRequest).toHaveBeenCalledTimes(2);
    expect(screen.getByRole("button", { name: "Not this runner… photo-c1" })).toBeEnabled();
  });
  await waitFor(() => expect(removalApi.reviewRemovalSuggestions).toHaveBeenCalledTimes(1));
  const [id, body] = vi.mocked(removalApi.reviewRemovalSuggestions).mock.calls[0];
  expect(id).toBe("case-1");
  expect(body).toEqual({ commandId: expect.any(String), expectedVersion: 7, promote: ["c1"], dismiss: [], demote: [] });
  await user.click(screen.getByRole("button", { name: "Not this runner… photo-c1" }));
  await user.click(within(await screen.findByRole("dialog", { name: "Remove this possible match?" })).getByRole("button", { name: "Remove" }));
  await waitFor(() => expect(removalApi.reviewRemovalSuggestions).toHaveBeenCalledTimes(2));
  expect(vi.mocked(removalApi.reviewRemovalSuggestions).mock.calls[1][1]).toMatchObject({ expectedVersion: 8, dismiss: ["c1"] });
});

it("demote is offered only on auto-hidden photos", async () => {
  serverCase = { ...runnerOpen, items: [...runnerOpen.items,
    removalItem("v1", "Photographer B", { autoHidden: false }),
    removalItem("d1", "Photographer B", { autoHidden: true, adminDecision: "MORE_EVIDENCE" })] };
  renderReview();
  await screen.findByRole("region", { name: "Hidden automatically" });
  const demotes = screen.getAllByRole("button", { name: /^Put back on sale…/ }).map((b) => b.textContent);
  expect(demotes).toHaveLength(2);
  expect(demotes.join()).toMatch(/photo-h1.*photo-h2/);
  expect(screen.queryByRole("button", { name: /^Hide this photo… photo-h1/ })).toBeNull();
});

it("runner cases have no evidence replace", async () => {
  serverCase = { ...runnerCase, status: "DRAFT", riskFlags: ["MATCHING_UNAVAILABLE"] };
  const { container } = renderReview();
  expect(await screen.findByRole("button", { name: "Match stored selfie" })).toBeVisible();
  expect(container.querySelector('input[type="file"]')).toBeNull();
  expect(screen.queryByText("Upload a selfie")).toBeNull();
});

it("a runner case without evidence offers only manual verify or cancel", async () => {
  serverCase = { ...runnerCase, status: "DRAFT", evidencePresent: false, evidenceCleanupState: "NONE" };
  const { container } = renderReview();
  expect(await screen.findByRole("button", { name: "Cancel request" })).toBeDisabled();
  expect(screen.getByText(/, or cancel the request\.$/)).toBeVisible();
  expect(container.querySelector('input[type="file"]')).toBeNull();
  expect(screen.queryByRole("button", { name: "Match stored selfie" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Search" })).toBeNull();
  expect(await screen.findByRole("checkbox", { name: /photo-orphan/ })).toBeVisible();
  expect(removalApi.getRemovalEvidence).not.toHaveBeenCalled();
});

// AX-36.
it("the header shows the status label, not the raw code", async () => {
  serverCase = { ...openCase, status: "RESOLVING" };
  renderReview();
  expect(await screen.findByText("Closing")).toBeVisible();
  expect(screen.queryByText("RESOLVING")).toBeNull();
});

it("the case page has an h1 while loading and on error", async () => {
  vi.mocked(removalApi.getAdminRemovalRequest).mockRejectedValue(new Error("down"));
  renderReview();
  expect(screen.getByRole("heading", { level: 1, name: "Removal request" })).toBeInTheDocument();
  expect(await screen.findByText("Couldn't load this request.")).toBeVisible();
  expect(screen.getByRole("heading", { level: 1, name: "Removal request" })).toBeInTheDocument();
});

// FT-21 (C-80): Not this runner… has no undo, so it waits for a confirm that says so.
it("Not this runner… calls nothing until Remove is confirmed", async () => {
  const user = userEvent.setup();
  serverCase = runnerOpen;
  vi.mocked(removalApi.reviewRemovalSuggestions).mockImplementation(async () => { serverCase = { ...runnerOpen, version: 8 }; return serverCase; });
  renderReview();
  await user.click(await screen.findByRole("button", { name: "Not this runner… photo-c1" }));
  const dialog = await screen.findByRole("dialog", { name: "Remove this possible match?" });
  expect(within(dialog).getByText("The photo stays public and nobody is told. You cannot bring it back to this list.")).toBeVisible();
  expect(removalApi.reviewRemovalSuggestions).not.toHaveBeenCalled();
  await user.click(within(dialog).getByRole("button", { name: "Remove" }));
  expect(await screen.findByText("Possible match removed.")).toBeVisible();
  expect(vi.mocked(removalApi.reviewRemovalSuggestions).mock.calls[0][1]).toMatchObject({ expectedVersion: 7, dismiss: ["c1"], promote: [], demote: [] });
});

it("Cancel on Not this runner… sends nothing, and Put back on sale says where the photo went", async () => {
  const user = userEvent.setup();
  serverCase = runnerOpen;
  vi.mocked(removalApi.reviewRemovalSuggestions).mockImplementation(async () => serverCase);
  renderReview();
  await user.click(await screen.findByRole("button", { name: "Not this runner… photo-c1" }));
  await user.click(within(await screen.findByRole("dialog", { name: "Remove this possible match?" })).getByRole("button", { name: "Cancel" }));
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(removalApi.reviewRemovalSuggestions).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: /^Put back on sale… for photo-h1/ }));
  const putBack = await screen.findByRole("dialog", { name: "Put this photo back on sale?" });
  expect(within(putBack).getByText("It goes back on sale and in search, and stays on this request as a possible match.")).toBeVisible();
  await user.click(within(putBack).getByRole("button", { name: "Put back on sale" }));
  expect(await screen.findByText("The photo is back on sale. It stays listed here as a possible match.")).toBeVisible();
  expect(vi.mocked(removalApi.reviewRemovalSuggestions).mock.calls[0][1]).toMatchObject({ demote: ["h1"], promote: [], dismiss: [] });
});

it("no suggestion button says promote, demote or dismiss", async () => {
  serverCase = runnerOpen;
  renderReview();
  const suggestions = await screen.findByRole("region", { name: "Possible matches (still public)" });
  expect(within(suggestions).getByText("Photos that may show this runner and were not hidden. Hide one to send it to its photographer for review, or mark it as not this runner.")).toBeVisible();
  const names = screen.getAllByRole("button").map((b) => b.textContent ?? "");
  expect(names.join(" | ")).not.toMatch(/promote|demote|dismiss|suggestion/i);
});

it("the promote confirm is disabled, not silent, while the case refreshes", async () => {
  const user = userEvent.setup();
  serverCase = runnerOpen;
  renderReview();
  await user.click(await screen.findByRole("button", { name: "Hide this photo… photo-c1" }));
  const dialog = await screen.findByRole("dialog", { name: "Hide this photo?" });
  vi.mocked(removalApi.getAdminRemovalRequest).mockReturnValue(new Promise(() => {}));
  await act(async () => { void client.invalidateQueries(); });
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Hide photo" })).toBeDisabled());
});

it("make public again needs a confirm and Cancel sends nothing", async () => {
  const user = userEvent.setup();
  serverCase = runnerOpen;
  renderReview();
  await user.click(await screen.findByRole("button", { name: /^Put back on sale… for photo-h1/ }));
  const dialog = await screen.findByRole("dialog", { name: "Put this photo back on sale?" });
  await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(removalApi.reviewRemovalSuggestions).not.toHaveBeenCalled();
});

it("the confirm step shows each photo, not its ID", async () => {
  const user = userEvent.setup();
  renderReview();
  await runMatch(user);
  await user.click(screen.getByRole("checkbox", { name: /photo-a/ }));
  expect(within(confirmation()).getByRole("img", { name: "Photo 1 of Private Fun Run" })).toBeVisible();
  expect(within(confirmation()).queryByText(/(^|[^-])photo-a/)).toBeNull();
  expect(within(confirmation()).getByText(/Selfie match/)).toBeVisible();
});

it("a manual note is not marked invalid before the admin types or leaves it", async () => {
  const user = userEvent.setup();
  renderReview();
  await user.click(await screen.findByRole("checkbox", { name: /photo-orphan/ }));
  const note = within(confirmation()).getByLabelText(/Note for photo 1/);
  expect(note).not.toHaveAttribute("aria-invalid");
  await user.click(note); await user.tab();
  expect(note).toHaveAttribute("aria-invalid", "true");
});

it("the event gallery loads more photos and keeps the first page", async () => {
  const user = userEvent.setup();
  const first = Array.from({ length: 24 }, (_, i) => removalPhoto(`photo-${i}`, "Photographer A"));
  vi.mocked(removalApi.listRemovalPhotos).mockImplementation(async (_id, filters = {}) => filters.offset
    ? { items: [removalPhoto("photo-last", "Photographer A")], total: 25, offset: 24, limit: 24 }
    : { items: first, total: 25, offset: 0, limit: 24 });
  renderReview();
  await screen.findByRole("checkbox", { name: /photo-0$/ });
  await user.click(screen.getByRole("button", { name: "Load 1 more" }));
  expect(await screen.findByRole("checkbox", { name: /photo-last/ })).toBeVisible();
  expect(screen.getByRole("checkbox", { name: /photo-0$/ })).toBeVisible();
  expect(removalApi.listRemovalPhotos).toHaveBeenLastCalledWith("case-1", { offset: 24, limit: 24 }, expect.anything());
});

// FT-32 (C-74, W32.1, W32.4): the header names the step, and a support request is called a request.
it("the header says which step the request is in", async () => {
  serverCase = draftCase;
  renderReview();
  expect(await screen.findByText("Step 1 of 4. Choose the photos this request is about, then verify them.")).toBeVisible();
  expect(screen.getByText(`Support request · ${formatLongDate(draftCase.createdAt)}`)).toBeVisible();
});

it("the Verification method field gives an example", async () => {
  const user = userEvent.setup();
  serverCase = { ...draftCase, status: "DRAFT", evidencePresent: false };
  renderReview();
  await user.click(await screen.findByRole("checkbox", { name: /photo-orphan/ }));
  expect(screen.getByLabelText("Verification method")).toHaveAttribute("placeholder", "For example: same bib, same race kit, selfie compared by eye");
});

// FU-12 (U12.6): the stored selfie is the request's evidence.
it("the stored selfie is called request evidence", async () => {
  serverCase = { ...runnerCase, status: "DRAFT" };
  renderReview();
  expect(await screen.findByText("Private request evidence. It is not added to anyone's selfie library.")).toBeVisible();
});

// FU-12: the step line is worked out again when an owner's deadline passes, without a refetch.
it("the step line moves on when the owner's deadline passes", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try {
    const deadline = new Date(Date.now() + 60_000).toISOString();
    serverCase = { ...openCase, items: [removalItem("a1", "A")], reviews: [ownerReview("A", { deadline })] };
    renderReview();
    expect(await screen.findByText(/^Step 2 of 4\. Waiting for 1 photographer to answer, until /)).toBeVisible();
    act(() => { vi.advanceTimersByTime(61_000); });
    expect(screen.getByText("Step 3 of 4. Ready for your decision on each photo.")).toBeVisible();
    expect(removalApi.getAdminRemovalRequest).toHaveBeenCalledTimes(1);
  } finally { vi.useRealTimers(); }
});

// A deadline past setTimeout's longest wait (about 24.8 days) still flips the line on time.
it("the step line moves on for a deadline weeks away", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try {
    const deadline = new Date(Date.now() + 30 * 24 * 3600 * 1000).toISOString();
    serverCase = { ...openCase, items: [removalItem("a1", "A")], reviews: [ownerReview("A", { deadline })] };
    renderReview();
    expect(await screen.findByText(/^Step 2 of 4\. Waiting for 1 photographer/)).toBeVisible();
    act(() => { vi.advanceTimersByTime(29 * 24 * 3600 * 1000); });
    expect(screen.getByText(/^Step 2 of 4\. Waiting for 1 photographer/)).toBeVisible();
    act(() => { vi.advanceTimersByTime(24 * 3600 * 1000 + 1000); });
    expect(screen.getByText("Step 3 of 4. Ready for your decision on each photo.")).toBeVisible();
  } finally { vi.useRealTimers(); }
});
