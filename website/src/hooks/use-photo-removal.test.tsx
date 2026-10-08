import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { useAdminQueueRealtime } from "./use-admin-queue-realtime";
import { useAdminWsStatusStore } from "@/store/admin-ws-status-store";
import type { PhotoRemovalRequestDto, PhotographerRemovalRequestDto } from "@/types/photo-removal";
import { removalKeys, useAdminRemovalActions, useAdminRemovalRequest, useAdminRemovalRequests, useRemovalPhotos, useRemovalHistory, useRemovalEvidence, usePhotographerRemovalRequest, usePhotographerRemovalRequests, usePhotographerRemovalActions } from "./use-photo-removal";

const admin = { id: "admin-a", role: "ADMIN" as const, name: "Admin", email: "a@example.test", createdAt: "2026-10-08" };
const oldAdminCase: PhotoRemovalRequestDto = {
  id: "case", event: { id: "event", name: "Race" }, status: "OPEN", version: 2,
  createdAt: "2026-10-08T00:00:00Z", updatedAt: "2026-10-08T00:00:00Z", resolvedAt: null,
  phaseCounts: { candidate: 0, verified: 1, excluded: 0, removalAuthorized: 0, removed: 0, retained: 0, retainedRestricted: 0, pendingOwnerReviews: 1 },
  requesterUserId: null, requesterContact: "requester@example.test", supportReference: "ticket-1", requesterReason: "Privacy",
  publicToOwnerReason: "Privacy removal request", creatorId: admin.id, verifierId: admin.id,
  verificationMethod: "support", verificationNote: "Contact checked", verifiedAt: "2026-10-08T00:00:00Z",
  evidencePresent: true, evidenceContentType: "image/png", evidenceGeneration: 1, evidenceCleanupState: "PRESENT",
  outstandingEvidenceCount: 0, reviews: [], guestBuyerRemovalNotices: [],
  items: [{ id: "item", photoIdSnapshot: "photo", ownerIdSnapshot: "owner", ownerNameSnapshot: "Owner", reviewId: null,
    state: "VERIFIED", matchSource: "SELFIE", matchScore: 99, verificationActorId: admin.id, verifiedAt: "2026-10-08T00:00:00Z",
    verificationReason: "Checked", ownerResponse: null, ownerResponseReason: null, ownerRespondedAt: null,
    adminDecision: null, adminDecisionReason: null, adminDecisionActorId: null, adminDecidedAt: null, retentionReviewAt: null,
    cleanupJobId: null, cleanupState: null, photo: null, reviewOwnerId: "owner", ownerReviewPending: true }],
};
const decision = { commandId: "stable-command", expectedVersion: 2, decisions: [{ itemId: "item", decision: "REMOVE" as const, reason: "verified" }] };
let client: QueryClient;
function wrapper({ children }: { children: ReactNode }) { return <QueryClientProvider client={client}>{children}</QueryClientProvider>; }
function session() { const state = useAuthStore.getState(); return [state.user?.id ?? null, state.user?.role ?? null, state.sessionVersion] as const; }
function deferred<T>() { let resolve!: (value: T) => void; let reject!: (error: Error) => void; const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }
beforeEach(() => { useAuthStore.getState().startSession(admin); client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } }); });
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); vi.useRealTimers(); });

it.each(["degraded", "healthy"] as const)("includes removal cases in the existing %s admin fallback", async (status) => {
  vi.useFakeTimers(); useAdminWsStatusStore.getState().setStatus(status);
  const key = removalKeys.admin(session()); client.setQueryDefaults(key, { gcTime: Infinity }); client.setQueryData(key, []);
  renderHook(() => useAdminQueueRealtime(true), { wrapper });
  await act(async () => { await vi.advanceTimersByTimeAsync(status === "healthy" ? 300_000 : 30_000); });
  expect(client.getQueryState(key)?.isInvalidated).toBe(true);
});

it.each(["switch", "same-user-login", "logout"])("lateResponseCannotPopulateAnotherSession: %s", async (transition) => {
  const read = deferred<unknown>(); const get = vi.spyOn(api, "get").mockReturnValueOnce(read.promise).mockResolvedValue({ items: [], total: 0, offset: 0, limit: 50 });
  const { result } = renderHook(useAdminRemovalRequests, { wrapper });
  await waitFor(() => expect(get).toHaveBeenCalledTimes(1));
  const firstSignal = get.mock.calls[0][1]?.signal;
  act(() => { useAuthStore.getState().logout(); client.clear(); if (transition !== "logout") useAuthStore.getState().startSession(transition === "switch" ? { ...admin, id: "admin-b" } : admin); });
  await act(async () => read.resolve({ items: [oldAdminCase], total: 1, offset: 0, limit: 50 }));
  expect(firstSignal?.aborted).toBe(true);
  expect(result.current.data?.items ?? []).not.toContainEqual(oldAdminCase);
  expect(client.getQueriesData({ queryKey: removalKeys.admin(session()) }).flatMap(([, data]) => data ? [data] : [])).not.toContainEqual(oldAdminCase);
});

it("failedDecisionDoesNotCreateSuccessState", async () => {
  const pending = deferred<unknown>(); vi.spyOn(api, "post").mockReturnValue(pending.promise);
  const key = [...removalKeys.admin(session()), "detail", "case"];
  client.setQueryData(key, oldAdminCase);
  function Decision() { const { decide } = useAdminRemovalActions(); const [success, setSuccess] = useState(false); return <><button onClick={() => void decide.mutateAsync({ id: "case", command: decision }).then(() => setSuccess(true)).catch(() => {})}>Decide</button>{success && <p>Removal authorized</p>}<span>{decide.error?.message}</span></>; }
  render(wrapper({ children: <Decision /> }));
  fireEvent.click(screen.getByText("Decide"));
  await waitFor(() => expect(api.post).toHaveBeenCalledTimes(1));
  expect(client.getQueryData(key)).toEqual(oldAdminCase);
  await act(async () => pending.reject(new Error("Backend refused")));
  await screen.findByText("Backend refused");
  expect(screen.queryByText("Removal authorized")).not.toBeInTheDocument();
  expect(client.getQueryData(key)).toEqual(oldAdminCase);
});

it("waits for canonical results, invalidates related queries and preserves submitted commands on retry", async () => {
  const post = vi.spyOn(api, "post").mockRejectedValueOnce(new Error("transport")).mockResolvedValue({ ...oldAdminCase, version: 3 });
  const keys = [[...removalKeys.admin(session()), "list", {}], [...removalKeys.admin(session()), "detail", "case"], [...removalKeys.admin(session()), "history", "case", {}], ["events"], ["public", "events"], ["photographer", "events"], ["orders"], ["inbox"], ["admin", "kpis"]];
  keys.forEach((key) => client.setQueryData(key, key.includes("detail") ? oldAdminCase : []));
  const { result } = renderHook(useAdminRemovalActions, { wrapper });
  await expect(result.current.decide.mutateAsync({ id: "case", command: decision })).rejects.toThrow("transport");
  await waitFor(() => expect(result.current.decide.variables?.command).toEqual(decision));
  await act(async () => { await result.current.decide.mutateAsync({ id: "case", command: decision }); });
  expect(post.mock.calls.map((call) => call[1])).toEqual([decision, decision]);
  expect(client.getQueryData(keys[1])).toEqual({ ...oldAdminCase, version: 3 });
  keys.filter((key) => !key.includes("detail")).forEach((key) => expect(client.getQueryState(key)?.isInvalidated).toBe(true));
});

it("409 refetches current state and blocks a changed command until reviewed", async () => {
  const post = vi.spyOn(api, "post").mockRejectedValueOnce(new ApiError([{ code: "CONFLICT", message: "Changed" }], 409)).mockResolvedValue({ ...oldAdminCase, version: 4 });
  const get = vi.spyOn(api, "get").mockResolvedValue({ ...oldAdminCase, version: 3 });
  const { result } = renderHook(useAdminRemovalActions, { wrapper });
  await act(async () => { await expect(result.current.decide.mutateAsync({ id: "case", command: decision })).rejects.toThrow("Changed"); });
  expect(get).toHaveBeenCalledWith("/admin/photo-removal-requests/case", expect.objectContaining({ signal: expect.any(AbortSignal) }));
  const changed = { ...decision, commandId: "new-command", expectedVersion: 3 };
  await expect(result.current.decide.mutateAsync({ id: "case", command: changed })).rejects.toThrow(/review/i);
  expect(post).toHaveBeenCalledTimes(1);
  expect(result.current.reviewCurrentState("case", 2)).toBe(false);
  expect(result.current.reviewCurrentState("case", 3)).toBe(true);
  await act(async () => { await result.current.decide.mutateAsync({ id: "case", command: changed }); });
  expect(post).toHaveBeenCalledTimes(2);
});

it.each(["decide", "replaceEvidence", "match"] as const)("rejects late %s results from a replaced session", async (action) => {
  const pending = deferred<unknown>(); vi.spyOn(api, action === "replaceEvidence" ? "fetch" : "post").mockReturnValue(pending.promise);
  const { result } = renderHook(useAdminRemovalActions, { wrapper });
  const operation = action === "decide" ? result.current.decide.mutateAsync({ id: "case", command: decision }) : action === "match" ? result.current.match.mutateAsync({ id: "case", command: { commandId: "search", expectedVersion: 2 } }) : result.current.replaceEvidence.mutateAsync({ id: "case", file: new File(["selfie"], "selfie.png"), commandId: "upload", expectedVersion: 2 });
  const rejected = expect(operation).rejects.toThrow("Session changed");
  await waitFor(() => expect(action === "replaceEvidence" ? api.fetch : api.post).toHaveBeenCalledTimes(1));
  act(() => { useAuthStore.getState().startSession({ ...admin, id: "admin-b" }); client.clear(); });
  pending.resolve(oldAdminCase); await rejected;
  expect(client.getQueriesData({ queryKey: removalKeys.admin(session()) })).toEqual([]);
});

it("gates all private reads and mutations by role", async () => {
  useAuthStore.getState().startSession({ ...admin, role: "RUNNER" });
  const get = vi.spyOn(api, "get"); const binary = vi.spyOn(api, "getBlob"); const post = vi.spyOn(api, "post");
  const { result } = renderHook(() => ({ list: useAdminRemovalRequests(), detail: useAdminRemovalRequest("case"), photos: useRemovalPhotos("case"), history: useRemovalHistory("case"), evidence: useRemovalEvidence("case"), ownerList: usePhotographerRemovalRequests(), ownerDetail: usePhotographerRemovalRequest("case"), actions: useAdminRemovalActions(), ownerActions: usePhotographerRemovalActions() }), { wrapper });
  expect(get).not.toHaveBeenCalled(); expect(binary).not.toHaveBeenCalled();
  await expect(result.current.actions.decide.mutateAsync({ id: "case", command: decision })).rejects.toThrow("Session changed");
  await expect(result.current.ownerActions.respond.mutateAsync({ id: "case", command: { commandId: "owner", expectedVersion: 2, itemIds: ["item"], response: "AGREE" } })).rejects.toThrow("Session changed");
  expect(post).not.toHaveBeenCalled();
});

it("does not imperatively refetch private cases from a disabled role", async () => {
  useAuthStore.getState().startSession({ ...admin, role: "RUNNER" });
  const get = vi.spyOn(api, "get").mockResolvedValue({ items: [], total: 0, offset: 0, limit: 50 });
  const { result } = renderHook(useAdminRemovalRequests, { wrapper });
  const refetch = await result.current.refetch();
  expect(get).not.toHaveBeenCalled();
  expect(refetch.error?.message).toBe("Session changed");
});

it("keeps owner projection separate and installs only a backend-confirmed response", async () => {
  useAuthStore.getState().startSession({ ...admin, role: "PHOTOGRAPHER" });
  const ownerCase: PhotographerRemovalRequestDto = { id: "case", event: { id: "event", name: "Race" }, status: "OPEN", version: 2,
    createdAt: "2026-10-08T00:00:00Z", updatedAt: "2026-10-08T00:00:00Z", resolvedAt: null,
    sanitizedReason: "Privacy", items: [], reviews: [] };
  const get = vi.spyOn(api, "get").mockResolvedValue(ownerCase);
  vi.spyOn(api, "post").mockResolvedValue({ ...ownerCase, version: 3 });
  const { result } = renderHook(() => ({ detail: usePhotographerRemovalRequest("case"), actions: usePhotographerRemovalActions() }), { wrapper });
  await waitFor(() => expect(result.current.detail.data).toEqual(ownerCase));
  expect(get).toHaveBeenCalledWith("/me/photographer/photo-removal-requests/case", expect.objectContaining({ signal: expect.any(AbortSignal) }));
  await act(async () => { await result.current.actions.respond.mutateAsync({ id: "case", command: { commandId: "owner", expectedVersion: 2, itemIds: ["item"], response: "OBJECT", reason: "wrong runner" } }); });
  expect(client.getQueriesData({ queryKey: removalKeys.admin(session()) })).toEqual([]);
});
