import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import * as removal from "./api-photo-removal";

afterEach(() => vi.restoreAllMocks());
const command = { commandId: "same-command", expectedVersion: 2 };
const signal = new AbortController().signal;

it("reads scoped cases, photos, evidence and history with cancellation and exact filters", async () => {
  const get = vi.spyOn(api, "get").mockResolvedValue({ items: [], total: 0, offset: 0, limit: 50 });
  const blob = new Blob(["private evidence"], { type: "image/png" });
  const binary = vi.spyOn(api, "getBlob").mockResolvedValue(blob);
  await removal.listAdminRemovalRequests({ offset: 0, limit: 20, eventId: "race/a", status: "OPEN", needsAction: false }, signal);
  expect(get).toHaveBeenLastCalledWith("/admin/photo-removal-requests?offset=0&limit=20&eventId=race%2Fa&status=OPEN&needsAction=false", { signal });
  await removal.getAdminRemovalRequest("case/a", signal);
  expect(get).toHaveBeenLastCalledWith("/admin/photo-removal-requests/case%2Fa", { signal });
  await removal.listRemovalPhotos("case/a", { bib: "B & 12", offset: 0 }, signal);
  expect(get).toHaveBeenLastCalledWith("/admin/photo-removal-requests/case%2Fa/photos?bib=B+%26+12&offset=0", { signal });
  await removal.listRemovalHistory("case/a", { limit: 10 }, signal);
  expect(get).toHaveBeenLastCalledWith("/admin/photo-removal-requests/case%2Fa/history?limit=10", { signal });
  expect(await removal.getRemovalEvidence("case/a", signal)).toBe(blob);
  expect(binary).toHaveBeenCalledWith("/admin/photo-removal-requests/case%2Fa/evidence", { signal });
  await removal.listPhotographerRemovalRequests({ needsAction: true }, signal);
  expect(get).toHaveBeenLastCalledWith("/me/photographer/photo-removal-requests?needsAction=true", { signal });
  await removal.getPhotographerRemovalRequest("case/a", signal);
  expect(get).toHaveBeenLastCalledWith("/me/photographer/photo-removal-requests/case%2Fa", { signal });
});

it("uses the canonical command routes and preserves command IDs on retry", async () => {
  const post = vi.spyOn(api, "post").mockResolvedValue({ id: "case" });
  const intake = { eventId: "event", supportReference: "support", requesterContact: "contact", reason: "reason", commandId: "create" };
  await removal.createRemovalRequest(intake);
  expect(post).toHaveBeenLastCalledWith("/admin/photo-removal-requests", intake);
  const verify = { ...command, verificationMethod: "support", verificationNote: "checked", selectedPhotos: [{ photoId: "photo", source: "MANUAL" as const, note: "checked" }] };
  const decision = { ...command, decisions: [{ itemId: "item", decision: "REMOVE" as const, reason: "verified" }] };
  for (const [operation, payload, suffix] of [
    [removal.findRemovalMatches, command, "matches"], [removal.verifyRemovalRequest, verify, "verify"],
    [removal.cancelRemovalRequest, { ...command, reason: "withdrawn" }, "cancel"],
    [removal.decideRemovalRequest, decision, "decisions"],
    [removal.retryRemovalCleanup, { ...command, itemIds: ["item"] }, "retry-cleanup"],
    [removal.addRemovalSupportUpdate, { ...command, note: "contacted" }, "support-updates"],
  ] as const) {
    await (operation as (id: string, command: unknown) => Promise<unknown>)("case", payload);
    expect(post).toHaveBeenLastCalledWith(`/admin/photo-removal-requests/case/${suffix}`, payload);
  }
  const response = { ...command, itemIds: ["item"], response: "OBJECT" as const, reason: "wrong runner" };
  await removal.respondToRemovalRequest("case", response);
  expect(post).toHaveBeenLastCalledWith("/me/photographer/photo-removal-requests/case/responses", response);
  post.mockRejectedValueOnce(new Error("network"));
  await expect(removal.decideRemovalRequest("case", decision)).rejects.toThrow("network");
  await removal.decideRemovalRequest("case", decision);
  expect(post.mock.calls.slice(-2).map((call) => call[1])).toEqual([decision, decision]);
});

it("replaces evidence with multipart PUT through shared transport", async () => {
  const fetch = vi.spyOn(api, "fetch").mockResolvedValue({ id: "case" });
  const file = new File(["private"], "selfie.png", { type: "image/png" });
  await removal.replaceRemovalEvidence("case", file, command.commandId, command.expectedVersion);
  const [path, options] = fetch.mock.calls[0];
  expect(path).toBe("/admin/photo-removal-requests/case/evidence");
  expect(options?.method).toBe("PUT");
  const body = options?.body as FormData;
  expect(body.get("selfie")).toBe(file);
  expect(body.get("commandId")).toBe(command.commandId);
  expect(body.get("expectedVersion")).toBe("2");
  expect([...body.keys()]).toEqual(["selfie", "commandId", "expectedVersion"]);
});
