import { webcrypto } from "node:crypto";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { resetUserScopedStores } from "./auth-reset";
import { adminIntentRecoveryNotice, clearAdminActionIntents, claimAdminIntent, completeAdminIntent } from "./admin-action-intents";

beforeEach(() => { vi.stubGlobal("crypto", webcrypto); sessionStorage.clear(); useAuthStore.getState().setUser(admin); });
afterEach(() => { clearAdminActionIntents(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });
it("reuses unfinished normalized payloads after reload without storing their content", async () => {
  const first = await claimAdminIntent(admin.id, "user.message", "user-1", { body: "Private text", subject: " Hello " });
  vi.resetModules();
  const reloaded = await import("./admin-action-intents");
  const replay = await reloaded.claimAdminIntent(admin.id, "user.message", "user-1", { subject: "Hello", body: "Private text" });
  expect(replay.key).toBe(first.key);
  expect(JSON.stringify(sessionStorage)).not.toContain("Private text");
  expect(sessionStorage.getItem("quickpitik-admin-action-intents")).not.toContain("Hello");
  reloaded.completeAdminIntent(replay);
  expect((await reloaded.claimAdminIntent(admin.id, "user.message", "user-1", { subject: "Hello", body: "Private text" })).key).not.toBe(first.key);
  reloaded.clearAdminActionIntents();
});
it("requires reconstruction rather than replacing an uncertain key with changed input", async () => {
  const intent = await claimAdminIntent(admin.id, "event.update", "event-1", { title: "Race", expectedRevision: 3, cover: new File(["first"], "a.jpg") });
  await expect(claimAdminIntent(admin.id, "event.update", "event-1", { title: "Race", expectedRevision: 3, cover: new File(["second"], "a.jpg") })).rejects.toThrow(/original|unchanged|reconstruct/i);
  expect((await claimAdminIntent(admin.id, "event.update", "event-1", { title: "Race", expectedRevision: 3, cover: new File(["first"], "renamed.jpg") })).key).toBe(intent.key);
});
it("isolates actors and targets and clears on the real auth reset", async () => {
  const first = await claimAdminIntent(admin.id, "user.message", "one", { body: "Hi" });
  expect((await claimAdminIntent("other", "user.message", "one", { body: "Hi" })).key).not.toBe(first.key);
  expect((await claimAdminIntent(admin.id, "user.message", "two", { body: "Hi" })).key).not.toBe(first.key);
  resetUserScopedStores();
  expect((await claimAdminIntent(admin.id, "user.message", "one", { body: "Hi" })).key).not.toBe(first.key);
});
it("retains the in-memory key and reports lost reload recovery when storage is unavailable", async () => {
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("Unavailable"); });
  const intent = await claimAdminIntent(admin.id, "user.message", "one", { body: "Hi" });
  expect(intent.reloadSafe).toBe(false);
  expect((await claimAdminIntent(admin.id, "user.message", "one", { body: "Hi" })).key).toBe(intent.key);
  completeAdminIntent(intent);
});
it.each(["succeeded", "missing"] as const)("releases the earlier key and claims the new draft when the server answers %s", async (outcome) => {
  const first = await claimAdminIntent(admin.id, "event.create", "create", { title: "First" });
  const resolve = vi.fn().mockResolvedValue(outcome);
  expect((await claimAdminIntent(admin.id, "event.create", "create", { title: "First" }, resolve)).key).toBe(first.key);
  expect(resolve).not.toHaveBeenCalled();
  const next = await claimAdminIntent(admin.id, "event.create", "create", { title: "Second" }, resolve);
  expect(resolve).toHaveBeenCalledExactlyOnceWith(first.key);
  expect(next.key).not.toBe(first.key); expect(next.reused).toBe(false);
  expect(sessionStorage.getItem("quickpitik-admin-action-intents")).not.toContain(first.key);
});
it.each(["open", "lookup failure"])("keeps the earlier key and blocks the new draft on %s", async (outcome) => {
  const first = await claimAdminIntent(admin.id, "event.create", "create", { title: "First" });
  const resolve = outcome === "open" ? vi.fn().mockResolvedValue("open") : vi.fn().mockRejectedValue(new Error("Offline"));
  const blocked = claimAdminIntent(admin.id, "event.create", "create", { title: "Second" }, resolve);
  await expect(blocked).rejects.toThrow(first.key);
  await expect(blocked).rejects.toThrow(/Unresolved admin request/);
  expect((await claimAdminIntent(admin.id, "event.create", "create", { title: "First" }, resolve)).key).toBe(first.key);
  expect(adminIntentRecoveryNotice(admin.id, "event.create", "create")).toContain(first.key);
});
