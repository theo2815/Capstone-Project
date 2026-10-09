import { webcrypto } from "node:crypto";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { resetUserScopedStores } from "./auth-reset";
import { adminIntentRecoveryNotice, clearAdminActionIntents, claimAdminIntent, completeAdminIntent } from "./admin-action-intents";

beforeEach(() => { vi.stubGlobal("crypto", webcrypto); sessionStorage.clear(); useAuthStore.getState().setUser(admin); });
afterEach(() => { clearAdminActionIntents(); vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });
const minute = 60_000;
const stored = () => JSON.parse(sessionStorage.getItem("quickpitik-admin-action-intents") ?? "[]") as Record<string, unknown>[];
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
it("releases the earlier key at once when the server says it went through, even just after sending", async () => {
  vi.useFakeTimers({ toFake: ["Date"] });
  const first = await claimAdminIntent(admin.id, "event.create", "create", { title: "First" });
  const resolve = vi.fn().mockResolvedValue("succeeded");
  expect((await claimAdminIntent(admin.id, "event.create", "create", { title: "First" }, resolve)).key).toBe(first.key);
  expect(resolve).not.toHaveBeenCalled();
  const next = await claimAdminIntent(admin.id, "event.create", "create", { title: "Second" }, resolve);
  expect(resolve).toHaveBeenCalledExactlyOnceWith(first.key);
  expect(next.key).not.toBe(first.key); expect(next.reused).toBe(false);
  expect(sessionStorage.getItem("quickpitik-admin-action-intents")).not.toContain(first.key);
});
it("keeps the earlier key on a no-record answer until the wait has passed, and says when to try again", async () => {
  vi.useFakeTimers({ toFake: ["Date"] });
  const first = await claimAdminIntent(admin.id, "event.create", "create", { title: "First" });
  const resolve = vi.fn().mockResolvedValue("missing");
  const blocked = claimAdminIntent(admin.id, "event.create", "create", { title: "Second" }, resolve);
  await expect(blocked).rejects.toThrow(first.key);
  await expect(blocked).rejects.toThrow(/about 5 minutes/);
  vi.setSystemTime(Date.now() + 4 * minute);
  await expect(claimAdminIntent(admin.id, "event.create", "create", { title: "Second" }, resolve)).rejects.toThrow(/about 1 minute\b/);
  expect(stored().map((row) => row.key)).toEqual([first.key]);
  expect(adminIntentRecoveryNotice(admin.id, "event.create", "create")).toContain(first.key);
});
it("releases the earlier key on a no-record answer once the wait has passed", async () => {
  vi.useFakeTimers({ toFake: ["Date"] });
  const first = await claimAdminIntent(admin.id, "event.create", "create", { title: "First" });
  const resolve = vi.fn().mockResolvedValue("missing");
  await expect(claimAdminIntent(admin.id, "event.create", "create", { title: "Second" }, resolve)).rejects.toThrow(first.key);
  vi.setSystemTime(Date.now() + 5 * minute);
  const next = await claimAdminIntent(admin.id, "event.create", "create", { title: "Second" }, resolve);
  expect(next.key).not.toBe(first.key); expect(next.reused).toBe(false);
  expect(stored().map((row) => row.key)).toEqual([next.key]);
});
it("counts the wait from the last time the same draft was sent", async () => {
  vi.useFakeTimers({ toFake: ["Date"] });
  const first = await claimAdminIntent(admin.id, "event.create", "create", { title: "First" });
  vi.setSystemTime(Date.now() + 4 * minute);
  expect((await claimAdminIntent(admin.id, "event.create", "create", { title: "First" })).key).toBe(first.key);
  vi.setSystemTime(Date.now() + 2 * minute);
  await expect(claimAdminIntent(admin.id, "event.create", "create", { title: "Second" }, async () => "missing")).rejects.toThrow(/about 3 minutes/);
  expect(stored().map((row) => row.key)).toEqual([first.key]);
});
it("treats a stored record with no send time as just sent", async () => {
  sessionStorage.setItem("quickpitik-admin-action-intents", JSON.stringify([
    { actor: admin.id, operation: "event.create", target: "create", key: "old-key", fingerprint: "old" }]));
  vi.resetModules();
  const reloaded = await import("./admin-action-intents");
  await expect(reloaded.claimAdminIntent(admin.id, "event.create", "create", { title: "Second" }, async () => "missing")).rejects.toThrow(/old-key/);
  await expect(reloaded.claimAdminIntent(admin.id, "event.create", "create", { title: "Second" }, async () => "missing")).rejects.toThrow(/about 5 minutes/);
  reloaded.clearAdminActionIntents();
});
it("stores a send time beside the key and still no draft content", async () => {
  await claimAdminIntent(admin.id, "user.message", "user-1", { subject: "Hello there", body: "Private text" });
  expect(stored().map((row) => Object.keys(row).sort())).toEqual([["actor", "fingerprint", "key", "operation", "sentAt", "target"]]);
  expect(stored()[0].sentAt).toEqual(expect.any(Number));
  expect(JSON.stringify(stored())).not.toMatch(/Private text|Hello there/);
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
it.each(["event.create", "event.update", "user.message"])("says in the lasting notice how long a different %s request is held", async (operation) => {
  await claimAdminIntent(admin.id, operation, "target-1", { title: "First" });
  const notice = adminIntentRecoveryNotice(admin.id, operation, "target-1");
  expect(notice).toMatch(/held back for about 5 minutes after it was sent/);
  expect(notice).toMatch(/Request reference: \S+\.$/);
});
it("stores the send time it gives an old record once, so a reload does not restart the wait", async () => {
  vi.useFakeTimers({ toFake: ["Date"] });
  sessionStorage.setItem("quickpitik-admin-action-intents", JSON.stringify([
    { actor: admin.id, operation: "event.create", target: "create", key: "old-key", fingerprint: "old" }]));
  vi.resetModules();
  const reloaded = await import("./admin-action-intents");
  expect(reloaded.adminIntentRecoveryNotice(admin.id, "event.create", "create")).toContain("old-key");
  const given = stored()[0].sentAt;
  expect(given).toBe(Date.now());
  vi.setSystemTime(Date.now() + 2 * minute);
  vi.resetModules();
  const again = await import("./admin-action-intents");
  await expect(again.claimAdminIntent(admin.id, "event.create", "create", { title: "Second" }, async () => "missing")).rejects.toThrow(/about 3 minutes/);
  expect(stored()[0].sentAt).toBe(given);
  again.clearAdminActionIntents();
});
