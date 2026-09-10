import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { syncVerificationNow } from "./photographer-verification-sync";
import { useAuthStore } from "@/store/auth-store";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";
import { resetUserScopedStores } from "./auth-reset";
const user = { id: "photographer", role: "PHOTOGRAPHER" as const, name: "Photo", email: "photo@example.test", createdAt: "2026-09-10" };
afterEach(() => { useAuthStore.getState().logout(); resetUserScopedStores(); vi.restoreAllMocks(); });

it.each(["logout", "switch", "relogin"])("discards old verification after %s without blocking the next session", async (transition) => {
  useAuthStore.getState().setUser(user);
  let finish!: (value: unknown) => void;
  vi.spyOn(api, "get").mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; })).mockResolvedValue({ status: "pending", missing: [] });
  const first = syncVerificationNow();
  expect(syncVerificationNow()).toBe(first);
  useAuthStore.getState().logout(); resetUserScopedStores();
  if (transition !== "logout") {
    useAuthStore.getState().setUser({ ...user, id: transition === "switch" ? "other" : user.id });
    await syncVerificationNow();
    expect(api.get).toHaveBeenCalledTimes(2);
  }
  finish({ status: "approved", missing: [] }); await first;
  expect(usePhotographerSettingsStore.getState().verificationStatus).toBe(transition === "logout" ? "incomplete" : "pending");
});

it("retries a current-session failure and ignores calls from a runner", async () => {
  vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(api, "get").mockRejectedValueOnce(new Error("Offline")).mockResolvedValue({ status: "approved", missing: [] });
  useAuthStore.getState().setUser({ ...user, role: "RUNNER" }); await syncVerificationNow();
  expect(api.get).not.toHaveBeenCalled();
  useAuthStore.getState().setUser(user); await syncVerificationNow(); await syncVerificationNow();
  expect(usePhotographerSettingsStore.getState().verificationStatus).toBe("approved");
});
