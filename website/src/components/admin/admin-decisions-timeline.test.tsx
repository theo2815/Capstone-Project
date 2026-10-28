import { cleanup, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { admin, adminUser } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { AdminDecisionsTimeline } from "./admin-decisions-timeline";

let client: QueryClient;
beforeEach(() => {
  useAuthStore.getState().setUser(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });
const renderTimeline = (props = {}) =>
  render(<QueryClientProvider client={client}><AdminDecisionsTimeline {...props} /></QueryClientProvider>);

const row = (over: object) => ({
  id: crypto.randomUUID(), decision: "approved", reason: null, decidedAt: "2026-10-27T09:30:00Z",
  targetType: "user", targetId: "11111111-2222-3333-4444-555555555555", adminId: "a1", adminName: "Dana Admin", ...over,
});

it("lists recent decisions with a readable label, the admin and the time", async () => {
  const get = vi.spyOn(api, "get").mockResolvedValue([
    row({ decision: "suspended", reason: "Chargebacks" }),
    row({ decision: "PAYOUT_MARKED_PAID", targetType: "payout", targetId: "PAY-1" }),
    row({ decision: "photo_removal_approved", targetType: "removal", adminId: "gone", adminName: null }),
  ]);
  renderTimeline({ recent: true });

  expect(await screen.findByText("Suspended")).toBeTruthy();
  expect(get).toHaveBeenCalledWith("/admin/decisions?limit=10", expect.anything());
  expect(screen.getByText("Payout marked paid")).toBeTruthy();
  expect(screen.getByText("Photo removal approved")).toBeTruthy();
  expect(screen.getAllByText(/by Dana Admin/)).toHaveLength(2);
  expect(screen.getByText(/by unknown admin/)).toBeTruthy();
  expect(screen.getAllByText(/OCT 27/).length).toBe(3);
  expect(screen.getByText("Chargebacks")).toBeTruthy();
  expect(screen.getByText("Payout PAY-1")).toBeTruthy();
});

it("shows the empty copy when nothing has been decided", async () => {
  vi.spyOn(api, "get").mockResolvedValue([]);
  renderTimeline({ recent: true });
  expect(await screen.findByText("No decisions on file yet.")).toBeTruthy();
});

it("with no user id and not opted in, shows the placeholder and makes no request", () => {
  const get = vi.spyOn(api, "get").mockResolvedValue([]);
  renderTimeline();
  expect(screen.getByText(/Decision history is available on each/)).toBeTruthy();
  expect(get).not.toHaveBeenCalled();
});

it("shows who made each decision on a photographer's page", async () => {
  vi.spyOn(api, "get").mockResolvedValue({
    ...adminUser,
    decisionLog: [
      { userId: adminUser.userId, decision: "rejected", reason: "Blurry cover", decidedAt: "2026-10-27T09:30:00Z", adminId: "a1", adminName: "Dana Admin" },
      { userId: adminUser.userId, decision: "approved", reason: null, decidedAt: "2026-10-26T09:30:00Z", adminId: "a2", adminName: null },
    ],
  });
  renderTimeline({ userId: adminUser.userId });

  expect(await screen.findByText("Rejected")).toBeTruthy();
  expect(screen.getByText(/by Dana Admin/)).toBeTruthy();
  expect(screen.getByText(/by unknown admin/)).toBeTruthy();
});
