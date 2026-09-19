import { act, cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { admin, dispute } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import Page from "./page";

vi.mock("next/navigation", () => ({ useParams: () => ({ id: "dispute-1" }), notFound: vi.fn() }));
vi.mock("@/hooks/use-admin-users", () => ({ useAdminUsers: () => ({ data: [] }), EMPTY_ADMIN_USERS: [] }));
let client: QueryClient;
beforeEach(() => {
  useAuthStore.getState().setUser(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => {
  cleanup();
  client.clear();
  useAuthStore.getState().logout();
  vi.restoreAllMocks();
});

it("opens a direct link from an empty cache and displays the backend event name and activity", async () => {
  let resolve!: (value: unknown) => void;
  vi.spyOn(api, "get").mockReturnValueOnce(new Promise((yes) => { resolve = yes; }));
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  expect(screen.getByRole("status")).toHaveTextContent("Loading dispute");
  await act(async () => { resolve({ items: [{ ...dispute, eventName: "Server race name", activity: [{
    id: "server-decision", decision: "escalated", decidedAt: "2026-09-10T01:00:00Z",
    resolution: null, refundAmount: null, reason: "Server audit trail survives refresh",
  }] }], total: 1 }); });
  expect(await screen.findByRole("heading", { name: "@runner vs @photographer" })).toBeInTheDocument();
  expect(screen.getAllByText(/Server race name/).length).toBeGreaterThan(0);
  expect(screen.getByText("Server audit trail survives refresh")).toBeInTheDocument();
});

it("shows a retryable read error instead of treating a failed request as a missing dispute", async () => {
  vi.spyOn(api, "get").mockRejectedValueOnce(new Error("Backend unavailable"));
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't load this dispute");
  vi.mocked(api.get).mockResolvedValueOnce({ items: [dispute], total: 1 });
  await userEvent.setup().click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByRole("heading", { name: "@runner vs @photographer" })).toBeInTheDocument();
});
