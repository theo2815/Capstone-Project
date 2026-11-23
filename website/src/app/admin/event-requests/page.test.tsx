import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { admin, adminEvent } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import AdminEventRequestsPage from "./page";
import { expectSectionIntro } from "@/test/admin-section-intro";
import { ROUTES } from "@/lib/constants";

vi.mock("next/navigation", () => ({
  usePathname: () => "/admin/event-requests",
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
}));

beforeEach(() => { useAuthStore.getState().setUser(admin); });
afterEach(() => { cleanup(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

// ER1: the header counts the list it sits above, never the KPI figure or a loading 0.
it("the header equals the list total, with a skeleton while loading", async () => {
  let answer!: (value: unknown) => void;
  vi.spyOn(api, "get").mockImplementation((url: string) =>
    url.startsWith("/admin/kpis")
      ? Promise.resolve({ pendingEventRequests: 9 })
      : new Promise((resolve) => { answer = resolve; }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><AdminEventRequestsPage /></QueryClientProvider>);
  const header = screen.getByRole("banner");
  expect(header).not.toHaveTextContent(/\d+ waiting/);
  expect(header.querySelector('[aria-busy="true"]')).not.toBeNull();
  await act(async () => { answer({ items: [{ ...adminEvent, reviewStatus: "pending" }, { ...adminEvent, id: "e-2", reviewStatus: "pending" }] }); });
  await waitFor(() => expect(header).toHaveTextContent("2 waiting"));
  client.clear();
});

it("shows its guide intro, look-alike links and definitions without hover", () => {
  vi.spyOn(api, "get").mockReturnValue(new Promise(() => {}));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><AdminEventRequestsPage /></QueryClientProvider>);
  expectSectionIntro(ROUTES.ADMIN_EVENT_REQUESTS);
});
