import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { admin, adminUser } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import AdminVerificationsPage from "./page";

// The two queues have their own tests; here only the header counts matter.
vi.mock("@/components/admin/verifications-queue", () => ({ VerificationsQueue: () => null }));
vi.mock("@/components/admin/seller-details-queue", () => ({ SellerDetailsQueue: () => null }));

beforeEach(() => { useAuthStore.getState().setUser(admin); });
afterEach(() => { cleanup(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

function mount(kpis: object, users: Promise<unknown>) {
  vi.spyOn(api, "get").mockImplementation((url: string) => url.startsWith("/admin/kpis") ? Promise.resolve(kpis) : users);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><AdminVerificationsPage /></QueryClientProvider>);
  return screen.getByRole("banner");
}

it("the header counts both queues once they load", async () => {
  const header = mount({ pendingVerifications: 1, pendingSellerDetails: 3 }, Promise.resolve({ items: [adminUser], total: 1 }));
  await waitFor(() => expect(header).toHaveTextContent("Verifications · 1 waiting · Seller details · 3 to review"));
});

it("an older backend without the seller-details count, or a failed list, never shows a 0", async () => {
  const header = mount({ pendingVerifications: 1 }, Promise.reject(new Error("Unavailable")));
  await waitFor(() => expect(api.get).toHaveBeenCalledWith(expect.stringContaining("/admin/kpis"), expect.anything()));
  await new Promise((r) => setTimeout(r, 50));
  expect(within(header).getByText("Verifications", { selector: "p" })).toBeInTheDocument();
  expect(header).not.toHaveTextContent(/\d/);
});
