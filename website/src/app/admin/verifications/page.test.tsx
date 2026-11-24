import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { admin, adminUser } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import AdminVerificationsPage from "./page";
import { expectSectionIntro } from "@/test/admin-section-intro";
import { ROUTES } from "@/lib/constants";

// The two queues have their own tests; here the header counts and where Seller details sits.
// The stand-in queue renders its slot where the real one does: after Incomplete, before Decision history.
vi.mock("@/components/admin/verifications-queue", () => ({
  VerificationsQueue: ({ afterIncomplete }: { afterIncomplete?: React.ReactNode }) => <>{afterIncomplete}<p>Decision history</p></>,
}));
vi.mock("@/components/admin/seller-details-queue", () => ({
  SellerDetailsQueue: ({ number = "04" }: { number?: string }) => <p>{`Seller details ${number}`}</p>,
}));

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
  await waitFor(() => expect(header).toHaveTextContent("Verifications · 1 profile waiting · Seller details · 3 waiting"));
});

// FU-6 (U6.1): one profile, every other count the plural.
it.each([[0, "0 profiles waiting"], [2, "2 profiles waiting"]])("the header reads %i as plural", async (n, text) => {
  const items = Array.from({ length: n }, (_, i) => ({ ...adminUser, userId: `pending-${i}` }));
  const header = mount({}, Promise.resolve({ items, total: n }));
  expect(await within(header).findByRole("link", { name: text })).toBeInTheDocument();
});

it("an older backend without the seller-details count, or a failed list, never shows a 0", async () => {
  const header = mount({ pendingVerifications: 1 }, Promise.reject(new Error("Unavailable")));
  await waitFor(() => expect(api.get).toHaveBeenCalledWith(expect.stringContaining("/admin/kpis"), expect.anything()));
  await new Promise((r) => setTimeout(r, 50));
  expect(within(header).getByText("Verifications", { selector: "p" })).toBeInTheDocument();
  expect(header).not.toHaveTextContent(/\d/);
});

it("FT-41: Seller details renders before Decision history, numbered 03", () => {
  mount({}, new Promise(() => {}));
  const order = screen.getAllByText(/^(Seller details \d\d|Decision history)$/).map((el) => el.textContent);
  expect(order).toEqual(["Seller details 03", "Decision history"]);
});

it("FT-41: the two header counts link to their lists", async () => {
  const header = mount({ pendingVerifications: 1, pendingSellerDetails: 3 }, Promise.resolve({ items: [adminUser], total: 1 }));
  expect(await within(header).findByRole("link", { name: "Seller details · 3 waiting" })).toHaveAttribute("href", "#seller-details");
  expect(await within(header).findByRole("link", { name: "1 profile waiting" })).toHaveAttribute("href", "#open-queue");
});

it("shows its guide intro, look-alike links and definitions without hover", () => {
  mount({}, new Promise(() => {}));
  expectSectionIntro(ROUTES.ADMIN_VERIFICATIONS);
});
