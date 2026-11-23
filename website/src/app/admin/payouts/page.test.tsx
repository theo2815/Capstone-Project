import { cleanup, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as adminApi from "@/lib/api-admin";
import { ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { admin, payoutCycle } from "@/test/fixtures/admin";
import AdminPayoutsPage from "./page";
import { expectSectionIntro } from "@/test/admin-section-intro";
import { ROUTES } from "@/lib/constants";

vi.mock("@/lib/api-admin", async (orig) => ({
  ...(await orig<typeof import("@/lib/api-admin")>()),
  fetchAdminPayouts: vi.fn(), fetchAdminPayoutCounts: vi.fn(),
}));
// The header is under test; the queue and reports have their own tests.
vi.mock("@/components/admin/payouts-queue", () => ({ PayoutsQueue: () => null }));
vi.mock("@/components/admin/payout-reports-section", () => ({ PayoutReportsSection: () => null }));

let client: QueryClient;
const mount = () => render(<QueryClientProvider client={client}><AdminPayoutsPage /></QueryClientProvider>);
const header = () => screen.getByText((_, el) => el?.tagName === "P" && /^Payouts/.test(el.textContent ?? ""));

beforeEach(() => {
  useAuthStore.getState().startSession(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  // The newest 200 cycles hold none awaiting review; the server counts 3 older ones.
  vi.mocked(adminApi.fetchAdminPayouts).mockResolvedValue(
    Array.from({ length: 200 }, (_, i) => ({ ...payoutCycle, id: `new-${i}`, status: "paid" as const })),
  );
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.resetAllMocks(); });

it("reads awaiting review from the counts route, not from the newest 200 cycles", async () => {
  vi.mocked(adminApi.fetchAdminPayoutCounts).mockResolvedValue({ pending_review: 3, paid: 257 });
  mount();
  await vi.waitFor(() => expect(header()).toHaveTextContent("3 waiting"));
});

it("shows the money total only when it sums every payout awaiting review", async () => {
  const pending = (n: number) => Array.from({ length: n }, (_, i) => ({ ...payoutCycle, id: `p-${i}`, amount: 100 }));
  vi.mocked(adminApi.fetchAdminPayouts).mockResolvedValue(pending(200));
  vi.mocked(adminApi.fetchAdminPayoutCounts).mockResolvedValue({ pending_review: 260 });
  const view = mount();
  await vi.waitFor(() => expect(header()).toHaveTextContent("260 waiting"));
  // The read stops at 200 rows, so a sum would leave the 60 oldest out.
  expect(header()).not.toHaveTextContent("to approve");
  view.unmount(); client.clear();
  vi.mocked(adminApi.fetchAdminPayouts).mockResolvedValue(pending(3));
  vi.mocked(adminApi.fetchAdminPayoutCounts).mockResolvedValue({ pending_review: 3 });
  mount();
  await vi.waitFor(() => expect(header()).toHaveTextContent("₱300.00 to approve"));
});

it("shows no awaiting review count when the counts route is missing", async () => {
  vi.mocked(adminApi.fetchAdminPayoutCounts).mockRejectedValue(new ApiError([{ code: "NOT_FOUND", message: "Not found" }], 404));
  // Under 200 rows the read holds every payout awaiting review, so the total still shows.
  vi.mocked(adminApi.fetchAdminPayouts).mockResolvedValue([payoutCycle]);
  mount();
  await vi.waitFor(() => expect(adminApi.fetchAdminPayoutCounts).toHaveBeenCalled());
  await new Promise((done) => setTimeout(done, 20));
  expect(header()).not.toHaveTextContent(/\d waiting/);
  expect(header()).toHaveTextContent("to approve");
});

it("shows its guide intro, look-alike links and definitions without hover", () => {
  vi.mocked(adminApi.fetchAdminPayoutCounts).mockResolvedValue({});
  mount();
  expectSectionIntro(ROUTES.ADMIN_PAYOUTS);
});
