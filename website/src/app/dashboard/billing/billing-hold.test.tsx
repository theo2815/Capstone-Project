import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { PhotographerPayout } from "@/lib/photographer-mock";
import { useAuthStore } from "@/store/auth-store";
import BillingPage from "./page";

const HOLD_BODY = /you can't request a payout until the review is finished/i;

const balance = {
  unpaidBalance: 1000, clearing: 0, reserved: 0, minimum: 500,
  hasOpenRequest: false, openRequest: null as PhotographerPayout | null, accountCooldownUntil: null,
  primaryAccount: { method: "gcash", accountNumber: "09170001111", accountName: "Seller", hasQr: true },
  payoutHoldReason: null as string | null,
};
const openRequest: PhotographerPayout = {
  id: "cycle-1", weekOf: "2026-11-09", requestedAt: "2026-11-10T00:00:00Z", amount: 1500,
  status: "pending", settledAt: null, method: "gcash", reference: null, holdReason: null,
};

let client: QueryClient;
function setup(overrides: Partial<typeof balance>) {
  useAuthStore.getState().setUser({ id: "seller-1", role: "PHOTOGRAPHER", name: "Seller", email: "seller@example.test", createdAt: "2026-09-01" });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(api, "get").mockImplementation((path) => {
    if (path.includes("/payouts/balance")) return Promise.resolve({ ...balance, ...overrides });
    if (path.includes("/billing/transactions")) return Promise.resolve({ items: [], total: 0, offset: 0, limit: 25, monthTotals: {} });
    return Promise.resolve({ items: [], total: 0, offset: 0, limit: 8 });
  });
  render(<QueryClientProvider client={client}><BillingPage /></QueryClientProvider>);
}

afterEach(() => { cleanup(); client?.clear(); vi.restoreAllMocks(); useAuthStore.getState().logout(); });

it("a hold shows the notice and disables the request button", async () => {
  setup({ payoutHoldReason: "UPLOAD_REVIEW" });
  expect(await screen.findByText(HOLD_BODY)).toBeVisible();
  expect(screen.getByText("Payouts on hold")).toBeVisible();
  expect(screen.getByText("On hold while we review some of your uploads.")).toBeVisible();
  expect(screen.getByRole("button", { name: "Request payout" })).toBeDisabled();
});

it("a hold shows the notice beside an open request", async () => {
  setup({ payoutHoldReason: "UPLOAD_REVIEW", hasOpenRequest: true, openRequest });
  expect(await screen.findByText(HOLD_BODY)).toBeVisible();
  expect(screen.getByText("Pending review")).toBeVisible();
});

// Guard: passes before and after the change (R2-100).
it("no notice without a hold reason", async () => {
  setup({});
  expect(await screen.findByRole("button", { name: "Request payout" })).toBeEnabled();
  expect(screen.queryByText(HOLD_BODY)).not.toBeInTheDocument();
  expect(screen.queryByText("Payouts on hold")).not.toBeInTheDocument();
});
