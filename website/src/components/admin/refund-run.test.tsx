import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { Dispute } from "@/lib/admin-disputes";
import { useAuthStore } from "@/store/auth-store";
import { adminDisputeKeys } from "@/hooks/use-admin-disputes";
import { admin, dispute } from "@/test/fixtures/admin";
import { DisputesQueue } from "./disputes-queue";
import { RefundRun } from "./refund-run";

const fixtures = vi.hoisted(() => ({ search: "", router: { replace: vi.fn() } }));
vi.mock("next/navigation", () => ({
  useRouter: () => fixtures.router,
  usePathname: () => "/admin/disputes",
  useSearchParams: () => new URLSearchParams(fixtures.search),
}));

const parked = (n: number, extra: Partial<Dispute> = {}): Dispute => ({
  ...dispute,
  id: `dispute-${n}`,
  runnerHandle: `runner-${n}`,
  eventName: `Event ${n}`,
  status: "escalated",
  resolution: "refund_full",
  refundAmount: 100,
  refundStatus: "manual_review",
  paymentId: `pay_${n}`,
  attention: "send_link",
  orderSnapshot: { ...dispute.orderSnapshot, paymentMethod: "qrph" },
  ...extra,
});
const page = (items: Dispute[]) => ({ items, total: items.length, offset: 0, limit: 50 });
const key = adminDisputeKeys.page(admin.id, { view: "refund_run", offset: 0 });

let client: QueryClient;
function seed(items: Dispute[]) {
  vi.spyOn(api, "get").mockResolvedValue(page(items));
  client.setQueryData(key, page(items));
}
beforeEach(() => {
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  fixtures.search = "";
  fixtures.router.replace.mockClear();
  useAuthStore.getState().setUser(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => {
  cleanup();
  client.clear();
  useAuthStore.getState().logout();
  Reflect.deleteProperty(HTMLElement.prototype, "scrollIntoView");
  vi.restoreAllMocks();
});

const wrap = (ui: React.ReactNode) => render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);

it("lists every refund the server returns for refund_run, each with its refund panel", async () => {
  seed([parked(1), parked(2)]);
  wrap(<RefundRun onOpen={vi.fn()} />);
  expect(await screen.findByText("@runner-1")).toBeInTheDocument();
  expect(screen.getByText("@runner-2")).toBeInTheDocument();
  expect(screen.getAllByRole("button", { name: "Mark refunded" })).toHaveLength(2);
  expect(screen.getByText("pay_1")).toBeInTheDocument();
  expect(screen.getByText("pay_2")).toBeInTheDocument();
  expect(api.get).not.toHaveBeenCalled();
});

it("a refund whose link expires soon says so", async () => {
  seed([parked(1, { linkExpiringSoon: true, refundLinkExpiresAt: "2099-01-01T00:00:00Z", refundLinkUrl: "https://pm.link/r1", refundLinkSentAt: "2026-11-01T00:00:00Z" })]);
  wrap(<RefundRun onOpen={vi.fn()} />);
  expect(await screen.findByText(/Link expires/)).toBeInTheDocument();
});

it("a check-PayMongo refund shows Cancel refund", async () => {
  seed([parked(1, {
    attention: "check_paymongo",
    refundLinkUrl: "https://pm.link/r1",
    refundLinkSentAt: "2026-10-01T00:00:00Z",
    refundWindowClosesAt: "2026-10-20T00:00:00Z",
    refundCancelsAt: "2026-10-25T00:00:00Z",
  })]);
  wrap(<RefundRun onOpen={vi.fn()} />);
  expect(await screen.findByRole("button", { name: "Cancel refund…" })).toBeInTheDocument();
});

it("Open opens the drawer and keeps the refund run view", async () => {
  const user = userEvent.setup();
  fixtures.search = "view=refund_run";
  seed([parked(1)]);
  wrap(<DisputesQueue />);
  const header = (await screen.findByText("@runner-1")).closest("article") as HTMLElement;
  await user.click(within(header).getByRole("button", { name: "Open dispute dispute-1" }));
  const url = fixtures.router.replace.mock.calls.at(-1)?.[0] as string;
  expect(url).toContain("view=refund_run");
  expect(url).toContain("row=dispute-1");
});
