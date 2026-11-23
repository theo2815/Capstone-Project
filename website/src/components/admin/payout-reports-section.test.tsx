import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { PayoutReport } from "@/lib/admin-payout-reports";
import { useAuthStore } from "@/store/auth-store";
import { admin, report } from "@/test/fixtures/admin";
import { PayoutReportsSection } from "./payout-reports-section";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
  usePathname: () => "/admin/payouts",
  useSearchParams: () => new URLSearchParams(),
}));

let client: QueryClient;
let reports: PayoutReport[];
let total: number;
beforeEach(() => {
  reports = [report];
  total = 1;
  useAuthStore.getState().setUser(admin);
  vi.spyOn(api, "get").mockImplementation(async (path: string) => {
    if (path === "/admin/kpis") return { openPayoutReports: 3 };
    if (path.startsWith("/admin/payouts/reports")) {
      const offset = Number(new URLSearchParams(path.split("?")[1]).get("offset"));
      return { items: reports, total, offset, limit: 50 };
    }
    return { items: [], total: 0, offset: 0, limit: 200 };
  });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => {
  cleanup();
  client.clear();
  useAuthStore.getState().logout();
  vi.restoreAllMocks();
});

function renderSection() {
  return render(<QueryClientProvider client={client}><PayoutReportsSection /></QueryClientProvider>);
}
const reportCalls = () => vi.mocked(api.get).mock.calls.map(([path]) => path).filter((p) => p.startsWith("/admin/payouts/reports"));

it("sends the status filter to the server", async () => {
  const user = userEvent.setup();
  renderSection();
  await waitFor(() => expect(reportCalls().some((p) => p.includes("status=open"))).toBe(true));
  await user.click(screen.getByRole("button", { name: /^Resolved/ }));
  await waitFor(() => expect(reportCalls().some((p) => p.includes("status=resolved"))).toBe(true));
});

it("shows the open count from the KPIs in the header", async () => {
  renderSection();
  expect(await screen.findByText("3 open")).toBeInTheDocument();
});

it("links the cycle id to its payout", async () => {
  renderSection();
  expect(await screen.findByRole("link", { name: "cycle-1" })).toHaveAttribute("href", "/admin/payouts?row=cycle-1");
});

it("makes the card's title its only button", async () => {
  renderSection();
  const item = (await screen.findByText("Photographer", { selector: "button" })).closest("li")!;
  expect(within(item).getAllByRole("button")).toHaveLength(1);
});

it("loads the next page from the server", async () => {
  const user = userEvent.setup();
  total = 2;
  renderSection();
  await screen.findByText("Photographer", { selector: "button" });
  reports = [{ ...report, id: "report-2", photographerName: "Second" }];
  await user.click(screen.getByRole("button", { name: "Load 1 more" }));
  expect(await screen.findByText("Second", { selector: "button" })).toBeInTheDocument();
  expect(reportCalls().some((p) => p.includes("offset=1"))).toBe(true);
});

it("is the payout-reports anchor the rail links to", async () => {
  const { container } = renderSection();
  await screen.findByText("Photographer", { selector: "button" });
  expect(container.querySelector("section#payout-reports")).not.toBeNull();
});

it("names each card's button with its report id", async () => {
  reports = [report, { ...report, id: "report-2" }];
  total = 2;
  renderSection();
  expect(await screen.findByRole("button", { name: /report-1/ })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /report-2/ })).toBeInTheDocument();
});

it("keeps a typed reply when a refresh drops the report from the list", async () => {
  const user = userEvent.setup();
  renderSection();
  await user.click(await screen.findByRole("button", { name: /report-1/ }));
  await user.click(screen.getByRole("button", { name: "Acknowledge…" }));
  const dialog = screen.getByRole("dialog", { name: /acknowledge · report-1/i });
  await user.type(within(dialog).getByRole("textbox"), "Checking with GCash.");
  reports = [];
  total = 0;
  await client.invalidateQueries({ queryKey: ["admin", "payouts", "reports"] });
  await waitFor(() => expect(screen.getByText("No open reports. You're caught up.")).toBeInTheDocument());
  // Query again: the node captured above would keep its value even after unmounting.
  expect(within(screen.getByRole("dialog", { name: /acknowledge · report-1/i })).getByRole("textbox")).toHaveValue("Checking with GCash.");
});
