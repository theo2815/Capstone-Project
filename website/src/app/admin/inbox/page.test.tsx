import { cleanup, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, expect, it, vi } from "vitest";
import type { AdminKpis } from "@/lib/api-admin";

const replace = vi.hoisted(() => vi.fn());
const url = vi.hoisted(() => ({ search: "" }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace, push: vi.fn() }),
  usePathname: () => "/admin/inbox",
  useSearchParams: () => new URLSearchParams(url.search),
}));

// The KPI query at the API boundary: pending (no data), failed, or settled with counts.
const kpis = vi.hoisted(() => ({ data: undefined as Partial<AdminKpis> | undefined, isPending: true, isError: false }));
vi.mock("@/hooks/use-admin-data", () => ({
  useAdminKpisQuery: () => ({ data: kpis.data, isPending: kpis.isPending, isError: kpis.isError }),
  useAdminKpis: () => kpis.data ?? null,
}));

const mounts = vi.hoisted(() => ({ verifications: 0 }));
vi.mock("@/components/admin/verifications-queue", () => ({
  VerificationsQueue: () => {
    useEffect(() => { mounts.verifications += 1; }, []);
    return <p>queue:verifications</p>;
  },
}));
vi.mock("@/components/admin/disputes-queue", () => ({ DisputesQueue: () => <p>queue:disputes</p> }));
vi.mock("@/components/admin/event-requests-queue", () => ({ EventRequestsQueue: () => <p>queue:events</p> }));
vi.mock("@/components/admin/flags-queue", () => ({ FlagsQueue: () => <p>queue:flags</p> }));
vi.mock("@/components/admin/payouts-queue", () => ({ PayoutsQueue: () => <p>queue:payouts</p> }));
vi.mock("@/components/admin/payout-reports-section", () => ({ PayoutReportsSection: () => <p>queue:payout-reports</p> }));
vi.mock("@/components/admin/seller-details-queue", () => ({ SellerDetailsQueue: () => <p>queue:seller-details</p> }));
vi.mock("@/components/admin/photo-removal/request-list", () => ({ RemovalRequestList: () => <p>queue:removals</p> }));
vi.mock("@/components/admin/relevance/relevance-cases", () => ({ RelevanceCases: () => <p>queue:relevance</p> }));

const ZERO: Partial<AdminKpis> = {
  pendingVerifications: 0, openDisputes: 0, disputesNeedingAdmin: 0, openFlags: 0, pendingEventRequests: 0,
  pendingSellerDetails: 0, payoutsAwaitingReview: 0, openPayoutReports: 0, relevanceNeedsPerson: 0, removalsNeedingAction: 0,
};

function settle(counts: Partial<AdminKpis>) {
  kpis.data = { ...ZERO, ...counts };
  kpis.isPending = false;
}

async function renderPage() {
  const { default: AdminInboxPage } = await import("./page");
  return render(<AdminInboxPage />);
}

afterEach(() => {
  cleanup();
  replace.mockReset();
  url.search = "";
  kpis.data = undefined;
  kpis.isPending = true;
  kpis.isError = false;
  mounts.verifications = 0;
  vi.doUnmock("@/lib/constants");
  vi.resetModules();
});

it("no ?type, 0 verifications and 2 disputes: opens the Disputes queue", async () => {
  settle({ pendingVerifications: 0, disputesNeedingAdmin: 2 });
  await renderPage();
  expect(replace).toHaveBeenCalledWith("/admin/inbox?type=disputes", { scroll: false });
});

it("no ?type and nothing waiting: opens Verifications", async () => {
  settle({});
  await renderPage();
  expect(replace).toHaveBeenCalledWith("/admin/inbox?type=verifications", { scroll: false });
});

it("no ?type and the KPI query failed: opens Verifications", async () => {
  kpis.isPending = false;
  kpis.isError = true;
  await renderPage();
  expect(replace).toHaveBeenCalledWith("/admin/inbox?type=verifications", { scroll: false });
});

it("no ?type while the KPIs load: mounts no queue and waits", async () => {
  await renderPage();
  expect(screen.queryByText(/^queue:/)).toBeNull();
  expect(replace).not.toHaveBeenCalled();
});

it("?type present: stays on that queue whatever the counts say", async () => {
  url.search = "type=payouts";
  settle({ disputesNeedingAdmin: 2 });
  await renderPage();
  expect(screen.getByText("queue:payouts")).toBeTruthy();
  expect(replace).not.toHaveBeenCalled();
});

it("attention skips a gated-off queue: with Flags off, a flags count alone opens Verifications", async () => {
  vi.doMock("@/lib/constants", async (importOriginal) => ({
    ...(await importOriginal<typeof import("@/lib/constants")>()),
    ADMIN_FLAGS_ENABLED: false,
  }));
  settle({ openFlags: 3 });
  await renderPage();
  expect(replace).toHaveBeenCalledWith("/admin/inbox?type=verifications", { scroll: false });
});

it("when the URL gains ?type=disputes, the Verifications queue is never mounted on the way", async () => {
  settle({ disputesNeedingAdmin: 2 });
  const { default: AdminInboxPage } = await import("./page");
  const { rerender } = render(<AdminInboxPage />);
  url.search = "type=disputes";
  rerender(<AdminInboxPage />);
  expect(screen.getByText("queue:disputes")).toBeTruthy();
  expect(mounts.verifications).toBe(0);
});

it("the heading is the rail label", async () => {
  settle({});
  await renderPage();
  expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Inbox.");
});

it("says what the Inbox is at every width, and what the open queue is for", async () => {
  url.search = "type=payouts";
  settle({});
  await renderPage();
  const intro = screen.getByText(/^Everything that is waiting for an admin, one queue at a time\./);
  expect(intro.className).not.toMatch(/(^|\s)hidden(\s|$)/);
  expect(
    screen.getByText("Photographers asking to be paid. Approving sends no money: you send it yourself, then mark the payout paid."),
  ).toBeTruthy();
});

it("the kicker counts what is waiting, the same total as the rail", async () => {
  url.search = "type=disputes";
  settle({ disputesNeedingAdmin: 2 });
  await renderPage();
  expect(screen.getByRole("banner").querySelector("p")).toHaveTextContent(/^Inbox · 2 waiting$/);
});

it("shows the counts once: no separate count strip above the chips", async () => {
  url.search = "type=disputes";
  settle({ disputesNeedingAdmin: 2 });
  await renderPage();
  expect(screen.queryByRole("navigation", { name: "Queue counts" })).toBeNull();
});

it("the kicker reads All clear only on a real 0, and the label alone while counts are unknown", async () => {
  url.search = "type=verifications";
  const { unmount } = await renderPage();
  expect(screen.getByRole("banner").querySelector("p")).toHaveTextContent(/^Inbox$/);
  unmount();
  settle({});
  await renderPage();
  expect(screen.getByRole("banner").querySelector("p")).toHaveTextContent(/^Inbox · All clear$/);
});
