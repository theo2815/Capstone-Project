import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { admin, adminUser, adminUserSettings } from "@/test/fixtures/admin";
import { adminUserKeys } from "@/hooks/use-admin-users";
import type { AdminUserRow } from "@/lib/admin-user-registry";
import { useAuthStore } from "@/store/auth-store";
import { VerificationsQueue } from "./verifications-queue";
import { AdminPhotographerCard } from "./admin-photographer-card";

const navigation = vi.hoisted(() => ({ search: "", replace: vi.fn() }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: navigation.replace }),
  usePathname: () => "/admin/verifications",
  useSearchParams: () => new URLSearchParams(navigation.search),
}));

let client: QueryClient;
const key = adminUserKeys.list(admin.id);
const second: AdminUserRow = { ...adminUser, userId: "photographer-2", brandName: "Second Studio", handle: "second-studio" };
const incomplete: AdminUserRow = { ...adminUser, userId: "photographer-3", brandName: "Half Studio", handle: "half-studio", verificationStatus: "incomplete" };
const suspendedIncomplete: AdminUserRow = { ...incomplete, userId: "photographer-4", brandName: "Banned Studio", handle: "banned", suspendedAt: "2026-11-01T00:00:00Z" };

function mount(rows?: AdminUserRow[]) {
  if (rows) client.setQueryData(key, rows);
  return render(<QueryClientProvider client={client}><VerificationsQueue /></QueryClientProvider>);
}

beforeEach(() => {
  useAuthStore.getState().setUser(admin); navigation.search = ""; navigation.replace.mockClear();
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); Reflect.deleteProperty(HTMLElement.prototype, "scrollIntoView"); });

it("the row opens through its title button, and the checkbox only selects", async () => {
  const { container } = mount([adminUser]);
  expect(container.querySelector('[role="button"]')).toBeNull();
  await userEvent.click(screen.getByRole("checkbox", { name: "Select Test Studio" }));
  expect(navigation.replace).not.toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Send back 1…" })).toBeInTheDocument();
  await userEvent.click(screen.getByRole("checkbox", { name: "Select Test Studio" }));
  await userEvent.click(screen.getByRole("button", { name: "Test Studio" }));
  expect(navigation.replace).toHaveBeenCalledWith(expect.stringContaining("row=photographer-1"), expect.anything());
});

it("FT-41: a slot renders between Incomplete and Decision history, which is then 04; without one it stays 03", () => {
  client.setQueryData(key, [adminUser]);
  const view = render(<QueryClientProvider client={client}><VerificationsQueue afterIncomplete={<section id="slot" />} /></QueryClientProvider>);
  const slot = document.getElementById("slot")!;
  const history = document.getElementById("recent-decisions")!;
  expect(document.getElementById("incomplete")!.compareDocumentPosition(slot) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(slot.compareDocumentPosition(history) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(within(history).getByText("04")).toBeInTheDocument();
  view.unmount();
  mount();
  expect(within(document.getElementById("recent-decisions")!).getByText("03")).toBeInTheDocument();
});

it("Incomplete leaves out suspended photographers and links each row to its photographer", () => {
  mount([incomplete, suspendedIncomplete]);
  const slab = document.getElementById("incomplete")!;
  expect(within(slab).queryByText("Banned Studio")).toBeNull();
  expect(within(slab).getByRole("link", { name: "Half Studio" })).toHaveAttribute("href", "/admin/photographers/half-studio");
});

it("Select all ticks every visible row, and bulk Reject is a danger button", async () => {
  mount([adminUser, second]);
  await userEvent.click(screen.getByRole("checkbox", { name: "Select all" }));
  const reject = screen.getByRole("button", { name: "Send back 2…" });
  expect(reject).toHaveClass("text-error");
  expect(screen.getByRole("button", { name: "Clear" })).toHaveClass("min-h-[44px]");
  await userEvent.click(screen.getByRole("checkbox", { name: "Select all" }));
  expect(screen.queryByRole("region", { name: "Verifications bulk action bar" })).toBeNull();
});

it("a failed first load (even with an empty message) shows an alert with Retry, and no zero counts", async () => {
  vi.spyOn(api, "get").mockRejectedValue(new Error(""));
  mount();
  const alert = await screen.findByRole("alert");
  expect(within(alert).getByText("Couldn't load the verifications.")).toBeInTheDocument();
  expect(within(alert).getByRole("button", { name: "Retry" })).toBeInTheDocument();
  expect(screen.queryByText("No profiles waiting. You're caught up.")).toBeNull();
  expect(screen.queryByText(/^0 (waiting|not finished)$/)).toBeNull();
  expect(screen.queryByText(/No photographers are part-way/)).toBeNull();
});

it("a row's completeness counts socials and payouts, the same number as the directory card", () => {
  const snapshot = { hasCover: true, hasBrandName: true, hasWatermark: true, hasHandle: true, hasRegion: true, socialCount: 0, payoutCount: 0 };
  mount([{ ...adminUser, settingsSnapshot: snapshot }]);
  expect(screen.getByText("5 of 7 fields")).toBeInTheDocument();
  cleanup();
  render(<AdminPhotographerCard row={{ ...adminUser, verificationStatus: "approved", settingsSnapshot: snapshot }} />);
  expect(screen.getByText("5/7 fields")).toBeInTheDocument();
});

// AX-39 (UA7): row values read as typed; mono caps stay for labels only.
it("an Incomplete row's values are not upper-cased", () => {
  mount([incomplete]);
  const slab = document.getElementById("incomplete")!;
  expect(within(slab).getByText(/@half-studio/).closest(".uppercase")).toBeNull();
});

// AX-57: drawer and detail values read as typed, wrap at 375, and link a no-handle photographer by id.
const notUpperCased = (els: HTMLElement[]) => { expect(els.length).toBeGreaterThan(0); for (const el of els) expect(el.closest(".uppercase")).toBeNull(); };
it("the drawer's subtitle, email and Joined are not upper-cased", async () => {
  navigation.search = "row=photographer-1";
  vi.spyOn(api, "get").mockImplementation((async (url: string) => (url.endsWith("/settings") ? adminUserSettings : url === "/admin/users/photographer-1" ? adminUser : [])) as typeof api.get);
  mount([adminUser]);
  const drawer = await screen.findByRole("dialog");
  notUpperCased(within(drawer).getAllByText(/Joined Sep 2026/));
  notUpperCased(await within(drawer).findAllByText("photographer@example.test"));
  await userEvent.click(within(drawer).getByRole("button", { name: /Who they are/ }));
  for (const el of within(drawer).getAllByText("photographer@example.test", { selector: "dd" })) expect(el).toHaveClass("break-words");
  for (const el of within(drawer).getAllByText("Test Studio", { selector: "p" })) expect(el).not.toHaveClass("truncate");
  await userEvent.click(within(drawer).getByRole("button", { name: /Account/ }));
  notUpperCased(await within(drawer).findAllByText(/^Sep 2026$/));
});

it("an Incomplete photographer with no handle links to their page by id", () => {
  mount([{ ...incomplete, userId: "photographer-5", brandName: "No Handle Studio", handle: null }]);
  const slab = document.getElementById("incomplete")!;
  expect(within(slab).getByRole("link", { name: "No Handle Studio" })).toHaveAttribute("href", "/admin/photographers/photographer-5");
});

it("an empty queue says nothing is waiting in one line, and the part-way list says there is nothing to do", () => {
  mount([]);
  expect(screen.getByText("No profiles waiting. You're caught up.")).toBeInTheDocument();
  expect(screen.queryByText(/Check back/)).toBeNull();
  expect(screen.getByText("Waiting for the ID check")).toBeInTheDocument();
  expect(screen.getByText("0 waiting")).toBeInTheDocument();
  expect(screen.getByText("Still filling in their settings. Nothing to do here.")).toBeInTheDocument();
  expect(screen.getByText("0 not finished")).toBeInTheDocument();
});

// FT-13 (C-02, C-06, C-07, C-08): plain words on the row, the drawer and the Incomplete slab.
it("FT-13: the row names its links and accounts, lists what is missing, and has a 44 px chevron", () => {
  const snapshot = { hasCover: false, hasBrandName: true, hasWatermark: false, hasHandle: true, hasRegion: true, socialCount: 1, payoutCount: 2 };
  mount([{ ...adminUser, settingsSnapshot: snapshot }]);
  const row = screen.getByRole("button", { name: "Test Studio" }).closest("article")!;
  expect(row).toHaveTextContent("· 1 social link · 2 payout accounts");
  expect(row).toHaveTextContent("Missing: cover banner, watermark");
  expect(within(row).getByRole("button", { name: "Open review for Test Studio" })).toHaveClass("size-11");
});
it("FT-13: the Incomplete slab says they finish it themselves", () => {
  mount([incomplete]);
  expect(within(document.getElementById("incomplete")!).getByText("They finish this themselves")).toBeInTheDocument();
});
it("FT-13: the drawer's first section is Who they are, and says where ID photos are checked", async () => {
  navigation.search = "row=photographer-1";
  vi.spyOn(api, "get").mockImplementation((async (url: string) => (url.endsWith("/settings") ? adminUserSettings : url === "/admin/users/photographer-1" ? adminUser : [])) as typeof api.get);
  mount([adminUser]);
  const drawer = await screen.findByRole("dialog");
  await userEvent.click(await within(drawer).findByRole("button", { name: /Who they are/ }));
  const link = within(drawer).getByRole("link", { name: "Seller details" });
  expect(link.closest("p")).toHaveTextContent("ID photos are checked under Seller details.");
});
