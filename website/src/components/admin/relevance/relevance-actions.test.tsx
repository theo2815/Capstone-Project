import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import type { RelevanceCaseDetail, RelevanceCaseRow, RelevanceHiddenPhoto } from "@/lib/admin-relevance";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { RelevanceCases } from "./relevance-cases";

// RV-2.6: the actions on a case. The row and the drawer change only after the server answers.
const fixtures = vi.hoisted(() => ({ search: "", router: { replace: vi.fn() } }));
vi.mock("next/navigation", () => ({
  useRouter: () => fixtures.router,
  usePathname: () => "/admin/relevance",
  useSearchParams: () => new URLSearchParams(fixtures.search),
}));

const autoHide: RelevanceCaseRow = {
  caseId: "c1", eventId: "e1", eventName: "Cebu City Marathon 2003", photographerHandle: "juanshots",
  photographerName: "Juan Cruz", verdict: "JUNK_DUMP", verdictNote: null, action: "HIDE_ALL", appliedMode: "AUTO_HIDE",
  wouldHave: false, batchPhotos: 420, junkPercent: 81, decidedAt: "2003-02-14T03:00:00Z",
};
const cantTell: RelevanceCaseRow = { ...autoHide, verdict: "CANT_TELL", action: "NONE", junkPercent: 40 };
const detail = (r: RelevanceCaseRow, stage = "APPLIED", extra: Partial<RelevanceCaseDetail> = {}): RelevanceCaseDetail => ({
  case: r, stage, sampleSize: 300, measured: 300, blurry: 100, dark: 10, notRace: 10, junk: 120, band: "MIXED",
  modelVerdict: r.verdict, adminReason: null, evidenceUsed: [], summarySent: null, model: null, promptVersion: null,
  wordsPhotos: 0, topWords: [], junkSamples: [], cleanSamples: [], alerts: [],
  confirmedAt: null, photographerStrikes: 0, payoutHoldAt: null, closeReason: null,
  ...extra,
});
const hold = (id: string, restoredAt: string | null = null): RelevanceHiddenPhoto => ({
  photoId: id, heldBy: "AUTO", hiddenAt: "2003-02-15T00:00:00Z", restoredAt, restoredBy: restoredAt ? "ADMIN" : null,
  status: restoredAt ? "LIVE" : "HIDDEN", thumbnailUrl: null,
});

let rows: RelevanceCaseRow[];
let one: () => Promise<RelevanceCaseDetail>;
let holds: () => Promise<{ items: RelevanceHiddenPhoto[]; total: number; offset: number; limit: number }>;
let post: (url: string) => Promise<unknown>;
let client: QueryClient;
beforeEach(() => {
  fixtures.search = "tab=cant_tell&row=c1";
  useAuthStore.getState().setUser(admin);
  useToastStore.getState().clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  rows = [cantTell];
  one = async () => detail(rows[0] ?? cantTell);
  holds = async () => ({ items: [], total: 0, offset: 0, limit: 100 });
  post = async () => ({});
  vi.spyOn(api, "get").mockImplementation(((url: string) =>
    url.startsWith("/admin/relevance/cases?") ? Promise.resolve({ items: rows, total: rows.length, offset: 0, limit: 50 })
      : url.includes("/hidden-photos") ? holds() : one()) as typeof api.get);
  vi.spyOn(api, "post").mockImplementation(((url: string) => post(url)) as typeof api.post);
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });
const mount = () => render(<QueryClientProvider client={client}><RelevanceCases /></QueryClientProvider>);
const toasts = () => useToastStore.getState().toasts.map((t) => t.message);
const drawer = () => screen.findByRole("dialog", { name: "Cebu City Marathon 2003" });
const buttons = (el: HTMLElement) => within(el).getAllByRole("button").map((b) => b.getAttribute("aria-label") ?? b.textContent);

it("an auto-hide case shows Restore and Keep hidden", async () => {
  const user = userEvent.setup();
  rows = [autoHide];
  holds = async () => ({ items: [hold("p1")], total: 1, offset: 0, limit: 100 });
  mount();
  const d = await drawer();
  await within(d).findByRole("button", { name: "Restore batch…" });
  expect(within(d).getByRole("button", { name: "Keep hidden…" })).toBeInTheDocument();
  expect(within(d).queryByRole("button", { name: "Looks fine…" })).not.toBeInTheDocument();
  expect(within(d).queryByRole("button", { name: "Hide batch…" })).not.toBeInTheDocument();
  await user.click(within(d).getByRole("button", { name: "Keep hidden…" }));
  await user.click(screen.getByRole("button", { name: "Confirm keep hidden" }));
  await waitFor(() => expect(api.post).toHaveBeenCalledWith("/admin/relevance/cases/c1/keep-hidden"));
  // A record-only case with no hold offers nothing; with a seeded hold it offers the restores only.
  cleanup();
  client.clear();
  rows = [{ ...autoHide, appliedMode: "RECORD_ONLY", wouldHave: true }];
  holds = async () => ({ items: [], total: 0, offset: 0, limit: 100 });
  mount();
  const ro = await drawer();
  await within(ro).findByText(/Record-only: nothing was hidden/);
  await within(ro).findByText("No photo of this batch is held.");
  expect(buttons(ro)).toEqual(["Close"]);
});

it("a can't tell case shows Looks fine and Hide batch", async () => {
  mount();
  const d = await drawer();
  await within(d).findByRole("button", { name: "Looks fine…" });
  expect(within(d).getByRole("button", { name: "Hide batch…" })).toBeInTheDocument();
  expect(within(d).queryByRole("button", { name: "Restore batch…" })).not.toBeInTheDocument();
  // A Can't tell case an admin already hid offers the restores instead.
  cleanup();
  client.clear();
  rows = [{ ...cantTell, action: "HIDE_ALL", verdictNote: "ADMIN_HIDE" }];
  mount();
  const hidden = await drawer();
  await within(hidden).findByRole("button", { name: "Restore batch…" });
  expect(within(hidden).queryByRole("button", { name: "Looks fine…" })).not.toBeInTheDocument();
  expect(within(hidden).queryByRole("button", { name: "Hide batch…" })).not.toBeInTheDocument();
});

it("each action waits for the server before the row changes", async () => {
  const user = userEvent.setup();
  let release!: (v: unknown) => void;
  post = () => new Promise((r) => { release = r; });
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Looks fine…" }));
  await user.click(screen.getByRole("button", { name: "Confirm looks fine" }));
  expect(api.post).toHaveBeenCalledWith("/admin/relevance/cases/c1/looks-fine");
  // The server has not answered: the row is still listed and the case still offers its actions.
  expect(screen.getAllByText("Cebu City Marathon 2003").length).toBeGreaterThan(1);
  expect(within(d).getByRole("button", { name: "Hide batch…" })).toBeInTheDocument();
  expect(toasts()).toEqual([]);
  rows = [];
  one = async () => detail({ ...cantTell }, "CLOSED");
  release(cantTell);
  await waitFor(() => expect(within(d).queryByRole("button", { name: "Hide batch…" })).not.toBeInTheDocument());
  expect(toasts()).toEqual(["Marked looks fine: Cebu City Marathon 2003"]);
  expect(await screen.findByText("No batch is waiting on a person to judge it.")).toBeInTheDocument();
});

it("a refused action shows the server's message", async () => {
  const user = userEvent.setup();
  post = async () => { throw new ApiError([{ code: "RELEVANCE_ACTIONS_OFF", message: "Hiding a batch needs the relevance mode auto-hide" }], 409); };
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Hide batch…" }));
  await user.click(screen.getByRole("button", { name: "Confirm hide batch" }));
  await waitFor(() => expect(toasts()).toEqual(["Hiding a batch needs the relevance mode auto-hide"]));
  expect(api.post).toHaveBeenCalledWith("/admin/relevance/cases/c1/hide-batch");
  // Nothing changed: the confirm stays open for another try and the case keeps its actions.
  expect(screen.getByRole("button", { name: "Confirm hide batch" })).toBeEnabled();
  expect(within(d).getByRole("button", { name: "Looks fine…" })).toBeInTheDocument();
});

it("restoring one photo marks it restored in the list", async () => {
  const user = userEvent.setup();
  rows = [autoHide];
  let restored = false;
  holds = async () => ({ items: [hold("p1", restored ? "2003-02-16T00:00:00Z" : null), hold("p2")], total: 3, offset: 0, limit: 100 });
  post = async () => { restored = true; return { photoId: "p1", status: "LIVE" }; };
  mount();
  const d = await drawer();
  const list = await within(d).findByRole("list", { name: "Held photos" });
  expect(within(d).getByText(/Showing 2 of 3/)).toBeInTheDocument();
  expect(within(list).getAllByRole("button", { name: /^Restore photo/ })).toHaveLength(2);
  await user.click(within(list).getAllByRole("button", { name: /^Restore photo/ })[0]);
  await waitFor(() => expect(within(list).getAllByRole("button", { name: /^Restore photo/ })).toHaveLength(1));
  expect(api.post).toHaveBeenCalledWith("/admin/relevance/cases/c1/photos/p1/restore");
  expect(within(list).getByText(/^Restored/)).toBeInTheDocument();
  expect(toasts()).toEqual(["Photo restored"]);
});

// Review fixes on PR #154.
it("an admin-hidden case reads its note as words", async () => {
  rows = [{ ...cantTell, action: "HIDE_ALL", verdictNote: "ADMIN_HIDE" }];
  mount();
  const d = await drawer();
  expect(await within(d).findByText("An admin hid this batch from this page.")).toBeInTheDocument();
  expect(within(d).queryByText("ADMIN_HIDE")).not.toBeInTheDocument();
});

it("a case that is no longer current offers no batch restore", async () => {
  rows = [];
  one = async () => detail(autoHide, "CLOSED");
  holds = async () => ({ items: [hold("p1")], total: 1, offset: 0, limit: 100 });
  mount();
  const d = await drawer();
  await within(d).findByRole("list", { name: "Held photos" });
  expect(buttons(d)).toEqual(["Close"]);
});

it("a record-only case with a hold offers only the restores", async () => {
  rows = [{ ...autoHide, appliedMode: "RECORD_ONLY", wouldHave: true }];
  holds = async () => ({ items: [hold("p1")], total: 1, offset: 0, limit: 100 });
  mount();
  const d = await drawer();
  await within(d).findByRole("button", { name: "Restore batch…" });
  expect(buttons(d).filter((b) => b !== "Close")).toEqual([expect.stringMatching(/^Restore photo/), "Restore batch…"]);
});

it("each photo Restore has its own label", async () => {
  rows = [autoHide];
  holds = async () => ({ items: [hold("p1"), hold("p2")], total: 2, offset: 0, limit: 100 });
  mount();
  const list = await within(await drawer()).findByRole("list", { name: "Held photos" });
  const labels = within(list).getAllByRole("button").map((b) => b.getAttribute("aria-label"));
  expect(new Set(labels).size).toBe(2);
});

it("a refused action re-reads the case", async () => {
  const user = userEvent.setup();
  post = async () => { throw new ApiError([{ code: "INVALID_STATE_TRANSITION", message: "This action does not fit the case's current state" }], 409); };
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Looks fine…" }));
  // Another admin closed the case meanwhile: the refusal shows, and the drawer reads the case again.
  one = async () => detail(cantTell, "CLOSED");
  await user.click(screen.getByRole("button", { name: "Confirm looks fine" }));
  await waitFor(() => expect(within(d).queryByRole("button", { name: "Hide batch…" })).not.toBeInTheDocument());
  expect(toasts()).toEqual(["This action does not fit the case's current state"]);
});

it("a confirm closes when a re-read offers a different set of actions", async () => {
  const user = userEvent.setup();
  post = async () => { throw new ApiError([{ code: "INVALID_STATE_TRANSITION", message: "This action does not fit the case's current state" }], 409); };
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Looks fine…" }));
  // Another admin hid the batch meanwhile: the case now offers the restores, so the Looks fine confirm must go.
  one = async () => detail({ ...cantTell, action: "HIDE_ALL", verdictNote: "ADMIN_HIDE" });
  await user.click(screen.getByRole("button", { name: "Confirm looks fine" }));
  await within(d).findByRole("button", { name: "Restore batch…" });
  expect(screen.queryByRole("button", { name: "Confirm looks fine" })).not.toBeInTheDocument();
  expect(toasts()).toEqual(["This action does not fit the case's current state"]);
});

it("Escape closes the drawer after a re-read removes the open action", async () => {
  const user = userEvent.setup();
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Looks fine…" }));
  expect(screen.getByRole("button", { name: "Confirm looks fine" })).toBeInTheDocument();
  one = async () => detail(cantTell, "CLOSED");
  await client.invalidateQueries({ queryKey: ["admin", "relevance"] });
  await waitFor(() => expect(screen.queryByRole("button", { name: "Confirm looks fine" })).not.toBeInTheDocument());
  await user.keyboard("{Escape}");
  expect(fixtures.router.replace).toHaveBeenLastCalledWith("/admin/relevance?tab=cant_tell", { scroll: false });
});

// RV-4.3: the consequences. A confirmed case shows its date and the count; End hold waits for the server.
const copied: RelevanceCaseRow = { ...autoHide, verdict: "COPIED" };
const held = (extra: Partial<RelevanceCaseDetail> = {}) =>
  detail(copied, "APPLIED", { confirmedAt: "2003-03-01T03:00:00Z", photographerStrikes: 1, payoutHoldAt: "2003-02-14T04:00:00Z", ...extra });
const field = (el: HTMLElement, label: string) => within(el).getByText(label).nextElementSibling?.textContent;

it("a confirmed case shows its date and the count", async () => {
  fixtures.search = "tab=all&row=c1";
  rows = [copied];
  one = async () => held({ payoutHoldAt: null });
  mount();
  const d = await drawer();
  await within(d).findByText("Confirmed on");
  expect(field(d, "Confirmed on")).toBe("MAR 01 · 2003");
  expect(field(d, "Cases in the last 6 months")).toBe("1");
  // No live hold: no hold date and no End hold.
  expect(within(d).queryByText("Payouts on hold since")).not.toBeInTheDocument();
  expect(within(d).queryByRole("button", { name: "End hold" })).not.toBeInTheDocument();
  // A case that is not confirmed shows none of it.
  cleanup();
  client.clear();
  one = async () => detail(copied);
  mount();
  const plain = await drawer();
  await within(plain).findByRole("button", { name: "Restore batch…" });
  expect(within(plain).queryByText("Confirmed on")).not.toBeInTheDocument();
  expect(within(plain).queryByText("Cases in the last 6 months")).not.toBeInTheDocument();
});

it("End hold waits for the server", async () => {
  const user = userEvent.setup();
  fixtures.search = "tab=all&row=c1";
  rows = [copied];
  one = async () => held();
  let release!: (v: unknown) => void;
  post = () => new Promise((r) => { release = r; });
  mount();
  const d = await drawer();
  await within(d).findByText("Payouts on hold since");
  expect(field(d, "Payouts on hold since")).toBe("FEB 14 · 2003");
  // Beside the button: one hold per Copied case, payouts resume when the last ends.
  expect(within(d).getByText("One photographer can have several holds, one per Copied case. Payouts resume only when the last one ends.")).toBeInTheDocument();
  await user.click(within(d).getByRole("button", { name: "End hold" }));
  expect(api.post).not.toHaveBeenCalled();
  // AdminFlagService tells the photographer only once no other Photo Relevance hold is open.
  expect(screen.getByText("Ends the payout hold this case placed. The case itself stays as it is. The photographer is told their payout hold was lifted only when no other Photo Relevance hold is left.")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Confirm end hold" }));
  expect(api.post).toHaveBeenCalledWith("/admin/relevance/cases/c1/end-hold");
  // The server has not answered: the hold still shows and nothing was said.
  expect(field(d, "Payouts on hold since")).toBe("FEB 14 · 2003");
  expect(within(d).getByRole("button", { name: "End hold" })).toBeInTheDocument();
  expect(toasts()).toEqual([]);
  one = async () => held({ payoutHoldAt: null });
  release(copied);
  await waitFor(() => expect(within(d).queryByText("Payouts on hold since")).not.toBeInTheDocument());
  expect(within(d).queryByRole("button", { name: "End hold" })).not.toBeInTheDocument();
  expect(toasts()).toEqual(["Hold ended: Cebu City Marathon 2003"]);
});

it("two End hold confirms in one tick send one request", async () => {
  const user = userEvent.setup();
  fixtures.search = "tab=all&row=c1";
  rows = [copied];
  one = async () => held();
  post = () => new Promise(() => {});
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "End hold" }));
  const ok = await screen.findByRole("button", { name: "Confirm end hold" });
  act(() => { ok.click(); ok.click(); });
  await act(async () => {});
  expect(vi.mocked(api.post).mock.calls).toHaveLength(1);
});

it("a refused End hold shows the server's message", async () => {
  const user = userEvent.setup();
  fixtures.search = "tab=all&row=c1";
  rows = [copied];
  one = async () => held();
  const message = "The admin flags switch is off, so this case's payout hold cannot be ended. Turn ADMIN_FLAGS_ENABLED on first.";
  post = async () => { throw new ApiError([{ code: "FLAGS_DISABLED", message }], 403); };
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "End hold" }));
  await user.click(screen.getByRole("button", { name: "Confirm end hold" }));
  await waitFor(() => expect(toasts()).toEqual([message]));
  // Nothing changed: the confirm stays open and the hold still shows.
  expect(screen.getByRole("button", { name: "Confirm end hold" })).toBeEnabled();
  expect(field(d, "Payouts on hold since")).toBe("FEB 14 · 2003");
});

it("a third-case alert links to the photographer", async () => {
  fixtures.search = "tab=all&row=c1";
  rows = [copied];
  const alert = { alertId: "a1", caseId: "c1", eventId: "e1", eventName: "Cebu City Marathon 2003", photographerHandle: "juanshots",
    photographerName: "Juan Cruz", wouldHave: false, photos: null, createdAt: "2003-03-01T03:00:00Z", seenAt: null };
  one = async () => held({ alerts: [{ ...alert, kind: "THIRD_CASE" }, { ...alert, alertId: "a2", kind: "COPIED" }] });
  mount();
  const d = await drawer();
  const link = await within(d).findByRole("link", { name: "Open photographer →" });
  expect(link).toHaveAttribute("href", "/admin/photographers/juanshots");
  // Only the third-case alert links there.
  expect(within(d).getAllByRole("link", { name: "Open photographer →" })).toHaveLength(1);
});

it("a case closed as restored by an admin that still holds photos offers Restore", async () => {
  const user = userEvent.setup();
  fixtures.search = "tab=all&row=c1";
  rows = [];
  one = async () => detail(autoHide, "CLOSED", { closeReason: "RESTORED_BY_ADMIN" });
  holds = async () => ({ items: [hold("p1")], total: 1, offset: 0, limit: 100 });
  post = async () => { throw new Error("down"); };
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Restore batch…" }));
  // AdminRelevanceService.restore also ends a Copied case's payout hold.
  expect(screen.getByText("Puts back on sale every photo this case hid in the batch. The case closes as restored by an admin and does not count. A Copied case's payout hold ends too. Photos the photographer hid stay hidden.")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Confirm restore" }));
  expect(api.post).toHaveBeenCalledWith("/admin/relevance/cases/c1/restore");
  await waitFor(() => expect(toasts()).toEqual(["Could not restore the batch. The case was read again: if photos are still held, Restore batch finishes the rest."]));
  // Closed for another reason, or with nothing left held: no Restore.
  for (const [reason, items] of [["SUPERSEDED", [hold("p1")]], ["RESTORED_BY_ADMIN", [hold("p1", "2003-02-16T00:00:00Z")]]] as const) {
    cleanup();
    client.clear();
    one = async () => detail(autoHide, "CLOSED", { closeReason: reason });
    holds = async () => ({ items: [...items], total: 1, offset: 0, limit: 100 });
    mount();
    const closed = await drawer();
    await within(closed).findByRole("list", { name: "Held photos" });
    expect(buttons(closed)).toEqual(["Close"]);
  }
});

it("a failed End hold with no server message says what to do next", async () => {
  const user = userEvent.setup();
  fixtures.search = "tab=all&row=c1";
  rows = [copied];
  one = async () => held();
  post = async () => { throw new Error("down"); };
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "End hold" }));
  await user.click(screen.getByRole("button", { name: "Confirm end hold" }));
  await waitFor(() => expect(toasts()).toEqual(["Could not end the hold. The case was read again: if the payout hold still shows, End hold tries again."]));
});
