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
  confirmedAt: null, photographerStrikes: 0, payoutHoldAt: null, payoutHoldPlaced: false, closeReason: null,
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
// The Technical details toggle (FT-20) only folds text; it is not an action on the case.
const buttons = (el: HTMLElement) => within(el).getAllByRole("button").map((b) => b.getAttribute("aria-label") ?? b.textContent)
  .filter((b) => !b?.startsWith("Technical details"));

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
  const invalidate = vi.spyOn(client, "invalidateQueries");
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
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "kpis"] });
  expect(useToastStore.getState().toasts.map((t) => t.kind)).toEqual(["success"]);
  expect(await screen.findByText("No batch is waiting for you. You're caught up.")).toBeInTheDocument();
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
  expect(useToastStore.getState().toasts.map((t) => t.kind)).toEqual(["success"]);
});

// Review fixes on PR #154.
it("an admin-hidden case reads its note as words", async () => {
  rows = [{ ...cantTell, action: "HIDE_ALL", verdictNote: "ADMIN_HIDE" }];
  mount();
  const d = await drawer();
  expect(await within(d).findByText("An admin hid this batch from this page.")).toBeInTheDocument();
  expect(within(d).queryByText("ADMIN_HIDE")).not.toBeInTheDocument();
});

// RVF-19: the junk a won appeal left held can be released; the server refuses while a newer case is current.
const CLOSED_RESTORE = "Puts back on sale the photos of this batch that are still held. The case stays closed as it is, and no payout hold changes. Photos the photographer hid stay hidden.";
it("a closed case with a releasable hold can be restored", async () => {
  const user = userEvent.setup();
  rows = [];
  one = async () => detail(autoHide, "CLOSED", { closeReason: "APPEAL_WON" });
  holds = async () => ({ items: [hold("p1")], total: 1, offset: 0, limit: 100 });
  post = async () => ({ photoId: "p1", status: "LIVE" });
  mount();
  const d = await drawer();
  await within(d).findByRole("button", { name: "Restore batch…" });
  expect(actionButtons(d)).toEqual([expect.stringMatching(/^Restore photo/), "Restore batch…"]);
  // The server only releases what is still held: the case keeps its close and no payout hold ends.
  await user.click(within(d).getByRole("button", { name: "Restore batch…" }));
  expect(screen.getByText(CLOSED_RESTORE)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Cancel" }));
  await user.click(within(d).getByRole("button", { name: /^Restore photo/ }));
  await waitFor(() => expect(api.post).toHaveBeenCalledWith("/admin/relevance/cases/c1/photos/p1/restore"));
  // The photographer's own hide is not the admin's to release: nothing to offer.
  cleanup();
  client.clear();
  holds = async () => ({ items: [{ ...hold("p1"), heldBy: "PHOTOGRAPHER" }], total: 1, offset: 0, limit: 100 });
  mount();
  const own = await drawer();
  await within(own).findByRole("list", { name: "Held photos" });
  expect(buttons(own)).toEqual(["Close"]);
});

it("a refused closed-case restore shows the server's message", async () => {
  const user = userEvent.setup();
  rows = [];
  one = async () => detail(autoHide, "CLOSED", { closeReason: "SUPERSEDED" });
  holds = async () => ({ items: [hold("p1")], total: 1, offset: 0, limit: 100 });
  post = async () => { throw new ApiError([{ code: "INVALID_STATE_TRANSITION", message: "This action does not fit the case's current state" }], 409); };
  mount();
  await user.click(await within(await drawer()).findByRole("button", { name: "Restore batch…" }));
  await user.click(screen.getByRole("button", { name: "Confirm restore" }));
  await waitFor(() => expect(toasts()).toEqual(["This action does not fit the case's current state"]));
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
  expect(within(d).queryByRole("button", { name: "End hold…" })).not.toBeInTheDocument();
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
  await user.click(within(d).getByRole("button", { name: "End hold…" }));
  expect(api.post).not.toHaveBeenCalled();
  // AdminFlagService tells the photographer only once no other Photo Relevance hold is open.
  expect(screen.getByText("Ends the payout hold this case placed. The case itself stays as it is. The photographer is told their payout hold was lifted only when no other Photo Relevance hold is left.")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Confirm end hold" }));
  expect(api.post).toHaveBeenCalledWith("/admin/relevance/cases/c1/end-hold");
  // The server has not answered: the hold still shows and nothing was said.
  expect(field(d, "Payouts on hold since")).toBe("FEB 14 · 2003");
  expect(within(d).getByRole("button", { name: "End hold…" })).toBeInTheDocument();
  expect(toasts()).toEqual([]);
  one = async () => held({ payoutHoldAt: null });
  release(copied);
  await waitFor(() => expect(within(d).queryByText("Payouts on hold since")).not.toBeInTheDocument());
  expect(within(d).queryByRole("button", { name: "End hold…" })).not.toBeInTheDocument();
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
  await user.click(await within(d).findByRole("button", { name: "End hold…" }));
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
  await user.click(await within(d).findByRole("button", { name: "End hold…" }));
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
  // A repeat restore of a closed case only releases what is left (RVF-19).
  expect(screen.getByText(CLOSED_RESTORE)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Confirm restore" }));
  expect(api.post).toHaveBeenCalledWith("/admin/relevance/cases/c1/restore");
  await waitFor(() => expect(toasts()).toEqual(["Could not restore the batch. The case was read again: if photos are still held, press Restore batch again to finish the rest."]));
  // Nothing left held: no Restore.
  cleanup();
  client.clear();
  one = async () => detail(autoHide, "CLOSED", { closeReason: "RESTORED_BY_ADMIN" });
  holds = async () => ({ items: [hold("p1", "2003-02-16T00:00:00Z")], total: 1, offset: 0, limit: 100 });
  mount();
  const closed = await drawer();
  await within(closed).findByRole("list", { name: "Held photos" });
  expect(buttons(closed)).toEqual(["Close"]);
});

// RV-4.5b: a no-consent case (record-only, NO_CONSENT) that would have hidden the batch.
const noConsent = (verdict: string): RelevanceCaseRow =>
  ({ ...autoHide, verdict, verdictNote: "NO_CONSENT", appliedMode: "RECORD_ONLY", wouldHave: true });
const actionButtons = (el: HTMLElement) => buttons(el).filter((b) => b !== "Close");
const NO_CONSENT_HIDE = "Takes every photo of this batch off sale as your decision, a chunk at a time over the next minutes. New uploads to the batch are taken off sale too. The photographer is told a person took the photos off sale and may appeal. It does not count as a confirmed case. Works only while the relevance mode is auto-hide. You can restore the batch or photos one by one.";
const CANT_TELL_HIDE = "Takes every photo of this batch off sale, a chunk at a time over the next minutes. Works only while the relevance mode is auto-hide. You can restore the batch or photos one by one.";

it("a can't tell case keeps its own Hide batch confirm", async () => {
  const user = userEvent.setup();
  mount();
  await user.click(await within(await drawer()).findByRole("button", { name: "Hide batch…" }));
  expect(screen.getByText(CANT_TELL_HIDE)).toBeInTheDocument();
  expect(screen.queryByText(NO_CONSENT_HIDE)).not.toBeInTheDocument();
});

it("a no-consent case that would have hidden the batch offers Hide batch", async () => {
  const user = userEvent.setup();
  fixtures.search = "tab=all&row=c1";
  rows = [noConsent("JUNK_DUMP")];
  mount();
  const d = await drawer();
  await within(d).findByRole("button", { name: "Hide batch…" });
  expect(actionButtons(d)).toEqual(["Hide batch…"]);
  await user.click(within(d).getByRole("button", { name: "Hide batch…" }));
  // Its own confirm: the admin's decision, new uploads too, the photographer may appeal, no confirmed case.
  expect(screen.getByText(NO_CONSENT_HIDE)).toBeInTheDocument();
  expect(screen.queryByText(CANT_TELL_HIDE)).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Confirm hide batch" }));
  await waitFor(() => expect(api.post).toHaveBeenCalledWith("/admin/relevance/cases/c1/hide-batch"));
  // Wrong event would have hidden it too; Mixed and Fits would not, and a record-only case without the note stays inert.
  for (const [row, expected] of [
    [noConsent("WRONG_EVENT"), ["Hide batch…"]],
    [noConsent("MIXED"), []],
    [noConsent("FITS"), []],
    [{ ...noConsent("JUNK_DUMP"), verdictNote: null }, []],
  ] as const) {
    cleanup();
    client.clear();
    rows = [row];
    mount();
    const other = await drawer();
    await within(other).findByText("No photo of this batch is held.");
    expect(actionButtons(other)).toEqual(expected);
  }
  // A case that is no longer current offers nothing.
  cleanup();
  client.clear();
  rows = [];
  one = async () => detail(noConsent("JUNK_DUMP"), "CLOSED");
  mount();
  const closed = await drawer();
  await within(closed).findByText("No photo of this batch is held.");
  expect(actionButtons(closed)).toEqual([]);
});

// RVF-9: the mark is its own field, so the case keeps its own note; the page shows both.
it("a no-consent case with its own note still offers the no-consent actions", async () => {
  const user = userEvent.setup();
  fixtures.search = "tab=all&row=c1";
  rows = [{ ...noConsent("JUNK_DUMP"), verdictNote: "SUSPICIOUS_WORDS", noConsent: true }];
  mount();
  const d = await drawer();
  await within(d).findByRole("button", { name: "Hide batch…" });
  expect(actionButtons(d)).toEqual(["Hide batch…"]);
  expect(within(d).getByText("Words read off the photos seemed to tell the checker what to answer.")).toBeInTheDocument();
  expect(within(d).getByText("The photographer has not accepted the Terms that allow automatic hiding.")).toBeInTheDocument();
  await user.click(within(d).getByRole("button", { name: "Hide batch…" }));
  expect(screen.getByText(NO_CONSENT_HIDE)).toBeInTheDocument();
});

it("a no-consent Copied case offers Hold payouts until a hold was placed", async () => {
  const user = userEvent.setup();
  fixtures.search = "tab=all&row=c1";
  rows = [noConsent("COPIED")];
  let release!: (v: unknown) => void;
  post = () => new Promise((r) => { release = r; });
  mount();
  const d = await drawer();
  await within(d).findByRole("button", { name: "Hold payouts…" });
  expect(actionButtons(d)).toEqual(["Hide batch…", "Hold payouts…"]);
  await user.click(within(d).getByRole("button", { name: "Hold payouts…" }));
  expect(api.post).not.toHaveBeenCalled();
  // Before a hide there is nothing to appeal (RelevanceAppealService.file needs a hiding case): only End hold lifts it then.
  expect(screen.getByText("Places the payout hold an automatic Copied case would have placed. The photographer is told their payouts are on hold. End hold lifts it; once the batch is hidden, an accepted appeal lifts it too. Works only while the relevance mode is auto-hide.")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Confirm hold payouts" }));
  expect(api.post).toHaveBeenCalledWith("/admin/relevance/cases/c1/hold-payouts");
  // The server has not answered: the action is still offered and nothing was said.
  expect(within(d).getByRole("button", { name: "Hold payouts…" })).toBeInTheDocument();
  expect(toasts()).toEqual([]);
  one = async () => detail(noConsent("COPIED"), "APPLIED", { payoutHoldPlaced: true, payoutHoldAt: "2003-02-14T04:00:00Z" });
  release(noConsent("COPIED"));
  await within(d).findByRole("button", { name: "End hold…" });
  expect(actionButtons(d)).toEqual(["Hide batch…", "End hold…"]);
  expect(toasts()).toEqual(["Payouts held: Cebu City Marathon 2003"]);
  // A hold placed once and ended is not offered again.
  cleanup();
  client.clear();
  one = async () => detail(noConsent("COPIED"), "APPLIED", { payoutHoldPlaced: true });
  mount();
  const ended = await drawer();
  await within(ended).findByRole("button", { name: "Hide batch…" });
  expect(actionButtons(ended)).toEqual(["Hide batch…"]);
});

it("a no-consent Copied case an admin hid still offers Hold payouts", async () => {
  fixtures.search = "tab=all&row=c1";
  rows = [{ ...noConsent("COPIED"), appliedMode: "AUTO_HIDE", action: "HIDE_ALL", verdictNote: "ADMIN_HIDE", wouldHave: false }];
  holds = async () => ({ items: [hold("p1")], total: 1, offset: 0, limit: 100 });
  mount();
  const d = await drawer();
  await within(d).findByRole("button", { name: "Hold payouts…" });
  expect(actionButtons(d).filter((b) => !b?.startsWith("Restore photo"))).toEqual(["Restore batch…", "Keep hidden…", "Hold payouts…"]);
  // An automatic Copied case holds payouts by itself: no Hold payouts there.
  cleanup();
  client.clear();
  rows = [copied];
  mount();
  const auto = await drawer();
  await within(auto).findByRole("button", { name: "Restore batch…" });
  expect(within(auto).queryByRole("button", { name: "Hold payouts…" })).not.toBeInTheDocument();
});

it("a refused Hold payouts shows the server's message", async () => {
  const user = userEvent.setup();
  fixtures.search = "tab=all&row=c1";
  rows = [noConsent("COPIED")];
  post = async () => { throw new ApiError([{ code: "RELEVANCE_ACTIONS_OFF", message: "Holding payouts needs the relevance mode auto-hide" }], 409); };
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Hold payouts…" }));
  await user.click(screen.getByRole("button", { name: "Confirm hold payouts" }));
  await waitFor(() => expect(toasts()).toEqual(["Holding payouts needs the relevance mode auto-hide"]));
  expect(screen.getByRole("button", { name: "Confirm hold payouts" })).toBeEnabled();
});

it("a failed End hold with no server message says what to do next", async () => {
  const user = userEvent.setup();
  fixtures.search = "tab=all&row=c1";
  rows = [copied];
  one = async () => held();
  post = async () => { throw new Error("down"); };
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "End hold…" }));
  await user.click(screen.getByRole("button", { name: "Confirm end hold" }));
  await waitFor(() => expect(toasts()).toEqual(["Could not end the hold. The case was read again: if the payout hold still shows, press End hold again."]));
});

// FT-20 (C-54, W20.9): a whole-batch hide or restore prints how many photos, and from whom.
it("Hide batch and a whole-batch Restore print the photo count; a junk-only Restore does not", async () => {
  const user = userEvent.setup();
  const count = "420 photos in this batch, from @juanshots.";
  const progress = "You can follow the progress under Held photos.";
  mount();
  await user.click(await within(await drawer()).findByRole("button", { name: "Hide batch…" }));
  expect(screen.getByText(count)).toBeInTheDocument();
  expect(screen.getByText(progress)).toBeInTheDocument();
  cleanup();
  client.clear();
  rows = [autoHide];
  holds = async () => ({ items: [hold("p1")], total: 1, offset: 0, limit: 100 });
  mount();
  await user.click(await within(await drawer()).findByRole("button", { name: "Restore batch…" }));
  expect(screen.getByText(count)).toBeInTheDocument();
  expect(screen.queryByText(progress)).not.toBeInTheDocument();
  cleanup();
  client.clear();
  rows = [{ ...autoHide, verdict: "MIXED", action: "HIDE_JUNK" }];
  mount();
  await user.click(await within(await drawer()).findByRole("button", { name: "Restore batch…" }));
  expect(screen.queryByText(count)).not.toBeInTheDocument();
});

// FT-20 (C-52, W20.8): where a restore is offered, one line says how to decide.
it("a case that can be restored says to look at the junk samples first", async () => {
  const guide = "Look at the junk samples first. If most are real photos of this race, restore the batch.";
  rows = [autoHide];
  holds = async () => ({ items: [hold("p1")], total: 1, offset: 0, limit: 100 });
  mount();
  const d = await drawer();
  await within(d).findByRole("button", { name: "Restore batch…" });
  expect(within(d).getByText(guide)).toBeInTheDocument();
  cleanup();
  client.clear();
  rows = [cantTell];
  mount();
  const other = await drawer();
  await within(other).findByRole("button", { name: "Hide batch…" });
  expect(within(other).queryByText(guide)).not.toBeInTheDocument();
  // A Copied batch can be real race photos and still be a copy: no guidance there.
  cleanup();
  client.clear();
  rows = [copied];
  mount();
  const copy = await drawer();
  await within(copy).findByRole("button", { name: "Restore batch…" });
  expect(within(copy).queryByText(guide)).not.toBeInTheDocument();
});
