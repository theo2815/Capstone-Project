import { act, cleanup, fireEvent, render, renderHook, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError, setStepUpHandler } from "@/lib/api";
import type { AdminPayoutCycle } from "@/lib/admin-payouts";
import { adminPayoutKeys, usePayoutActions } from "@/hooks/use-admin-payouts";
import { admin, payoutCycle } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { StepUpProvider } from "@/components/auth/step-up-provider";
import { PayoutsQueue } from "./payouts-queue";

const navigation = vi.hoisted(() => ({ search: "", replace: vi.fn() }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: navigation.replace }),
  usePathname: () => "/admin/payouts",
  useSearchParams: () => new URLSearchParams(navigation.search),
}));

let client: QueryClient;
let rows: AdminPayoutCycle[];
// The legacy all-statuses list other admin views still read; settle() patches it.
const key = adminPayoutKeys.list(admin.id);
// A tiny payouts backend: status lists, id search, offset paging and the counts route (AX-10).
let countsMissing = false;
let failLists = 0;
let nextTabList: (() => Promise<unknown>) | null = null;
function serve(path: string): unknown {
  const url = new URL(path, "http://test");
  if (url.pathname === "/admin/payouts/counts") {
    if (countsMissing) throw new ApiError([{ code: "NOT_FOUND", message: "Not found" }], 404);
    const q = url.searchParams.get("q") ?? "";
    const counts: Record<string, number> = {};
    for (const row of rows) if (row.id.includes(q)) counts[row.status] = (counts[row.status] ?? 0) + 1;
    return { counts };
  }
  if (url.pathname !== "/admin/payouts") return { items: [], total: 0, offset: 0, limit: 50 };
  if (failLists > 0) { failLists -= 1; throw new Error("Unavailable"); }
  const statuses = url.searchParams.get("status")?.split(",");
  if (statuses && nextTabList) { const next = nextTabList; nextTabList = null; return next(); }
  const q = url.searchParams.get("q") ?? "";
  const offset = Number(url.searchParams.get("offset") ?? 0);
  const limit = Number(url.searchParams.get("limit") ?? 200);
  const matched = rows.filter((row) => (!statuses || statuses.includes(row.status)) && row.id.includes(q));
  return { items: matched.slice(offset, offset + limit), total: matched.length, offset, limit };
}
const listCalls = () => vi.mocked(api.get).mock.calls.map(([path]) => String(path))
  .filter((path) => path.startsWith("/admin/payouts?") && path.includes("status="));
const refresh = () => client.invalidateQueries({ queryKey: ["admin", "payouts"] });
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function render_() {
  return render(<QueryClientProvider client={client}><PayoutsQueue /><StepUpProvider /></QueryClientProvider>);
}
// Each tab reads its own server list, so wait for it before looking at rows.
async function mount() {
  const view = render_();
  await waitFor(() => expect(document.querySelector('[aria-busy="true"] .animate-pulse')).toBeNull());
  return view;
}

it("shows cross-view payout pending feedback while independent rows remain usable", async () => {
  rows = [payoutCycle, { ...payoutCycle, id: "cycle-2" }]; client.setQueryData(key, rows);
  const request = deferred<AdminPayoutCycle>();
  vi.spyOn(api, "post").mockReturnValue(request.promise);
  const actions = renderHook(usePayoutActions, { wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider> });
  await mount();
  let pending!: Promise<unknown>;
  act(() => { pending = actions.result.current.approve(payoutCycle.id); });
  await waitFor(() => expect(screen.getByText("Approving…")).toBeInTheDocument());
  const approvals = screen.getAllByRole("button", { name: /Approve|Approving/ });
  expect(approvals.filter((button) => button.hasAttribute("disabled"))).toHaveLength(1);
  expect(approvals.filter((button) => !button.hasAttribute("disabled"))).toHaveLength(1);
  await act(async () => { request.reject(new Error("Step-up cancelled")); await pending; });
  await waitFor(() => expect(screen.getAllByRole("button", { name: "Approve" }).every((button) => !button.hasAttribute("disabled"))).toBe(true));
});
it("opens an unrelated payout drawer while a selected row approval is pending", async () => {
  rows = [payoutCycle, { ...payoutCycle, id: "cycle-2" }]; client.setQueryData(key, rows);
  const request = deferred<AdminPayoutCycle>();
  const post = vi.spyOn(api, "post").mockReturnValue(request.promise);
  // The approval runs from another view (the row's own Approve now opens a modal confirm).
  const actions = renderHook(usePayoutActions, { wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider> });
  await mount();
  await userEvent.click(screen.getByRole("checkbox", { name: `Select payout ${payoutCycle.id}` }));
  act(() => { void actions.result.current.approve(payoutCycle.id); });
  await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
  await userEvent.click(screen.getByRole("button", { name: "Open payout cycle-2" }));
  expect(screen.getByRole("dialog")).toHaveTextContent("cycle-2");
  expect(within(screen.getByRole("dialog")).getByRole("button", { name: "Approve" })).toBeEnabled();
  await act(async () => { request.reject(new Error("Unavailable")); });
});

// Payouts show one status tab at a time; open the tab a seeded status lives on.
const TAB_FOR: Record<AdminPayoutCycle["status"], string> = {
  pending_review: "pending", approved: "in-progress", processing: "in-progress", held: "held", paid: "paid",
  failed: "closed", returned: "closed", rejected: "closed", cancelled: "closed",
};
function seed(status: AdminPayoutCycle["status"] = "pending_review") {
  rows = [{ ...payoutCycle, status }]; client.setQueryData(key, rows);
  navigation.search = status === "pending_review" ? "" : `status=${TAB_FOR[status]}`;
}
beforeEach(() => {
  useAuthStore.getState().setUser(admin);
  useToastStore.getState().clear();
  navigation.search = ""; navigation.replace.mockClear();
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  vi.spyOn(console, "error").mockImplementation(() => {});
  countsMissing = false; failLists = 0; nextTabList = null;
  vi.spyOn(api, "get").mockImplementation(async (path: string) => serve(path) as never);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }); seed();
});
afterEach(() => {
  cleanup(); client.clear(); useAuthStore.getState().logout(); setStepUpHandler(null);
  Reflect.deleteProperty(HTMLElement.prototype, "scrollIntoView");
  vi.restoreAllMocks(); vi.unstubAllGlobals();
});

it("keeps the payout URL selection until its read finishes", async () => {
  navigation.search = `row=${payoutCycle.id}`;
  const request = deferred<unknown>();
  nextTabList = () => request.promise;
  render_();
  expect(navigation.replace).not.toHaveBeenCalled();
  expect(await screen.findByRole("dialog")).toHaveTextContent(payoutCycle.id);
  await act(async () => { request.resolve(serve("/admin/payouts?status=pending_review&offset=0&limit=50")); });
  expect(screen.getByRole("dialog")).toHaveTextContent(payoutCycle.id);
  expect(navigation.replace).not.toHaveBeenCalled();
});

it("renders a retryable read error instead of dropping the selected payout", async () => {
  navigation.search = `row=${payoutCycle.id}`;
  failLists = 2;
  await mount();
  expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't load payouts");
  expect(navigation.replace).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByRole("dialog")).toHaveTextContent(payoutCycle.id);
});

it("blocks repeated row/drawer approvals while pending and keeps the first approval pending", async () => {
  const user = userEvent.setup();
  const request = deferred<AdminPayoutCycle>();
  const post = vi.spyOn(api, "post").mockReturnValueOnce(request.promise);
  await mount();
  await user.click(screen.getByRole("button", { name: `Open payout ${payoutCycle.id}` }));
  const drawer = screen.getByRole("dialog");
  const approve = within(drawer).getByRole("button", { name: "Approve" });
  await user.click(approve);
  await user.click(within(screen.getByRole("dialog", { name: "Approve payout" })).getByRole("button", { name: "Approve" }));
  expect(approve).toBeDisabled();
  expect(screen.getByRole("button", { name: "Approving…" })).toBeDisabled();
  expect(screen.getAllByRole("button", { name: "Approve" }).every((button) => button.hasAttribute("disabled"))).toBe(true);
  await user.keyboard("e"); await user.click(approve);
  expect(post).toHaveBeenCalledTimes(1);
  expect(useToastStore.getState().toasts).toHaveLength(0);
  rows = [{ ...payoutCycle, approvedBy: admin.id }];
  await act(async () => { request.resolve(rows[0]); });
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  expect(client.getQueryData(key)).toEqual(rows);
  expect(useToastStore.getState().toasts.at(-1)?.message).toBe(`First approval recorded. A second admin must approve ${payoutCycle.id}.`);
  await user.click(screen.getByRole("button", { name: `Open payout ${payoutCycle.id}` }));
  expect(screen.getByRole("dialog")).toHaveTextContent("1 of 2");
  expect(screen.queryByRole("button", { name: "Mark paid…" })).not.toBeInTheDocument();
  // A later server refresh, not a local override, unlocks sending.
  rows = [{ ...rows[0], status: "approved", secondApprovedBy: "admin-b", reviewedAt: "2026-09-10T05:00:00Z" }];
  await act(async () => { await refresh(); });
  await waitFor(() => expect(within(screen.getByRole("dialog")).getByRole("button", { name: "Mark paid…" })).toBeEnabled());
});

// FT-23 (C-16, W23.2): the backend refuses a second approval by the same admin, so it is not offered.
it("the admin who gave the first approval cannot approve again; without approvedBy Approve stays enabled", async () => {
  const user = userEvent.setup();
  const older = { ...payoutCycle, id: "cycle-2" } as Partial<AdminPayoutCycle>;
  delete older.approvedBy; // an older backend sends no approvedBy
  rows = [{ ...payoutCycle, approvedBy: admin.id }, older as AdminPayoutCycle]; client.setQueryData(key, rows);
  const post = vi.spyOn(api, "post");
  await mount();
  const own = document.querySelector<HTMLElement>(`[data-row-id="${payoutCycle.id}"]`)!;
  const other = document.querySelector<HTMLElement>('[data-row-id="cycle-2"]')!;
  const sentence = "You gave the first approval. A different admin must give the second.";
  expect(within(own).getByRole("button", { name: "Approve" })).toBeDisabled();
  expect(own).toHaveTextContent(sentence);
  expect(within(other).getByRole("button", { name: "Approve" })).toBeEnabled();
  expect(other).not.toHaveTextContent(sentence);
  await user.click(screen.getByRole("button", { name: `Open payout ${payoutCycle.id}` }));
  const drawer = screen.getByRole("dialog");
  expect(within(drawer).getByRole("button", { name: "Approve" })).toBeDisabled();
  expect(drawer).toHaveTextContent(sentence);
  await user.keyboard("e");
  expect(screen.queryByRole("dialog", { name: "Approve payout" })).not.toBeInTheDocument();
  expect(post).not.toHaveBeenCalled();
});

// FT-23 (C-15, W23.4): the approval that completes a payout says what to do next.
it("an approval that makes the payout approved says to send the money", async () => {
  const user = userEvent.setup();
  vi.spyOn(api, "post").mockImplementationOnce(async () => {
    rows = [{ ...payoutCycle, status: "approved", approvedBy: admin.id, reviewedAt: "2026-09-10T05:00:00Z" }]; return rows[0];
  });
  await mount();
  await user.click(screen.getByRole("button", { name: "Approve" }));
  await user.click(within(screen.getByRole("dialog", { name: "Approve payout" })).getByRole("button", { name: "Approve" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  expect(useToastStore.getState().toasts.at(-1)).toMatchObject({ kind: "success", message: `Approved ${payoutCycle.id}. Send the money, then mark it paid.` });
});

it.each([
  ["hold", "pending_review", "Hold…", "Hold payout", "held"],
  ["mark-paid", "approved", "Mark paid…", "Mark paid", "paid"],
  ["reject", "pending_review", "Reject…", "Reject payout", "rejected"],
  ["mark-failed", "processing", "Mark failed…", "Mark failed", "failed"],
  ["mark-returned", "paid", "Mark returned…", "Mark returned", "returned"],
] as const)("retains %s input on failure and closes only after successful retry", async (action, status, trigger, submitLabel, finalStatus) => {
  const user = userEvent.setup(); seed(status);
  const request = deferred<AdminPayoutCycle>();
  const post = vi.spyOn(api, "post").mockReturnValueOnce(request.promise);
  await mount();
  await user.click(screen.getByRole("button", { name: `Open payout ${payoutCycle.id}` }));
  await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: trigger }));
  const form = screen.getAllByRole("dialog").find((dialog) => within(dialog).queryByRole("textbox"))!;
  const input = within(form).getByRole("textbox");
  // FT-33 (W33.1): the reason field names who reads it; Mark paid has no reason field.
  if (action !== "mark-paid") expect(input).toHaveAccessibleDescription(action === "hold" ? "Seen by: the photographer, exactly as written." : "Seen by: the photographer.");
  if (action === "hold") await user.click(within(form).getAllByRole("radio")[0]!);
  await user.type(input, "Reference or note retained");
  await user.click(within(form).getByRole("button", { name: submitLabel }));
  expect(within(form).getByRole("button", { name: action === "hold" ? "Holding…" : action === "mark-paid" ? "Marking paid…" : action === "reject" ? "Rejecting…" : action === "mark-failed" ? "Marking failed…" : "Marking returned…" })).toBeDisabled();
  expect(input).toBeDisabled();
  await user.keyboard("{Escape}");
  expect(form).toBeInTheDocument();
  expect(client.getQueryData(key)).toEqual(rows);
  await act(async () => { request.reject(new Error("Unavailable")); });
  expect(input).toHaveValue("Reference or note retained");
  await waitFor(() => expect(within(form).getByRole("button", { name: submitLabel })).toBeEnabled());
  expect(useToastStore.getState().toasts.at(-1)?.kind).toBe("error");
  if (action === "hold") expect(within(form).getByRole("alert")).toHaveTextContent(/Couldn.t finish/);
  post.mockImplementationOnce(async () => {
    rows = [{ ...rows[0], status: finalStatus }]; return rows[0];
  });
  await user.click(within(form).getByRole("button", { name: submitLabel }));
  await waitFor(() => expect(form).not.toBeInTheDocument());
  expect(post).toHaveBeenCalledTimes(2);
  expect(post.mock.calls[0][0]).toBe(`/admin/payouts/${payoutCycle.id}/${action}`);
  expect(client.getQueryData(key)).toEqual(rows);
  // FT-23 (C-27, W23.6 to W23.8): the toast says who was told, as a success.
  expect(useToastStore.getState().toasts.at(-1)).toMatchObject({ kind: "success", message: {
    hold: "On hold. The photographer was told why.",
    "mark-paid": "Marked paid. The photographer was told, with reference Reference or note retained.",
    reject: "Rejected. The photographer was told, and the money is back in their balance.",
    "mark-failed": "Marked failed. The photographer was told, and the money is back in their balance.",
    "mark-returned": "Marked returned. The photographer was told to check their payout account.",
  }[action] });
});

it.each(["approve", "hold"])("keeps bulk %s pending through partial-result reconciliation", async (action) => {
  const user = userEvent.setup();
  rows = [payoutCycle, { ...payoutCycle, id: "cycle-2" }]; client.setQueryData(key, rows);
  const request = deferred<unknown>();
  const post = vi.spyOn(api, "post").mockReturnValueOnce(request.promise);
  await mount();
  for (const row of rows) await user.click(screen.getByRole("checkbox", { name: `Select payout ${row.id}` }));
  await user.click(screen.getByRole("button", { name: action === "approve" ? "Approve 2" : "Hold 2…" }));
  const modal = screen.getByRole("dialog");
  const submit = action === "approve" ? "Approve" : "Hold 2 payouts";
  if (action === "hold") await user.click(within(modal).getAllByRole("radio")[0]!);
  await user.click(within(modal).getByRole("button", { name: submit }));
  expect(within(modal).getByRole("button", { name: action === "approve" ? "Approving…" : "Holding…" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Clear" })).toBeDisabled();
  await act(async () => { request.resolve({ groupId: "partial", results: [
    { id: payoutCycle.id, ok: true }, { id: "cycle-2", ok: false, error: "A different admin must give the second approval for this amount." },
  ] }); });
  expect(modal).toBeInTheDocument();
  // FT-16 (C-22): the approve confirm names the payout that failed, with the server's reason.
  if (action === "approve") {
    expect(within(modal).getByRole("alert")).toHaveTextContent("cycle-2: A different admin must give the second approval for this amount.");
    expect(within(modal).getByRole("alert")).not.toHaveTextContent(payoutCycle.id + ":");
  }
  const retryLabel = action === "approve" ? "Approve" : "Hold payout";
  expect(within(modal).getByRole("button", { name: retryLabel })).toBeEnabled();
  expect(screen.getByRole("checkbox", { name: `Select payout ${payoutCycle.id}` })).not.toBeChecked();
  expect(screen.getByRole("checkbox", { name: "Select payout cycle-2" })).toBeChecked();
  expect(useToastStore.getState().toasts.at(-1)?.message).toContain("1 succeeded, 1 failed");
  if (action === "hold") expect(within(modal).getByRole("alert")).toHaveTextContent(/Couldn.t finish/);
  expect(useToastStore.getState().toasts.some((toast) => toast.kind === "success")).toBe(false);
  post.mockResolvedValueOnce({ groupId: "retry", results: [{ id: "cycle-2", ok: true }] });
  await user.click(within(modal).getByRole("button", { name: retryLabel }));
  await waitFor(() => expect(modal).not.toBeInTheDocument());
  expect(post).toHaveBeenLastCalledWith("/admin/payouts/bulk", expect.objectContaining({ ids: ["cycle-2"] }), expect.anything());
});

// FT-16 (C-22): the bulk confirm lists what it approves and uses the same verb as the single one.
it("the bulk approve confirm lists every selected payout and its button is Approve", async () => {
  const user = userEvent.setup();
  rows = [payoutCycle, { ...payoutCycle, id: "cycle-2", brandName: "Second Studio", amount: 1200 }]; client.setQueryData(key, rows);
  await mount();
  for (const row of rows) await user.click(screen.getByRole("checkbox", { name: `Select payout ${row.id}` }));
  await user.click(screen.getByRole("button", { name: "Approve 2" }));
  const modal = screen.getByRole("dialog");
  const list = within(modal).getByRole("list");
  expect(within(list).getAllByRole("listitem")).toHaveLength(2);
  expect(list).toHaveTextContent(payoutCycle.id);
  expect(list).toHaveTextContent("cycle-2");
  expect(list).toHaveTextContent("Second Studio");
  expect(within(modal).getByRole("button", { name: "Approve" })).toBeEnabled();
  expect(within(modal).queryByRole("button", { name: "Confirm approve" })).toBeNull();
  expect(modal).toHaveTextContent("Approving sends no money. A larger payout needs a second, different admin before it can be paid.");
});

it.each(["approve", "hold"])("closes bulk %s and clears selection only after confirmation", async (action) => {
  const user = userEvent.setup();
  const request = deferred<unknown>();
  vi.spyOn(api, "post").mockReturnValueOnce(request.promise);
  await mount();
  await user.click(screen.getByRole("checkbox", { name: `Select payout ${payoutCycle.id}` }));
  await user.click(screen.getByRole("button", { name: action === "approve" ? "Approve 1" : "Hold 1…" }));
  const modal = screen.getByRole("dialog");
  if (action === "hold") await user.click(within(modal).getAllByRole("radio")[0]!);
  await user.click(within(modal).getByRole("button", { name: action === "approve" ? "Approve" : "Hold payout" }));
  expect(modal).toBeInTheDocument();
  expect(screen.getByRole("checkbox")).toBeChecked();
  rows = [{ ...payoutCycle, status: action === "hold" ? "held" : "pending_review", approvedBy: admin.id }];
  await act(async () => { request.resolve({ groupId: "batch", results: [{ id: payoutCycle.id, ok: true }] }); });
  await waitFor(() => expect(modal).not.toBeInTheDocument());
  // Approve leaves the row on this tab (unticked); hold moves it to the Held tab. Both fail if the row is wrong.
  const box = screen.queryByRole("checkbox", { name: `Select payout ${payoutCycle.id}` });
  if (action === "approve") expect(box).not.toBeChecked();
  else expect(box).toBeNull();
  expect(screen.queryByRole("button", { name: action === "approve" ? "Approve 1" : "Hold 1…" })).not.toBeInTheDocument();
  // Bulk results carry no records, so the tab re-reads its server list before the modal closes.
  expect(listCalls()).toHaveLength(2);
  // FT-23 (W23.10, W23.7): the bulk toasts, typed as the wording sheet has them.
  expect(useToastStore.getState().toasts.at(-1)).toMatchObject({ kind: "success", message: action === "approve"
    ? "Approvals recorded for 1 payout." : "1 payouts on hold. The photographers were told why." });
});

it("preserves provider-controlled action guards, including drawer keyboard shortcuts", async () => {
  const user = userEvent.setup(); rows = [{ ...payoutCycle, status: "processing", provider: "paymongo" }];
  navigation.search = "status=in-progress";
  client.setQueryData(key, rows); const post = vi.spyOn(api, "post"); await mount();
  await user.click(screen.getByRole("button", { name: `Open payout ${payoutCycle.id}` }));
  expect(screen.getByRole("dialog")).toHaveTextContent("Payment provider manages this transfer");
  expect(screen.queryByRole("button", { name: "Mark paid…" })).not.toBeInTheDocument();
  await user.keyboard("eh"); expect(post).not.toHaveBeenCalled();
});

it("retains a mark-paid form through step-up cancellation, invalid credentials, and proof retry", async () => {
  const user = userEvent.setup(); seed("approved");
  const proofRequest = deferred<Response>();
  const fetch = vi.fn(async (url: string, options: RequestInit) => {
    if (url.endsWith("/me/reauth")) {
      return JSON.parse(options.body as string).password === "wrong"
        ? new Response(JSON.stringify({ success: false, errors: [{ code: "INVALID_PASSWORD", message: "Incorrect password" }] }), { status: 400 })
        : new Response(JSON.stringify({ success: true, data: { stepUpToken: "fixture-proof" } }));
    }
    if (new Headers(options.headers).get("X-Step-Up")) return proofRequest.promise;
    return new Response(JSON.stringify({ success: false, errors: [{ code: "STEP_UP_REQUIRED", message: "Reauthenticate" }] }), { status: 403 });
  });
  vi.stubGlobal("fetch", fetch); await mount();
  await user.click(screen.getByRole("button", { name: "Mark paid…" }));
  const form = screen.getByRole("dialog");
  await user.type(within(form).getByRole("textbox"), "GC-REFERENCE");
  await user.click(within(form).getByRole("button", { name: "Mark paid" }));
  let stepUp = await screen.findByRole("dialog", { name: "Confirm it's you" });
  expect(within(form).getByRole("button", { name: "Marking paid…" })).toBeDisabled();
  await user.keyboard("{Escape}");
  await waitFor(() => expect(stepUp).not.toBeInTheDocument());
  expect(within(form).getByRole("textbox")).toHaveValue("GC-REFERENCE");
  expect(client.getQueryData(key)).toEqual(rows);
  await user.click(await within(form).findByRole("button", { name: "Mark paid" }));
  stepUp = await screen.findByRole("dialog", { name: "Confirm it's you" });
  await user.type(within(stepUp).getByLabelText("Password"), "wrong");
  await user.click(within(stepUp).getByRole("button", { name: "Confirm" }));
  expect(await within(stepUp).findByText("Incorrect password")).toBeInTheDocument();
  expect(within(form).getByRole("button", { name: "Marking paid…" })).toBeDisabled();
  await user.clear(within(stepUp).getByLabelText("Password"));
  await user.type(within(stepUp).getByLabelText("Password"), "fixture-password");
  await user.click(within(stepUp).getByRole("button", { name: "Confirm" }));
  const retried = fetch.mock.calls.filter(([, opts]) => new Headers(opts.headers).get("X-Step-Up"));
  expect(retried).toHaveLength(1);
  expect(JSON.parse(retried[0][1].body as string)).toEqual({ paymentReference: "GC-REFERENCE" });
  expect(within(form).getByRole("button", { name: "Marking paid…" })).toBeDisabled();
  rows = [{ ...rows[0], status: "paid", paymentReference: "GC-REFERENCE" }];
  await act(async () => { proofRequest.resolve(new Response(JSON.stringify({ success: true, data: rows[0] }))); });
  await waitFor(() => expect(form).not.toBeInTheDocument());
  expect(client.getQueryData(key)).toEqual(rows);
  expect(useAuthStore.getState().isAuthenticated).toBe(true);
  expect(useToastStore.getState().toasts.at(-1)?.message).toBe("Marked paid. The photographer was told, with reference GC-REFERENCE.");
});

it("a held photographer's payout names the flag and disables approve and mark paid", async () => {
  const user = userEvent.setup();
  rows = [
    { ...payoutCycle, relevanceFlagId: "flag-1", relevanceHold: true },
    { ...payoutCycle, id: "cycle-2", status: "approved", relevanceFlagId: "flag-1", relevanceHold: true },
  ]; client.setQueryData(key, rows);
  const view = await mount();
  expect(screen.getAllByText("Payouts paused: a photo check on this photographer is still open")).toHaveLength(1);
  expect(screen.getAllByRole("link", { name: /photo check/i })[0]).toHaveAttribute("href", "/admin/flags?row=flag-1");
  expect(screen.getByRole("button", { name: "Approve" })).toBeDisabled();
  expect(screen.getAllByRole("button", { name: "Hold…" })[0]).toBeEnabled();
  view.unmount(); navigation.search = "status=in-progress"; await mount();
  expect(screen.getAllByText("Payouts paused: a photo check on this photographer is still open")).toHaveLength(1);
  expect(screen.getByRole("button", { name: "Mark processing" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Mark paid…" })).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "Open payout cycle-2" }));
  const drawer = screen.getByRole("dialog");
  expect(within(drawer).getByRole("button", { name: "Mark paid…" })).toBeDisabled();
  expect(within(drawer).getByRole("button", { name: "Hold…" })).toBeEnabled();
  expect(within(drawer).getByRole("button", { name: "Reject…" })).toBeEnabled();
});
it("an unconfirmed flag shows a warning and leaves the buttons on", async () => {
  rows = [{ ...payoutCycle, relevanceFlagId: "flag-1", relevanceHold: false }]; client.setQueryData(key, rows);
  await mount();
  expect(screen.getByText("A photo check on this photographer is open, not yet confirmed")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Approve" })).toBeEnabled();
});
it("the bulk bar refuses approve while a selected payout is on hold, and E does nothing in the drawer", async () => {
  const user = userEvent.setup();
  rows = [{ ...payoutCycle, relevanceFlagId: "flag-1", relevanceHold: true }, { ...payoutCycle, id: "cycle-2" }];
  client.setQueryData(key, rows);
  const post = vi.spyOn(api, "post");
  await mount();
  await user.click(screen.getByRole("checkbox", { name: `Select payout ${payoutCycle.id}` }));
  await user.click(screen.getByRole("checkbox", { name: "Select payout cycle-2" }));
  expect(screen.getByText("Some selected payouts are paused by an open photo check")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /^Approve \d/ })).toBeDisabled();
  await user.click(screen.getByRole("button", { name: `Open payout ${payoutCycle.id}` }));
  await user.keyboard("e");
  expect(post).not.toHaveBeenCalled();
});

it("the flag link in a row does not open the drawer or toggle selection, and Enter on it is not cancelled", async () => {
  const user = userEvent.setup();
  rows = [{ ...payoutCycle, relevanceFlagId: "flag-1", relevanceHold: true }]; client.setQueryData(key, rows);
  await mount();
  const link = screen.getByRole("link", { name: "Payouts paused: a photo check on this photographer is still open" });
  await user.click(link);
  expect(navigation.replace).not.toHaveBeenCalled();
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(screen.getByRole("checkbox", { name: `Select payout ${payoutCycle.id}` })).not.toBeChecked();
  expect(fireEvent.keyDown(link, { key: "Enter" })).toBe(true);
  expect(navigation.replace).not.toHaveBeenCalled();
  expect(screen.queryByRole("dialog")).toBeNull();
});

it("an admin shortcut key pressed on the flag link still reaches the page", async () => {
  rows = [{ ...payoutCycle, relevanceFlagId: "flag-1", relevanceHold: true }]; client.setQueryData(key, rows);
  await mount();
  const link = screen.getByRole("link", { name: "Payouts paused: a photo check on this photographer is still open" });
  const seen = vi.fn();
  window.addEventListener("keydown", seen);
  try {
    fireEvent.keyDown(link, { key: "j" });
    expect(seen).toHaveBeenCalledTimes(1);
  } finally {
    window.removeEventListener("keydown", seen);
  }
});

it("shows one status tab at a time with counts, and keeps the tab in the URL", async () => {
  const user = userEvent.setup();
  rows = [payoutCycle, { ...payoutCycle, id: "cycle-2", status: "paid", paymentReference: "GC-1" }]; client.setQueryData(key, rows);
  await mount();
  const tabs = screen.getByRole("group", { name: "Payout status" });
  expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
  expect(within(tabs).getByRole("button", { name: /Pending review 1/, pressed: true })).toBeInTheDocument();
  expect(within(tabs).getByRole("button", { name: /Paid 1/, pressed: false })).toBeInTheDocument();
  expect(screen.getByRole("checkbox", { name: "Select payout cycle-1" })).toBeInTheDocument();
  expect(screen.queryByRole("checkbox", { name: "Select payout cycle-2" })).not.toBeInTheDocument();
  await user.click(within(tabs).getByRole("button", { name: /Paid 1/ }));
  expect(navigation.replace).toHaveBeenLastCalledWith(expect.stringContaining("status=paid"), expect.anything());
});

it("clears the ticked payouts when the URL status changes (Back, Forward or a link)", async () => {
  const user = userEvent.setup();
  rows = [payoutCycle, { ...payoutCycle, id: "cycle-2", status: "held" }]; client.setQueryData(key, rows);
  const view = await mount();
  await user.click(screen.getByRole("checkbox", { name: "Select payout cycle-1" }));
  expect(screen.getByRole("button", { name: "Approve 1" })).toBeInTheDocument();
  const again = () => view.rerender(<QueryClientProvider client={client}><PayoutsQueue /><StepUpProvider /></QueryClientProvider>);
  navigation.search = "status=held"; again();
  await waitFor(() => expect(screen.getByRole("checkbox", { name: "Select payout cycle-2" })).toBeInTheDocument());
  expect(screen.queryByRole("button", { name: /^Approve \d/ })).not.toBeInTheDocument();
  navigation.search = ""; again();
  expect(await screen.findByRole("checkbox", { name: "Select payout cycle-1" })).not.toBeChecked();
});

it("clears the ticked payouts when the status tab changes", async () => {
  const user = userEvent.setup();
  rows = [payoutCycle, { ...payoutCycle, id: "cycle-2", status: "held" }]; client.setQueryData(key, rows);
  await mount();
  const tabs = screen.getByRole("group", { name: "Payout status" });
  await user.click(screen.getByRole("checkbox", { name: "Select payout cycle-1" }));
  expect(screen.getByRole("button", { name: "Approve 1" })).toBeInTheDocument();
  await user.click(within(tabs).getByRole("button", { name: /Held 1/ }));
  expect(screen.queryByRole("button", { name: /^Approve \d/ })).not.toBeInTheDocument();
  await user.click(within(tabs).getByRole("button", { name: /Pending review 1/ }));
  expect(screen.getByRole("checkbox", { name: "Select payout cycle-1" })).not.toBeChecked();
});

// AX-29 (PAY2): every money step asks first, like bulk Approve.
it.each([
  ["a row Approve", "pending_review", "approve", "Approve payout"],
  ["the drawer E key", "pending_review", "approve", "Approve payout"],
  ["Mark processing", "approved", "mark-processing", "Mark processing"],
] as const)("%s needs the confirm before anything is sent", async (how, status, action, title) => {
  const user = userEvent.setup(); seed(status);
  const post = vi.spyOn(api, "post").mockImplementation(async () => {
    rows = [{ ...rows[0], status: action === "approve" ? "approved" : "processing" }]; return rows[0];
  });
  await mount();
  if (how === "the drawer E key") {
    await user.click(screen.getByRole("button", { name: `Open payout ${payoutCycle.id}` }));
    await user.keyboard("e");
  } else {
    await user.click(screen.getByRole("button", { name: how === "a row Approve" ? "Approve" : "Mark processing" }));
  }
  expect(post).not.toHaveBeenCalled();
  const confirm = screen.getByRole("dialog", { name: title });
  await user.click(within(confirm).getByRole("button", { name: action === "approve" ? "Approve" : "Mark processing" }));
  await waitFor(() => expect(confirm).not.toBeInTheDocument());
  expect(post).toHaveBeenCalledTimes(1);
  expect(post.mock.calls[0][0]).toBe(`/admin/payouts/${payoutCycle.id}/${action}`);
});

it("a failed single approval keeps the confirm open, and a retry sends once more", async () => {
  const user = userEvent.setup();
  const post = vi.spyOn(api, "post").mockRejectedValueOnce(new Error("Unavailable"));
  await mount();
  await user.click(screen.getByRole("button", { name: "Approve" }));
  const confirm = screen.getByRole("dialog", { name: "Approve payout" });
  await user.click(within(confirm).getByRole("button", { name: "Approve" }));
  await waitFor(() => expect(within(confirm).getByRole("button", { name: "Approve" })).toBeEnabled());
  expect(confirm).toBeInTheDocument();
  expect(useToastStore.getState().toasts.at(-1)?.kind).toBe("error");
  post.mockImplementationOnce(async () => { rows = [{ ...rows[0], status: "approved" }]; return rows[0]; });
  await user.click(within(confirm).getByRole("button", { name: "Approve" }));
  await waitFor(() => expect(confirm).not.toBeInTheDocument());
  expect(post).toHaveBeenCalledTimes(2);
});

// AX-29 (PAY3): a money queue never reads "nothing to do" after a failed load.
it("a failed first load shows an alert, not tabs that read nothing to do", async () => {
  failLists = 1; countsMissing = true;
  await mount();
  const alert = await screen.findByRole("alert");
  expect(within(alert).getByText("Couldn't load payouts.")).toBeInTheDocument();
  expect(within(alert).getByRole("button", { name: "Retry" })).toBeInTheDocument();
  expect(screen.queryByText("No payouts waiting. You're caught up.")).not.toBeInTheDocument();
  // The tabs stay so the admin can switch, but none reads a number it does not know.
  expect(within(screen.getByRole("group", { name: "Payout status" })).getByRole("button", { name: "Pending review" })).toBeInTheDocument();
});

// AX-29 (PAY4): checkboxes only where a bulk action exists, and only for rows still on the tab.
it.each(["paid", "cancelled"] as const)("no checkbox appears on a %s payout", async (status) => {
  seed(status); await mount();
  expect(screen.getByRole("button", { name: `Open payout ${payoutCycle.id}` })).toBeInTheDocument();
  expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
});
it("a ticked payout that a refetch moves to another tab drops out of the selection", async () => {
  const user = userEvent.setup();
  rows = [payoutCycle, { ...payoutCycle, id: "cycle-2" }]; client.setQueryData(key, rows);
  await mount();
  await user.click(screen.getByRole("checkbox", { name: "Select payout cycle-1" }));
  expect(screen.getByRole("button", { name: "Approve 1" })).toBeInTheDocument();
  rows = [{ ...payoutCycle, status: "held" }, { ...payoutCycle, id: "cycle-2" }];
  await act(async () => { await refresh(); });
  await waitFor(() => expect(screen.queryByRole("button", { name: "Approve 1" })).not.toBeInTheDocument());
});

// AX-29 (PAY5, PAY6): house buttons in the drawer, and values in their real case.
it("the drawer's Hold and Reject are danger buttons, and Approve is the primary one", async () => {
  const user = userEvent.setup(); await mount();
  await user.click(screen.getByRole("button", { name: `Open payout ${payoutCycle.id}` }));
  const drawer = screen.getByRole("dialog");
  expect(within(drawer).getByRole("button", { name: "Hold…" })).toHaveClass("text-error", "min-h-[44px]");
  expect(within(drawer).getByRole("button", { name: "Reject…" })).toHaveClass("text-error");
  expect(within(drawer).getByRole("button", { name: "Approve" })).toHaveClass("bg-fresh", "min-h-[44px]");
});
it("a payment reference renders as typed in the drawer", async () => {
  const user = userEvent.setup(); seed("paid");
  rows = [{ ...rows[0], paymentReference: "gc-77b221", paidAt: "2026-09-12T00:00:00Z" }]; client.setQueryData(key, rows);
  await mount();
  await user.click(screen.getByRole("button", { name: `Open payout ${payoutCycle.id}` }));
  const value = within(screen.getByRole("dialog")).getByText("gc-77b221");
  expect(value).not.toHaveClass("uppercase");
});

// AX-30 (PAY1, X6): each tab is its own server list, so nothing drops off past 200 rows.
const many = (n: number, status: AdminPayoutCycle["status"], prefix: string) =>
  Array.from({ length: n }, (_, i) => ({ ...payoutCycle, id: `${prefix}-${i + 1}`, status }));
const statusParam = (path: string) => new URL(path, "http://test").searchParams;
it("each tab asks the server for its own statuses", async () => {
  const user = userEvent.setup();
  await mount();
  const tabs = screen.getByRole("group", { name: "Payout status" });
  for (const label of ["In progress", "Held", "Paid", "Closed"]) {
    await user.click(within(tabs).getByRole("button", { name: new RegExp(`^${label}`) }));
  }
  await waitFor(() => expect(listCalls().map((path) => statusParam(path).get("status"))).toEqual([
    "pending_review", "approved,processing", "held", "paid", "failed,returned,rejected,cancelled",
  ]));
  expect(listCalls().every((path) => statusParam(path).get("offset") === "0")).toBe(true);
});
it("Load more appends the next server page", async () => {
  const user = userEvent.setup(); rows = many(60, "pending_review", "p");
  await mount();
  expect(screen.getAllByRole("checkbox")).toHaveLength(50);
  await user.click(screen.getByRole("button", { name: "Load 10 more" }));
  await waitFor(() => expect(screen.getAllByRole("checkbox")).toHaveLength(60));
  expect(statusParam(listCalls().at(-1)!).get("offset")).toBe("50");
  expect(screen.getByRole("checkbox", { name: "Select payout p-60" })).toBeInTheDocument();
});
it("tab counts come from the counts route, and a missing route shows none", async () => {
  rows = [...many(257, "paid", "new"), ...many(3, "pending_review", "old")];
  const view = await mount();
  const tabs = screen.getByRole("group", { name: "Payout status" });
  expect(await within(tabs).findByRole("button", { name: /Paid 257/ })).toBeInTheDocument();
  expect(within(tabs).getByRole("button", { name: /Pending review 3/ })).toBeInTheDocument();
  view.unmount(); client.clear(); countsMissing = true;
  await mount();
  const plain = screen.getByRole("group", { name: "Payout status" });
  expect(within(plain).getByRole("button", { name: "Pending review" })).toBeInTheDocument();
  expect(within(plain).getByRole("button", { name: "Paid" })).toBeInTheDocument();
});
it("switching tab starts the new tab at its first page", async () => {
  const user = userEvent.setup(); rows = [...many(60, "pending_review", "p"), ...many(60, "held", "h")];
  await mount();
  await user.click(screen.getByRole("button", { name: "Load 10 more" }));
  await waitFor(() => expect(screen.getAllByRole("checkbox")).toHaveLength(60));
  await user.click(within(screen.getByRole("group", { name: "Payout status" })).getByRole("button", { name: /^Held/ }));
  await waitFor(() => expect(screen.getByRole("checkbox", { name: "Select payout h-1" })).toBeInTheDocument());
  expect(screen.getAllByRole("checkbox")).toHaveLength(50);
  const last = statusParam(listCalls().at(-1)!);
  expect([last.get("status"), last.get("offset")]).toEqual(["held", "0"]);
});
it("approving a payout while the paged tab is mounted refreshes the tab", async () => {
  const user = userEvent.setup(); rows = many(60, "pending_review", "p");
  const post = vi.spyOn(api, "post").mockImplementation(async () => {
    rows = rows.map((row) => row.id === "p-1" ? { ...row, status: "approved" as const } : row); return rows[0];
  });
  await mount();
  await user.click(screen.getByRole("button", { name: "Load 10 more" }));
  await waitFor(() => expect(screen.getAllByRole("checkbox")).toHaveLength(60));
  const before = listCalls().length;
  const row = screen.getByRole("checkbox", { name: "Select payout p-1" }).closest("li")!;
  await user.click(within(row).getByRole("button", { name: "Approve" }));
  await user.click(within(screen.getByRole("dialog", { name: "Approve payout" })).getByRole("button", { name: "Approve" }));
  await waitFor(() => expect(screen.queryByRole("checkbox", { name: "Select payout p-1" })).not.toBeInTheDocument());
  expect(post).toHaveBeenCalledTimes(1);
  expect(listCalls().length).toBeGreaterThan(before);
  expect(screen.getAllByRole("checkbox")).toHaveLength(59);
  expect(useToastStore.getState().toasts.some((toast) => toast.kind === "error")).toBe(false);
});
it("a ?row= link to a payout on another tab or a later page opens its drawer", async () => {
  rows = [...many(60, "pending_review", "p"), { ...payoutCycle, id: "cycle-old", status: "paid" }];
  navigation.search = "row=cycle-old";
  await mount();
  expect(await screen.findByRole("dialog")).toHaveTextContent("cycle-old");
  expect(navigation.replace).not.toHaveBeenCalled();
});
it("a ?row= link to a payout that does not exist is cleared", async () => {
  navigation.search = "row=cycle-gone";
  await mount();
  await waitFor(() => expect(navigation.replace).toHaveBeenCalledWith(expect.not.stringContaining("row="), expect.anything()));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});
it("the search box is labelled Search payouts, and the list and counts use its text", async () => {
  rows = [payoutCycle, { ...payoutCycle, id: "cycle-2" }];
  navigation.search = "q=cycle-2";
  await mount();
  expect(screen.getByRole("searchbox", { name: "Search payouts" })).toHaveValue("cycle-2");
  expect(statusParam(listCalls().at(-1)!).get("q")).toBe("cycle-2");
  expect(vi.mocked(api.get).mock.calls.some(([path]) => String(path) === "/admin/payouts/counts?q=cycle-2")).toBe(true);
  expect(screen.queryByRole("checkbox", { name: "Select payout cycle-1" })).not.toBeInTheDocument();
  expect(screen.getByRole("checkbox", { name: "Select payout cycle-2" })).toBeInTheDocument();
});

// FT-28: the backend also matches the photographer's name, brand and handle (W28.1, W28.3).
it("the search box names what it finds, and an empty search result names the search", async () => {
  rows = [];
  navigation.search = "q=@nobody";
  await mount();
  expect(screen.getByRole("searchbox", { name: "Search payouts" })).toHaveAttribute("placeholder", "Search by payout id, name or @handle");
  expect(statusParam(listCalls().at(-1)!).get("q")).toBe("@nobody");
  expect(await screen.findByText("No payouts in this tab match this search.")).toBeInTheDocument();
});

// AX-57: drawer and detail values read as typed, wrap at 375, and link a no-handle photographer by id.
const notUpperCased = (els: HTMLElement[]) => { expect(els.length).toBeGreaterThan(0); for (const el of els) expect(el.closest(".uppercase")).toBeNull(); };
it("the drawer's subtitle wraps a long handle in its own case, and its dates are not upper-cased", async () => {
  navigation.search = `row=${payoutCycle.id}`;
  await mount();
  const drawer = await screen.findByRole("dialog");
  const handles = within(drawer).getAllByText(/@photographer/, { selector: "p" });
  notUpperCased(handles);
  for (const el of handles) expect(el.className).toMatch(/break-(words|all)/);
  notUpperCased(within(drawer).getAllByText(/Sep 10 · 2026/));
  expect(within(drawer).queryByText(/SEP 10 · 2026/)).toBeNull();
});

it("an empty Pending review tab says the admin is caught up", async () => {
  rows = [];
  await mount();
  expect(await screen.findByText("No payouts waiting. You're caught up.")).toBeInTheDocument();
  expect(screen.getByText("Waiting for a decision")).toBeInTheDocument();
});

it("loading shows skeleton rows and no Loading sentence", () => {
  nextTabList = () => new Promise(() => {});
  const { container } = render_();
  expect(container.querySelector('[aria-busy="true"] .animate-pulse')).not.toBeNull();
  expect(screen.queryByText(/Loading/)).toBeNull();
});
