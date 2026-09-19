import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, setStepUpHandler } from "@/lib/api";
import type { AdminPayoutCycle } from "@/lib/admin-payouts";
import { adminPayoutKeys } from "@/hooks/use-admin-payouts";
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
const key = adminPayoutKeys.list(admin.id);
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function mount() {
  return render(<QueryClientProvider client={client}><PayoutsQueue /><StepUpProvider /></QueryClientProvider>);
}
function seed(status: AdminPayoutCycle["status"] = "pending_review") {
  rows = [{ ...payoutCycle, status }]; client.setQueryData(key, rows);
}
beforeEach(() => {
  useAuthStore.getState().setUser(admin);
  useToastStore.getState().clear();
  navigation.search = ""; navigation.replace.mockClear();
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  vi.spyOn(console, "error").mockImplementation(() => {});
  vi.spyOn(api, "get").mockImplementation(async () => ({ items: rows, total: rows.length }));
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }); seed();
});
afterEach(() => {
  cleanup(); client.clear(); useAuthStore.getState().logout(); setStepUpHandler(null);
  Reflect.deleteProperty(HTMLElement.prototype, "scrollIntoView");
  vi.restoreAllMocks(); vi.unstubAllGlobals();
});

it.each([false, true])("keeps the payout URL selection until its read finishes (stale cache: %s)", async (stale) => {
  navigation.search = `row=${payoutCycle.id}`;
  if (stale) client.setQueryData(key, [], { updatedAt: 1 });
  else client.removeQueries({ queryKey: key });
  const request = deferred<{ items: AdminPayoutCycle[] }>();
  vi.mocked(api.get).mockReturnValueOnce(request.promise);
  mount();
  expect(navigation.replace).not.toHaveBeenCalled();
  await act(async () => { request.resolve({ items: rows }); });
  expect(await screen.findByRole("dialog")).toHaveTextContent(payoutCycle.id);
  expect(navigation.replace).not.toHaveBeenCalled();
});

it("renders a retryable read error instead of dropping the selected payout", async () => {
  navigation.search = `row=${payoutCycle.id}`;
  client.removeQueries({ queryKey: key });
  vi.mocked(api.get).mockRejectedValueOnce(new Error("Unavailable"));
  mount();
  expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't load payouts");
  expect(navigation.replace).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByRole("dialog")).toHaveTextContent(payoutCycle.id);
});

it("blocks repeated row/drawer approvals while pending and keeps the first approval pending", async () => {
  const user = userEvent.setup();
  const request = deferred<AdminPayoutCycle>();
  const post = vi.spyOn(api, "post").mockReturnValueOnce(request.promise);
  mount();
  await user.click(screen.getByRole("button", { name: `Open payout ${payoutCycle.id}` }));
  const drawer = screen.getByRole("dialog");
  const approve = within(drawer).getByRole("button", { name: "Approve" });
  await user.click(approve);
  expect(approve).toBeDisabled();
  expect(screen.getAllByRole("button", { name: "Approve" }).every((button) => button.hasAttribute("disabled"))).toBe(true);
  await user.keyboard("e"); await user.click(approve);
  expect(post).toHaveBeenCalledTimes(1);
  expect(useToastStore.getState().toasts).toHaveLength(0);
  rows = [{ ...payoutCycle, approvedBy: admin.id }];
  await act(async () => { request.resolve(rows[0]); });
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  expect(client.getQueryData(key)).toEqual(rows);
  expect(useToastStore.getState().toasts.at(-1)?.message).toContain("Approval recorded");
  await user.click(screen.getByRole("button", { name: `Open payout ${payoutCycle.id}` }));
  expect(screen.getByRole("dialog")).toHaveTextContent("1 of 2");
  expect(screen.queryByRole("button", { name: "Mark paid…" })).not.toBeInTheDocument();
  // A later server refresh, not a local override, unlocks sending.
  rows = [{ ...rows[0], status: "approved", secondApprovedBy: "admin-b", reviewedAt: "2026-09-10T05:00:00Z" }];
  await act(async () => { await client.invalidateQueries({ queryKey: key }); });
  await waitFor(() => expect(within(screen.getByRole("dialog")).getByRole("button", { name: "Mark paid…" })).toBeEnabled());
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
  mount();
  await user.click(screen.getByRole("button", { name: `Open payout ${payoutCycle.id}` }));
  await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: trigger }));
  const form = screen.getAllByRole("dialog").find((dialog) => within(dialog).queryByRole("textbox"))!;
  const input = within(form).getByRole("textbox");
  await user.type(input, "Reference or note retained");
  await user.click(within(form).getByRole("button", { name: submitLabel }));
  expect(within(form).getByRole("button", { name: "Saving…" })).toBeDisabled();
  expect(input).toBeDisabled();
  await user.keyboard("{Escape}");
  expect(form).toBeInTheDocument();
  expect(client.getQueryData(key)).toEqual(rows);
  await act(async () => { request.reject(new Error("Unavailable")); });
  expect(input).toHaveValue("Reference or note retained");
  expect(within(form).getByRole("button", { name: submitLabel })).toBeEnabled();
  expect(useToastStore.getState().toasts.at(-1)?.kind).toBe("error");
  post.mockImplementationOnce(async () => {
    rows = [{ ...rows[0], status: finalStatus }]; return rows[0];
  });
  await user.click(within(form).getByRole("button", { name: submitLabel }));
  await waitFor(() => expect(form).not.toBeInTheDocument());
  expect(post).toHaveBeenCalledTimes(2);
  expect(post.mock.calls[0][0]).toBe(`/admin/payouts/${payoutCycle.id}/${action}`);
  expect(client.getQueryData(key)).toEqual(rows);
});

it.each(["approve", "hold"])("keeps bulk %s pending through partial-result reconciliation", async (action) => {
  const user = userEvent.setup();
  rows = [payoutCycle, { ...payoutCycle, id: "cycle-2" }]; client.setQueryData(key, rows);
  const request = deferred<unknown>();
  const post = vi.spyOn(api, "post").mockReturnValueOnce(request.promise);
  mount();
  for (const row of rows) await user.click(screen.getByRole("checkbox", { name: `Select payout ${row.id}` }));
  await user.click(screen.getByRole("button", { name: action === "approve" ? "Approve 2" : "Hold 2…" }));
  const modal = screen.getByRole("dialog");
  const submit = action === "approve" ? "Confirm approve" : "Hold 2 payouts";
  await user.click(within(modal).getByRole("button", { name: submit }));
  expect(within(modal).getByRole("button", { name: "Saving…" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Clear" })).toBeDisabled();
  await act(async () => { request.resolve({ groupId: "partial", results: [
    { id: payoutCycle.id, ok: true }, { id: "cycle-2", ok: false },
  ] }); });
  expect(modal).toBeInTheDocument();
  const retryLabel = action === "approve" ? "Confirm approve" : "Hold payout";
  expect(within(modal).getByRole("button", { name: retryLabel })).toBeEnabled();
  expect(screen.getByRole("checkbox", { name: `Select payout ${payoutCycle.id}` })).not.toBeChecked();
  expect(screen.getByRole("checkbox", { name: "Select payout cycle-2" })).toBeChecked();
  expect(useToastStore.getState().toasts.at(-1)?.message).toContain("1 succeeded, 1 failed");
  expect(useToastStore.getState().toasts.some((toast) => toast.kind === "success")).toBe(false);
  post.mockResolvedValueOnce({ groupId: "retry", results: [{ id: "cycle-2", ok: true }] });
  await user.click(within(modal).getByRole("button", { name: retryLabel }));
  await waitFor(() => expect(modal).not.toBeInTheDocument());
  expect(post).toHaveBeenLastCalledWith("/admin/payouts/bulk", expect.objectContaining({ ids: ["cycle-2"] }), expect.anything());
});

it.each(["approve", "hold"])("closes bulk %s and clears selection only after confirmation", async (action) => {
  const user = userEvent.setup();
  const request = deferred<unknown>();
  vi.spyOn(api, "post").mockReturnValueOnce(request.promise);
  mount();
  await user.click(screen.getByRole("checkbox", { name: `Select payout ${payoutCycle.id}` }));
  await user.click(screen.getByRole("button", { name: action === "approve" ? "Approve 1" : "Hold 1…" }));
  const modal = screen.getByRole("dialog");
  await user.click(within(modal).getByRole("button", { name: action === "approve" ? "Confirm approve" : "Hold payout" }));
  expect(modal).toBeInTheDocument();
  expect(screen.getByRole("checkbox")).toBeChecked();
  rows = [{ ...payoutCycle, status: action === "hold" ? "held" : "pending_review", approvedBy: admin.id }];
  await act(async () => { request.resolve({ groupId: "batch", results: [{ id: payoutCycle.id, ok: true }] }); });
  await waitFor(() => expect(modal).not.toBeInTheDocument());
  expect(screen.getByRole("checkbox")).not.toBeChecked();
  expect(client.getQueryData(key)).toEqual(rows);
  expect(useToastStore.getState().toasts.at(-1)?.message).toContain(action === "approve" ? "Approvals recorded" : "Held");
});

it("preserves provider-controlled action guards, including drawer keyboard shortcuts", async () => {
  const user = userEvent.setup(); rows = [{ ...payoutCycle, status: "processing", provider: "paymongo" }];
  client.setQueryData(key, rows); const post = vi.spyOn(api, "post"); mount();
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
  vi.stubGlobal("fetch", fetch); mount();
  await user.click(screen.getByRole("button", { name: "Mark paid…" }));
  const form = screen.getByRole("dialog");
  await user.type(within(form).getByRole("textbox"), "GC-REFERENCE");
  await user.click(within(form).getByRole("button", { name: "Mark paid" }));
  let stepUp = await screen.findByRole("dialog", { name: "Confirm it's you" });
  expect(within(form).getByRole("button", { name: "Saving…" })).toBeDisabled();
  await user.keyboard("{Escape}");
  await waitFor(() => expect(stepUp).not.toBeInTheDocument());
  expect(within(form).getByRole("textbox")).toHaveValue("GC-REFERENCE");
  expect(client.getQueryData(key)).toEqual(rows);
  await user.click(within(form).getByRole("button", { name: "Mark paid" }));
  stepUp = await screen.findByRole("dialog", { name: "Confirm it's you" });
  await user.type(within(stepUp).getByLabelText("Password"), "wrong");
  await user.click(within(stepUp).getByRole("button", { name: "Confirm" }));
  expect(await within(stepUp).findByText("Incorrect password")).toBeInTheDocument();
  expect(within(form).getByRole("button", { name: "Saving…" })).toBeDisabled();
  await user.clear(within(stepUp).getByLabelText("Password"));
  await user.type(within(stepUp).getByLabelText("Password"), "fixture-password");
  await user.click(within(stepUp).getByRole("button", { name: "Confirm" }));
  const retried = fetch.mock.calls.filter(([, opts]) => new Headers(opts.headers).get("X-Step-Up"));
  expect(retried).toHaveLength(1);
  expect(JSON.parse(retried[0][1].body as string)).toEqual({ paymentReference: "GC-REFERENCE" });
  expect(within(form).getByRole("button", { name: "Saving…" })).toBeDisabled();
  rows = [{ ...rows[0], status: "paid", paymentReference: "GC-REFERENCE" }];
  await act(async () => { proofRequest.resolve(new Response(JSON.stringify({ success: true, data: rows[0] }))); });
  await waitFor(() => expect(form).not.toBeInTheDocument());
  expect(client.getQueryData(key)).toEqual(rows);
  expect(useAuthStore.getState().isAuthenticated).toBe(true);
  expect(useToastStore.getState().toasts.at(-1)?.message).toContain("Paid");
});
