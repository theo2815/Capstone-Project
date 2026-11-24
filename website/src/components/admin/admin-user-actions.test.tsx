import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { webcrypto } from "node:crypto";
import { clearAdminActionIntents } from "@/lib/admin-action-intents";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { admin, adminUser, adminUserSettings } from "@/test/fixtures/admin";
import { adminUserKeys, useAdminUsers } from "@/hooks/use-admin-users";
import type { AdminUserRow } from "@/lib/admin-user-registry";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { AdminActionAside } from "./admin-action-aside";
import { VerificationsQueue } from "./verifications-queue";
const navigation = vi.hoisted(() => ({ search: "", replace: vi.fn() }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace: navigation.replace }), usePathname: () => "/admin/inbox", useSearchParams: () => new URLSearchParams(navigation.search) }));
let client: QueryClient, rows: AdminUserRow[];
const key = adminUserKeys.list(admin.id);
function deferred<T>() {
  let resolve!: (value: T) => void; let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject };
}
function mount(queue = false) { return render(<QueryClientProvider client={client}>{queue ? <VerificationsQueue /> : <AdminActionAside row={rows[0]} />}</QueryClientProvider>); }
beforeEach(() => {
  vi.stubGlobal("crypto", webcrypto); clearAdminActionIntents();
  useAuthStore.getState().setUser(admin); useToastStore.getState().clear(); navigation.search = ""; navigation.replace.mockClear();
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }); rows = [adminUser]; client.setQueryData(key, rows);
  vi.spyOn(api, "get").mockImplementation(async (path) => path.endsWith("/settings") ? adminUserSettings : path === "/regions" ? [] : path.startsWith("/admin/users?") ? { items: rows, total: rows.length } : { ...rows[0], decisionLog: [] });
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); clearAdminActionIntents(); vi.unstubAllGlobals(); vi.restoreAllMocks(); Reflect.deleteProperty(HTMLElement.prototype, "scrollIntoView"); });
it("keeps a delayed message open and its draft intact through repeat input and failure", async () => {
  const request = deferred<unknown>();
  const post = vi.spyOn(api, "post").mockReturnValueOnce(request.promise);
  const user = userEvent.setup(); mount();
  await user.click(screen.getByRole("button", { name: "Send message…" }));
  const dialog = screen.getByRole("dialog");
  const subject = within(dialog).getByLabelText("Subject");
  const body = within(dialog).getByLabelText("Message");
  await user.type(subject, "Review question"); await user.type(body, "Keep this message");
  await user.click(within(dialog).getByRole("button", { name: "Send" }));
  expect(dialog).toBeInTheDocument();
  expect(subject).toHaveValue("Review question"); expect(subject).toBeDisabled();
  expect(within(dialog).getByRole("button", { name: "Sending…" })).toHaveAttribute("aria-busy", "true");
  await user.keyboard("{Enter}{Escape}");
  await user.click(screen.getByRole("button", { name: "Send message…" }));
  expect(post).toHaveBeenCalledTimes(1);
  await act(async () => request.reject(new Error("Connection lost")));
  expect(body).toHaveValue("Keep this message"); await waitFor(() => expect(subject).toBeEnabled());
  post.mockResolvedValueOnce({ id: "message-1" });
  await user.click(within(dialog).getByRole("button", { name: "Send" }));
  await waitFor(() => expect(dialog).not.toBeInTheDocument());
  expect(post).toHaveBeenCalledTimes(2);
});
it("does not rebase an open rejection onto a replacement verification after a revision conflict", async () => {
  function LiveAside() { const query = useAdminUsers(); return query.data?.[0] ? <AdminActionAside row={query.data[0]} /> : null; }
  const post = vi.spyOn(api, "post").mockImplementation(async () => {
    rows = [{ ...adminUser, verificationRevision: 8 }];
    throw new ApiError([{ code: "INVALID_STATE_TRANSITION", field: "expectedRevision", message: "Refresh before deciding" }], 409);
  });
  render(<QueryClientProvider client={client}><LiveAside /></QueryClientProvider>);
  const user = userEvent.setup(); await user.click(screen.getByRole("button", { name: "Send back…" }));
  const dialog = screen.getByRole("dialog"); await user.click(within(dialog).getAllByRole("radio")[0]!); 
  await user.click(within(dialog).getByRole("button", { name: "Send back" }));
  await waitFor(() => expect(client.getQueryData(key)).toEqual(rows));
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Send back" })).toBeEnabled());
  await user.click(within(dialog).getByRole("button", { name: "Send back" }));
  await waitFor(() => expect(post).toHaveBeenCalledTimes(2));
  expect(post.mock.calls.map((call) => call[1])).toEqual([expect.objectContaining({ expectedRevision: 7 }), expect.objectContaining({ expectedRevision: 7 })]);
  expect(useToastStore.getState().toasts.some((toast) => toast.kind === "success")).toBe(false);
});
it.each(["drawer", "bulk"])("keeps the reviewed revision in an open %s rejection after conflict", async (surface) => {
  if (surface === "drawer") navigation.search = `row=${adminUser.userId}`;
  const post = vi.spyOn(api, "post").mockImplementation(async () => {
    rows = [{ ...adminUser, verificationRevision: 8 }];
    throw new ApiError([{ code: "INVALID_STATE_TRANSITION", field: "expectedRevision", message: "Review again" }], 409);
  });
  mount(true); const user = userEvent.setup();
  if (surface === "bulk") await user.click(screen.getByRole("checkbox", { name: "Select Test Studio" }));
  await user.click(await screen.findByRole("button", { name: surface === "bulk" ? "Send back 1…" : "Send back…" }));
  const submit = screen.getByRole("button", { name: "Send back" }); const dialog = screen.getAllByRole("dialog").at(-1)!; await user.click(within(dialog).getAllByRole("radio")[0]!); 
  await user.click(submit);
  await waitFor(() => expect(client.getQueryData(key)).toEqual(rows));
  await waitFor(() => expect(submit).toBeEnabled()); expect(within(dialog).getByRole("alert")).toHaveTextContent(/Couldn.t finish/); 
  await user.click(submit);
  await waitFor(() => expect(post).toHaveBeenCalledTimes(2));
  expect(post.mock.calls.map((call) => call[1])).toEqual([expect.objectContaining({ expectedRevision: 7 }), expect.objectContaining({ expectedRevision: 7 })]);
});
it.each([
  ["Send back…", "Send back", "reject"], ["Suspend…", "Suspend", "suspend"], ["Reset verification…", "Reset verification", "reset-verification"],
])("%s preserves reason on failure and prevents repeat submission", async (openLabel, submitLabel, endpoint) => {
  const request = deferred<unknown>(); const post = vi.spyOn(api, "post").mockReturnValueOnce(request.promise); const user = userEvent.setup(); mount();
  await user.click(screen.getByRole("button", { name: openLabel })); const dialog = screen.getByRole("dialog");
  await user.click(within(dialog).getAllByRole("radio")[0]!); const note = within(dialog).getByLabelText("Note (optional, max 280)"); await user.type(note, "Keep this reason");
  await user.click(within(dialog).getByRole("button", { name: submitLabel }));
  expect(note).toBeDisabled(); await user.keyboard("{Escape}{Enter}"); expect(post).toHaveBeenCalledTimes(1); expect(dialog).toBeInTheDocument();
  expect(useToastStore.getState().toasts).toEqual([]);
  await act(async () => request.reject(new Error("Unavailable")));
  expect(note).toHaveValue("Keep this reason"); await waitFor(() => expect(note).toBeEnabled()); expect(dialog).toBeInTheDocument(); expect(within(dialog).getByRole("alert")).toHaveTextContent(/Couldn.t finish/); 
  post.mockResolvedValueOnce(adminUser); await user.click(within(dialog).getByRole("button", { name: submitLabel }));
  await waitFor(() => expect(dialog).not.toBeInTheDocument());
  expect(post).toHaveBeenLastCalledWith(`/admin/users/${adminUser.userId}/${endpoint}`, { reason: expect.stringContaining("Keep this reason"), ...(endpoint === "suspend" ? {} : { expectedRevision: 7 }) });
});
it("the photographer Suspend dialog says nobody is emailed and payouts are not stopped (FT-12)", async () => {
  const post = vi.spyOn(api, "post").mockResolvedValue(adminUser); const user = userEvent.setup(); mount();
  await user.click(screen.getByRole("button", { name: "Suspend…" })); const dialog = screen.getByRole("dialog", { name: "Suspend account" });
  expect(dialog).toHaveTextContent(`${adminUser.brandName ?? adminUser.name} is signed out shortly and cannot sign in until you unsuspend. Their photos come off sale and their public page stops opening; photos already bought can still be downloaded. Payouts are not stopped: hold those on the Payouts page. They are not emailed and cannot sign in to read the reason, so tell them yourself if they should know.`);
  expect(within(dialog).getByLabelText("Many payment disputes from buyers")).toBeInTheDocument();
  expect(within(dialog).getByPlaceholderText("Add detail. They can read this once they are unsuspended.")).toBeInTheDocument();
  await user.click(within(dialog).getByLabelText("Compliance review requested"));
  await user.click(within(dialog).getByRole("button", { name: "Suspend" }));
  await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(useToastStore.getState().toasts[0]).toMatchObject({ kind: "success", message: `${adminUser.brandName ?? adminUser.name} is suspended. They were not emailed.` }));
});
it("awaits unsuspension and locks the action cluster", async () => {
  rows = [{ ...adminUser, suspendedAt: "2026-09-10T00:00:00Z" }];
  const request = deferred<unknown>(); const post = vi.spyOn(api, "post").mockReturnValue(request.promise); mount();
  const submit = screen.getByRole("button", { name: "Unsuspend" });
  await userEvent.click(submit); await userEvent.click(submit);
  expect(screen.getAllByRole("button").every((button) => button.hasAttribute("disabled"))).toBe(true);
  expect(post).toHaveBeenCalledTimes(1); expect(useToastStore.getState().toasts).toEqual([]);
  await act(async () => request.resolve(adminUser));
  expect(useToastStore.getState().toasts[0]).toMatchObject({ kind: "success", message: `${adminUser.brandName ?? adminUser.name} can sign in again. They are told in their inbox.` });
});
it("preserves a deep-linked verification through loading, failure, and retry", async () => {
  navigation.search = `row=${adminUser.userId}`; client.setQueryData(key, []);
  const request = deferred<unknown>(); vi.mocked(api.get).mockReturnValueOnce(request.promise); mount(true);
  expect(navigation.replace).not.toHaveBeenCalled(); expect(document.querySelector('[aria-busy="true"] .animate-pulse')).not.toBeNull(); expect(screen.queryByText(/Loading/)).toBeNull();
  await act(async () => request.reject(new Error("Unavailable")));
  expect(within(await screen.findByRole("alert")).getByText("Couldn't load verification.")).toBeInTheDocument(); expect(navigation.replace).not.toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Retry" })).toHaveAttribute("type", "button");
  await userEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByRole("dialog")).toHaveTextContent("Test Studio"); expect(navigation.replace).not.toHaveBeenCalled();
});
it("retains only failed bulk rejections, keeps their reason, and retries them once", async () => {
  rows = [adminUser, { ...adminUser, userId: "photographer-2", brandName: "Second Studio" }]; client.setQueryData(key, rows);
  const request = deferred<unknown>(); const user = userEvent.setup();
  const post = vi.spyOn(api, "post").mockImplementation(async (path) => {
    if (path.includes("photographer-2")) return request.promise;
    rows = rows.map((row) => row.userId === adminUser.userId ? { ...row, verificationStatus: "incomplete" } : row); return rows[0];
  });
  mount(true); await user.click(screen.getByRole("checkbox", { name: "Select Test Studio" })); await user.click(screen.getByRole("checkbox", { name: "Select Second Studio" }));
  await user.click(screen.getByRole("button", { name: "Send back 2…" })); const dialog = screen.getByRole("dialog"); const note = within(dialog).getByLabelText("Note (optional, max 280)");
  await user.click(within(dialog).getAllByRole("radio")[0]!); await user.type(note, "Fix watermark"); await user.click(within(dialog).getByRole("button", { name: "Send back" }));
  expect(note).toBeDisabled(); await user.keyboard("{Enter}"); expect(post).toHaveBeenCalledTimes(2);
  await act(async () => request.reject(new Error("Unavailable")));
  expect(note).toHaveValue("Fix watermark"); expect(dialog).toHaveTextContent("1 photographer gets your reason"); expect(within(dialog).getByRole("alert")).toHaveTextContent(/Couldn.t finish/); 
  post.mockImplementationOnce(async () => { rows = rows.map((row) => ({ ...row, verificationStatus: "incomplete" })); return rows[1]; });
  await user.click(within(dialog).getByRole("button", { name: "Send back" })); await waitFor(() => expect(dialog).not.toBeInTheDocument());
  expect(post).toHaveBeenCalledTimes(3); expect(post.mock.calls[2][0]).toContain("photographer-2/reject");
});
it("names the pending bulk send back and blocks repeat Enter until it settles", async () => {
  const request = deferred<unknown>(); const post = vi.spyOn(api, "post").mockReturnValue(request.promise);
  mount(true); const user = userEvent.setup();
  await user.click(screen.getByRole("checkbox", { name: "Select Test Studio" }));
  await user.click(screen.getByRole("button", { name: "Send back 1…" })); const dialog = screen.getByRole("dialog"); await user.click(within(dialog).getAllByRole("radio")[0]!); 
  await user.click(within(dialog).getByRole("button", { name: "Send back" }));
  expect(within(screen.getByRole("dialog")).getByRole("button", { name: "Sending back…" })).toHaveAttribute("aria-busy", "true");
  expect(screen.getByRole("button", { name: "Test Studio", hidden: true }).closest("article")).toHaveAttribute("aria-busy", "true");
  await user.keyboard("{Enter}"); expect(post).toHaveBeenCalledTimes(1);
  await act(async () => { request.resolve(adminUser); });
});
// Q7 (AA-I17): profiles approve themselves once identity passes; only a held profile gets an Approve (below).
it("drawer shows no Approve and E does nothing", async () => {
  navigation.search = `row=${adminUser.userId}`;
  const post = vi.spyOn(api, "post"); mount(true); const user = userEvent.setup();
  const drawer = await screen.findByRole("dialog");
  expect(within(drawer).getByRole("button", { name: "Send back…" })).toBeInTheDocument();
  expect(within(drawer).queryByRole("button", { name: /approv/i })).toBeNull();
  await user.keyboard("e"); await user.keyboard("E");
  expect(post).not.toHaveBeenCalled();
});
const HOLD_LINE = "Held: it was sent back or reset before, so it needs your approval.";
it("a held profile shows the hold line and an Approve button", async () => {
  rows = [{ ...adminUser, adminHold: true }]; client.setQueryData(key, rows); navigation.search = `row=${adminUser.userId}`;
  // An older backend's detail answer has no adminHold: the hold comes from the list row.
  const listGet = vi.mocked(api.get).getMockImplementation()!;
  vi.mocked(api.get).mockImplementation(async (path) => path === `/admin/users/${adminUser.userId}` ? { ...adminUser, decisionLog: [] } : listGet(path));
  mount(true);
  const drawer = await screen.findByRole("dialog");
  await waitFor(() => expect(api.get).toHaveBeenCalledWith(`/admin/users/${adminUser.userId}`)); await act(async () => {});
  expect(within(drawer).getByText(HOLD_LINE)).toBeInTheDocument();
  expect(within(drawer).getByRole("button", { name: "Approve…" })).toBeInTheDocument();
});
it("a held profile shows Approve from the detail answer when the list has no row for it", async () => {
  rows = []; client.setQueryData(key, rows); navigation.search = `row=${adminUser.userId}`;
  const listGet = vi.mocked(api.get).getMockImplementation()!;
  vi.mocked(api.get).mockImplementation(async (path) => path === `/admin/users/${adminUser.userId}` ? { ...adminUser, adminHold: true, decisionLog: [] } : listGet(path));
  mount(true);
  const drawer = await screen.findByRole("dialog");
  expect(await within(drawer).findByRole("button", { name: "Approve…" })).toBeInTheDocument();
  expect(within(drawer).getByText(HOLD_LINE)).toBeInTheDocument();
});
it("the drawer subtitle has no empty place between two dots when region and city are blank", async () => {
  rows = [{ ...adminUser, region: null, city: "" }]; client.setQueryData(key, rows); navigation.search = `row=${adminUser.userId}`;
  mount(true);
  const drawer = await screen.findByRole("dialog");
  expect(await within(drawer).findByText(/^@test-studio/)).not.toHaveTextContent(/·\s*·/);
});
it("the list row has no leading dot when region and city are blank", async () => {
  rows = [{ ...adminUser, region: null, city: "" }]; client.setQueryData(key, rows);
  mount(true);
  // The details line now reads in sentence case: "joined Sep 2026 · …".
  const line = (await screen.findByText(/^joined/)).parentElement!;
  expect(line.textContent!.trim()).toMatch(/^joined/);
});
it("the list row keeps the city and its dot when the region is blank", async () => {
  rows = [{ ...adminUser, region: "", city: "Cebu" }]; client.setQueryData(key, rows);
  mount(true);
  expect((await screen.findByText(/^joined/)).parentElement!.textContent!.trim()).toMatch(/^Cebu ·\s*joined/);
});
it("the detail answer's hold wins over a stale list row", async () => {
  rows = [{ ...adminUser, adminHold: false }]; client.setQueryData(key, rows); navigation.search = `row=${adminUser.userId}`;
  const listGet = vi.mocked(api.get).getMockImplementation()!;
  vi.mocked(api.get).mockImplementation(async (path) => path === `/admin/users/${adminUser.userId}` ? { ...adminUser, adminHold: true, decisionLog: [] } : listGet(path));
  mount(true);
  const drawer = await screen.findByRole("dialog");
  await waitFor(() => expect(api.get).toHaveBeenCalledWith(`/admin/users/${adminUser.userId}`)); await act(async () => {});
  expect(within(drawer).getByRole("button", { name: "Approve…" })).toBeInTheDocument();
});
it("a profile that is not held has no hold line and no Approve button", async () => {
  navigation.search = `row=${adminUser.userId}`; mount(true);
  const drawer = await screen.findByRole("dialog");
  expect(within(drawer).queryByText(HOLD_LINE)).toBeNull();
  expect(within(drawer).queryByRole("button", { name: "Approve…" })).toBeNull();
});
it("approve waits for the server before showing success and shows the server's error on a conflict", async () => {
  rows = [{ ...adminUser, adminHold: true }]; client.setQueryData(key, rows); navigation.search = `row=${adminUser.userId}`;
  // An older backend's detail answer has no adminHold: the hold comes from the list row.
  const listGet = vi.mocked(api.get).getMockImplementation()!;
  vi.mocked(api.get).mockImplementation(async (path) => path === `/admin/users/${adminUser.userId}` ? { ...adminUser, decisionLog: [] } : listGet(path));
  const request = deferred<unknown>(); const post = vi.spyOn(api, "post").mockReturnValueOnce(request.promise);
  mount(true); const user = userEvent.setup();
  const drawer = await screen.findByRole("dialog");
  await user.click(within(drawer).getByRole("button", { name: "Approve…" }));
  await user.click(within(screen.getByRole("dialog", { name: "Approve Test Studio?" })).getByRole("button", { name: "Approve" }));
  expect(post).toHaveBeenCalledTimes(1);
  expect(post).toHaveBeenCalledWith(`/admin/users/${adminUser.userId}/approve`, { expectedRevision: 7 });
  expect(useToastStore.getState().toasts).toEqual([]);
  await act(async () => request.resolve({ ...adminUser, adminHold: false }));
  expect(useToastStore.getState().toasts.map((t) => [t.kind, t.message])).toEqual([["success", "Hold released · Test Studio"]]);
  useToastStore.getState().clear();
  rows = [{ ...adminUser, adminHold: true }]; client.setQueryData(key, rows);
  post.mockRejectedValueOnce(new ApiError([{ code: "CONFLICT", field: "", message: "This profile is not on hold." }], 409));
  await user.click(await within(await screen.findByRole("dialog")).findByRole("button", { name: "Approve…" }));
  await user.click(within(screen.getByRole("dialog", { name: "Approve Test Studio?" })).getByRole("button", { name: "Approve" }));
  await waitFor(() => expect(useToastStore.getState().toasts.map((t) => [t.kind, t.message])).toEqual([["error", "This profile is not on hold."]]));
});
it("bulk bar offers only Send back", async () => {
  mount(true); const user = userEvent.setup();
  await user.click(screen.getByRole("checkbox", { name: "Select Test Studio" }));
  const bar = screen.getByRole("region", { name: "Verifications bulk action bar" });
  expect(within(bar).getByRole("button", { name: "Send back 1…" })).toBeInTheDocument();
  expect(within(bar).queryByRole("button", { name: /approv/i })).toBeNull();
});
it("aside shows Reject, Reset, Suspend but no Approve", () => {
  mount();
  for (const name of ["Send back…", "Reset verification…", "Suspend…"]) expect(screen.getByRole("button", { name })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /approv/i })).toBeNull();
});
it("open-queue rows say they wait for the identity check and link to Seller details", async () => {
  mount(true);
  const link = await screen.findByRole("link", { name: "Seller details" });
  expect(link).toHaveAttribute("href", "/admin/verifications#seller-details");
  expect(link.closest("p")).toHaveTextContent("Waits for the identity check. Decide it under Seller details.");
});
it("shows the join month for a backend timestamp instead of an invalid date", async () => {
  mount(true);
  await waitFor(() => expect(document.body).toHaveTextContent("joined Sep 2026"));
  expect(document.body).not.toHaveTextContent(/invalid date/i);
});
it("does not send back a row ticked after the bulk rejection form opened", async () => {
  rows = [adminUser, { ...adminUser, userId: "photographer-2", brandName: "Second Studio" }]; client.setQueryData(key, rows);
  const post = vi.spyOn(api, "post").mockImplementation(async () => rows[0]); const user = userEvent.setup();
  mount(true); await user.click(screen.getByRole("checkbox", { name: "Select Test Studio" }));
  await user.click(screen.getByRole("button", { name: "Send back 1…" })); const dialog = screen.getByRole("dialog");
  // Focus is not trapped in the form, so the keyboard can still tick a row whose revision was never captured.
  await user.click(screen.getByRole("checkbox", { name: "Select Second Studio", hidden: true })); await user.click(within(dialog).getAllByRole("radio")[0]!); 
  await user.click(within(dialog).getByRole("button", { name: "Send back" }));
  await waitFor(() => expect(dialog).not.toBeInTheDocument());
  expect(post.mock.calls.map((call) => call[0])).toEqual([`/admin/users/${adminUser.userId}/reject`]);
});
async function lateTick(user: ReturnType<typeof userEvent.setup>) {
  rows = [adminUser, { ...adminUser, userId: "photographer-2", brandName: "Second Studio" }]; client.setQueryData(key, rows);
  mount(true); await user.click(screen.getByRole("checkbox", { name: "Select Test Studio" }));
  await user.click(screen.getByRole("button", { name: "Send back 1…" })); const dialog = screen.getByRole("dialog");
  await user.click(screen.getByRole("checkbox", { name: "Select Second Studio", hidden: true }));
  return dialog;
}
it("counts in the bulk rejection form only the rows that will be sent back", async () => {
  expect(await lateTick(userEvent.setup())).toHaveTextContent("1 photographer gets your reason");
});
it("says so when no ticked row can be sent back", async () => {
  const post = vi.spyOn(api, "post").mockImplementation(async () => rows[0]); const user = userEvent.setup();
  const dialog = await lateTick(user);
  await user.click(screen.getByRole("checkbox", { name: "Select Test Studio", hidden: true })); await user.click(within(dialog).getAllByRole("radio")[0]!); 
  await user.click(within(dialog).getByRole("button", { name: "Send back" }));
  expect(useToastStore.getState().toasts).toEqual([expect.objectContaining({ kind: "error",
    message: "These rows changed after this form opened. Close the form and tick them again." })]);
  expect(post).not.toHaveBeenCalled(); expect(dialog).toBeInTheDocument();
});
// FT-13 (C-02 to C-05, C-09, B-35): one verb, a confirm before a held approve, the Held pill, the "already decided" toast.
const toasts = () => useToastStore.getState().toasts.map((t) => [t.kind, t.message]);
it("FT-13: the drawer, the bulk bar and the photographer page all say Send back", async () => {
  navigation.search = `row=${adminUser.userId}`; mount(true); const user = userEvent.setup();
  const drawer = await screen.findByRole("dialog");
  expect(within(drawer).getByRole("button", { name: "Send back…" })).toBeInTheDocument();
  cleanup(); navigation.search = ""; mount(true);
  await user.click(screen.getByRole("checkbox", { name: "Select Test Studio" }));
  expect(screen.getByRole("button", { name: "Send back 1…" })).toBeInTheDocument();
  cleanup(); mount();
  expect(screen.getByRole("button", { name: "Send back…" })).toBeInTheDocument();
});
it("FT-13: Approve on a held profile asks first and calls the server only from the confirm", async () => {
  rows = [{ ...adminUser, adminHold: true }]; client.setQueryData(key, rows); navigation.search = `row=${adminUser.userId}`;
  const post = vi.spyOn(api, "post").mockResolvedValue({ ...adminUser, adminHold: false });
  mount(true); const user = userEvent.setup();
  await user.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Approve…" }));
  const confirm = screen.getByRole("dialog", { name: "Approve Test Studio?" });
  expect(confirm).toHaveTextContent("This releases the hold. If their ID check has passed, the profile goes live now and they are told. If it has not, the profile goes live by itself when the check passes.");
  expect(post).not.toHaveBeenCalled();
  await user.click(within(confirm).getByRole("button", { name: "Approve" }));
  expect(post).toHaveBeenCalledWith(`/admin/users/${adminUser.userId}/approve`, { expectedRevision: 7 });
  await waitFor(() => expect(toasts()).toEqual([["success", "Hold released · Test Studio"]]));
});
it("FT-13: a held row says Held on the list, and the drawer says why", async () => {
  rows = [{ ...adminUser, adminHold: true }, { ...adminUser, userId: "photographer-2", brandName: "Second Studio" }]; client.setQueryData(key, rows);
  mount(true);
  const held = screen.getByRole("button", { name: "Test Studio" }).closest("article")!;
  expect(within(held).getByText("Held")).toBeInTheDocument();
  expect(within(screen.getByRole("button", { name: "Second Studio" }).closest("article")!).queryByText("Held")).toBeNull();
  cleanup(); navigation.search = `row=${adminUser.userId}`; mount(true);
  expect(within(await screen.findByRole("dialog")).getByText("Held: it was sent back or reset before, so it needs your approval.")).toBeInTheDocument();
});
it("FT-13: an open row decided by someone else closes with a toast; this admin's own approve does not", async () => {
  navigation.search = `row=${adminUser.userId}`; mount(true);
  await screen.findByRole("dialog"); await act(async () => {});
  rows = [{ ...adminUser, verificationStatus: "approved" }];
  await act(async () => { await client.invalidateQueries(); });
  await waitFor(() => expect(toasts()).toEqual([["info", "Test Studio was already decided by someone else."]]));
  cleanup(); useToastStore.getState().clear(); client.clear();
  rows = [{ ...adminUser, adminHold: true }]; client.setQueryData(key, rows);
  vi.spyOn(api, "post").mockImplementation(async () => { rows = [{ ...adminUser, verificationStatus: "approved", adminHold: false }]; return rows[0]; });
  mount(true); const user = userEvent.setup();
  await user.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Approve…" }));
  await user.click(within(screen.getByRole("dialog", { name: "Approve Test Studio?" })).getByRole("button", { name: "Approve" }));
  await waitFor(() => expect(toasts()).toEqual([["success", "Hold released · Test Studio"]]));
  await act(async () => { await client.invalidateQueries(); });
  expect(toasts()).toEqual([["success", "Hold released · Test Studio"]]);
});
it("FT-13: a partly failed bulk send back names the rows still ticked", async () => {
  rows = [adminUser, { ...adminUser, userId: "photographer-2", brandName: "Second Studio" }]; client.setQueryData(key, rows);
  vi.spyOn(api, "post").mockImplementation(async (path) => { if (path.includes("photographer-2")) throw new Error("Unavailable"); return rows[0]; });
  mount(true); const user = userEvent.setup();
  await user.click(screen.getByRole("checkbox", { name: "Select all" }));
  await user.click(screen.getByRole("button", { name: "Send back 2…" })); const dialog = screen.getByRole("dialog");
  await user.click(within(dialog).getAllByRole("radio")[0]!); await user.click(within(dialog).getByRole("button", { name: "Send back" }));
  await waitFor(() => expect(toasts()).toContainEqual(["error", "1 of 2 sent back. Not sent back, still ticked: Second Studio."]));
});
it("FT-13: when the row is decided elsewhere under the approve confirm, the confirm does not come back by itself", async () => {
  rows = [{ ...adminUser, adminHold: true }]; client.setQueryData(key, rows); navigation.search = `row=${adminUser.userId}`;
  mount(true); const user = userEvent.setup();
  await user.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Approve…" }));
  expect(screen.getByRole("dialog", { name: "Approve Test Studio?" })).toBeInTheDocument();
  rows = [{ ...adminUser, verificationStatus: "approved", adminHold: false }];
  await act(async () => { await client.invalidateQueries(); });
  await waitFor(() => expect(screen.queryByRole("dialog", { name: "Approve Test Studio?" })).toBeNull());
  // The same row comes back held (sent back and sent again): opening it shows the drawer, not the old confirm.
  rows = [{ ...adminUser, adminHold: true }];
  await act(async () => { await client.invalidateQueries(); });
  await user.click(await screen.findByRole("button", { name: "Test Studio" }));
  expect(await within(await screen.findByRole("dialog")).findByRole("button", { name: "Approve…" })).toBeInTheDocument();
  expect(screen.queryByRole("dialog", { name: "Approve Test Studio?" })).toBeNull();
});
// FT-35 (C-62): Send back first, the green Approve last; both open a dialog, so both end in "…".
it("a held profile's drawer buttons read Send back, then Approve", async () => {
  rows = [{ ...adminUser, adminHold: true }]; client.setQueryData(key, rows); navigation.search = `row=${adminUser.userId}`;
  mount(true);
  const drawer = await screen.findByRole("dialog");
  const footer = drawer.querySelector("footer")!;
  expect(within(footer).getAllByRole("button").map((b) => b.textContent)).toEqual(["Send back…", "Approve…"]);
});
