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
  const user = userEvent.setup(); await user.click(screen.getByRole("button", { name: "Reject…" }));
  const dialog = screen.getByRole("dialog");
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
  await user.click(await screen.findByRole("button", { name: surface === "bulk" ? "Reject 1…" : "Reject…" }));
  const submit = screen.getByRole("button", { name: "Send back" });
  await user.click(submit);
  await waitFor(() => expect(client.getQueryData(key)).toEqual(rows));
  await waitFor(() => expect(submit).toBeEnabled());
  await user.click(submit);
  await waitFor(() => expect(post).toHaveBeenCalledTimes(2));
  expect(post.mock.calls.map((call) => call[1])).toEqual([expect.objectContaining({ expectedRevision: 7 }), expect.objectContaining({ expectedRevision: 7 })]);
});
it.each([
  ["Reject…", "Send back", "reject"], ["Suspend…", "Suspend", "suspend"], ["Reset verification…", "Reset verification", "reset-verification"],
])("%s preserves reason on failure and prevents repeat submission", async (openLabel, submitLabel, endpoint) => {
  const request = deferred<unknown>(); const post = vi.spyOn(api, "post").mockReturnValueOnce(request.promise); const user = userEvent.setup(); mount();
  await user.click(screen.getByRole("button", { name: openLabel })); const dialog = screen.getByRole("dialog");
  const note = within(dialog).getByLabelText("Note (optional, max 280)"); await user.type(note, "Keep this reason");
  await user.click(within(dialog).getByRole("button", { name: submitLabel }));
  expect(note).toBeDisabled(); await user.keyboard("{Escape}{Enter}"); expect(post).toHaveBeenCalledTimes(1); expect(dialog).toBeInTheDocument();
  expect(useToastStore.getState().toasts).toEqual([]);
  await act(async () => request.reject(new Error("Unavailable")));
  expect(note).toHaveValue("Keep this reason"); await waitFor(() => expect(note).toBeEnabled()); expect(dialog).toBeInTheDocument();
  post.mockResolvedValueOnce(adminUser); await user.click(within(dialog).getByRole("button", { name: submitLabel }));
  await waitFor(() => expect(dialog).not.toBeInTheDocument());
  expect(post).toHaveBeenLastCalledWith(`/admin/users/${adminUser.userId}/${endpoint}`, { reason: expect.stringContaining("Keep this reason"), ...(endpoint === "suspend" ? {} : { expectedRevision: 7 }) });
});
it.each([false, true])("awaits approval/unsuspension and locks the action cluster (suspended=%s)", async (suspended) => {
  if (suspended) rows = [{ ...adminUser, suspendedAt: "2026-09-10T00:00:00Z" }];
  const request = deferred<unknown>(); const post = vi.spyOn(api, "post").mockReturnValue(request.promise); mount();
  const submit = screen.getByRole("button", { name: suspended ? "Unsuspend" : "Approve" });
  await userEvent.click(submit); await userEvent.click(submit);
  expect(screen.getAllByRole("button").every((button) => button.hasAttribute("disabled"))).toBe(true);
  expect(post).toHaveBeenCalledTimes(1); expect(useToastStore.getState().toasts).toEqual([]);
  await act(async () => request.resolve(adminUser));
  expect(useToastStore.getState().toasts[0].kind).toBe("success");
});
it("preserves a deep-linked verification through loading, failure, and retry", async () => {
  navigation.search = `row=${adminUser.userId}`; client.setQueryData(key, []);
  const request = deferred<unknown>(); vi.mocked(api.get).mockReturnValueOnce(request.promise); mount(true);
  expect(navigation.replace).not.toHaveBeenCalled(); expect(screen.getByRole("status")).toHaveTextContent("Loading verification");
  await act(async () => request.reject(new Error("Unavailable")));
  expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't load verification"); expect(navigation.replace).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("button", { name: "Try again" }));
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
  await user.click(screen.getByRole("button", { name: "Reject 2…" })); const dialog = screen.getByRole("dialog"); const note = within(dialog).getByLabelText("Note (optional, max 280)");
  await user.type(note, "Fix watermark"); await user.click(within(dialog).getByRole("button", { name: "Send back" }));
  expect(note).toBeDisabled(); await user.keyboard("{Enter}"); expect(post).toHaveBeenCalledTimes(2);
  await act(async () => request.reject(new Error("Unavailable")));
  expect(note).toHaveValue("Fix watermark"); expect(dialog).toHaveTextContent("1 photographer will see");
  post.mockImplementationOnce(async () => { rows = rows.map((row) => ({ ...row, verificationStatus: "incomplete" })); return rows[1]; });
  await user.click(within(dialog).getByRole("button", { name: "Send back" })); await waitFor(() => expect(dialog).not.toBeInTheDocument());
  expect(post).toHaveBeenCalledTimes(3); expect(post.mock.calls[2][0]).toContain("photographer-2/reject");
});
