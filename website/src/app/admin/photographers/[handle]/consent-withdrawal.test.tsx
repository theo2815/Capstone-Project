import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { AdminActionAside } from "@/components/admin/admin-action-aside";
import { adminUserKeys, useAdminUserDetail } from "@/hooks/use-admin-users";
import { api, ApiError } from "@/lib/api";
import { clearRelevanceConsentWithdrawal, recordRelevanceConsentWithdrawal, type AdminUserDetail } from "@/lib/api-admin";
import type { AdminUserRow } from "@/lib/admin-user-registry";
import { admin, adminUser } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";

// RV-L7 part c: the admin records and clears a photographer's withdrawn consent to automatic actions.
const PATH = "/admin/relevance/photographers/photographer-1/consent-withdrawal";
const WITHDRAWN_AT = "2026-11-12T03:00:00Z";
let client: QueryClient, detail: AdminUserDetail;
function deferred<T>() {
  let resolve!: (value: T) => void; let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject };
}
// The page passes the detail query's data as the row, so the aside re-renders from the cache.
function LiveAside() { const query = useAdminUserDetail(adminUser.userId); return query.data ? <AdminActionAside row={query.data} /> : null; }
function mount(row?: AdminUserRow) {
  return render(<QueryClientProvider client={client}>{row ? <AdminActionAside row={row} /> : <LiveAside />}</QueryClientProvider>);
}
const section = () => screen.getByRole("region", { name: "Automatic photo checks" });
beforeEach(() => {
  useAuthStore.getState().setUser(admin); useToastStore.getState().clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  detail = { ...adminUser, decisionLog: [], relevanceConsentWithdrawnAt: null };
  client.setQueryData(adminUserKeys.detail(admin.id, adminUser.userId), detail);
  vi.spyOn(api, "get").mockImplementation(async () => detail);
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

it("the control shows only for a photographer", () => {
  mount({ ...adminUser, role: "RUNNER" });
  expect(screen.queryByRole("region", { name: "Automatic photo checks" })).not.toBeInTheDocument();
  cleanup();
  mount({ ...adminUser, relevanceConsentWithdrawnAt: null });
  expect(section()).toBeInTheDocument();
});

it("a photographer who has not withdrawn shows the Record withdrawal action", () => {
  mount();
  expect(within(section()).getByText("Consent to automatic hides and payout holds: not withdrawn.")).toBeInTheDocument();
  expect(within(section()).getByText("Use this only when the photographer has emailed to withdraw consent.")).toBeInTheDocument();
  expect(within(section()).getByRole("button", { name: "Record withdrawal" })).toBeEnabled();
  expect(within(section()).queryByRole("button", { name: "Clear withdrawal" })).not.toBeInTheDocument();
});

it("a withdrawn photographer shows the date and the Clear withdrawal action", () => {
  detail = { ...detail, relevanceConsentWithdrawnAt: WITHDRAWN_AT };
  client.setQueryData(adminUserKeys.detail(admin.id, adminUser.userId), detail);
  mount();
  const date = within(section()).getByText("NOV 12 · 2026");
  expect(date).toHaveClass("tnum", "whitespace-nowrap"); // the date never breaks inside itself
  expect(date.parentElement).toHaveTextContent(
    "Consent withdrawn on NOV 12 · 2026. Automatic hides and payout holds are off for this photographer; admins review each case.",
  );
  expect(within(section()).getByRole("button", { name: "Clear withdrawal" })).toBeEnabled();
  expect(within(section()).queryByRole("button", { name: "Record withdrawal" })).not.toBeInTheDocument();
});

it("recording asks for a confirm, calls the route once and shows the new state only after the server answers", async () => {
  const request = deferred<unknown>();
  const put = vi.spyOn(api, "put").mockReturnValue(request.promise as Promise<never>);
  const user = userEvent.setup(); mount();
  const action = within(section()).getByRole("button", { name: "Record withdrawal" });
  await user.click(action);
  const dialog = screen.getByRole("dialog", { name: "Record that this photographer withdrew consent?" });
  expect(dialog).toHaveTextContent(
    'Use this when the photographer asks by email to withdraw consent. From now on nothing of theirs is taken off sale and no payout is held automatically. Their cases are shown to admins as "would have hidden". Photos already off sale and payouts already on hold stay as they are.',
  );
  expect(put).not.toHaveBeenCalled();
  await user.click(within(dialog).getByRole("button", { name: "Record withdrawal" }));
  const busy = within(dialog).getByRole("button", { name: "Recording…" });
  expect(busy).toHaveAttribute("aria-busy", "true");
  await user.click(busy); await user.keyboard("{Escape}");
  expect(put).toHaveBeenCalledTimes(1);
  expect(put).toHaveBeenCalledWith(PATH, undefined);
  expect(dialog).toBeInTheDocument();
  expect(within(section()).getByText("Consent to automatic hides and payout holds: not withdrawn.")).toBeInTheDocument();
  detail = { ...detail, relevanceConsentWithdrawnAt: WITHDRAWN_AT };
  await act(async () => request.resolve({ photographerId: adminUser.userId, consentWithdrawnAt: WITHDRAWN_AT }));
  await waitFor(() => expect(dialog).not.toBeInTheDocument());
  expect(within(section()).getByText("NOV 12 · 2026")).toBeInTheDocument();
  expect(useToastStore.getState().toasts.map((t) => t.message)).toContain("Withdrawal recorded");
  expect(within(section()).getByRole("button", { name: "Clear withdrawal" })).toHaveFocus();
});

it("clearing asks for a confirm, calls the route once and shows the new state only after the server answers", async () => {
  detail = { ...detail, relevanceConsentWithdrawnAt: WITHDRAWN_AT };
  client.setQueryData(adminUserKeys.detail(admin.id, adminUser.userId), detail);
  const request = deferred<unknown>();
  const del = vi.spyOn(api, "delete").mockReturnValue(request.promise as Promise<never>);
  const user = userEvent.setup(); mount();
  await user.click(within(section()).getByRole("button", { name: "Clear withdrawal" }));
  const dialog = screen.getByRole("dialog", { name: "Clear the withdrawal?" });
  expect(dialog).toHaveTextContent(
    "Use this only when the photographer asks to give consent again. Automatic hides and payout holds can then apply to them again, if they accepted the current Terms.",
  );
  expect(del).not.toHaveBeenCalled();
  await user.click(within(dialog).getByRole("button", { name: "Clear withdrawal" }));
  await user.click(within(dialog).getByRole("button", { name: "Clearing…" }));
  expect(del).toHaveBeenCalledTimes(1);
  expect(del).toHaveBeenCalledWith(PATH);
  expect(within(section()).getByText("NOV 12 · 2026")).toBeInTheDocument();
  detail = { ...detail, relevanceConsentWithdrawnAt: null };
  await act(async () => request.resolve({ photographerId: adminUser.userId, consentWithdrawnAt: null }));
  await waitFor(() => expect(dialog).not.toBeInTheDocument());
  expect(within(section()).getByText("Consent to automatic hides and payout holds: not withdrawn.")).toBeInTheDocument();
  expect(useToastStore.getState().toasts.map((t) => t.message)).toContain("Withdrawal cleared");
  expect(within(section()).getByRole("button", { name: "Record withdrawal" })).toHaveFocus();
});

it("a refused action shows the server's message and leaves the state as it was", async () => {
  vi.spyOn(api, "put").mockRejectedValue(new ApiError([{ code: "USER_NOT_FOUND", message: "User not found" }], 404));
  const user = userEvent.setup(); mount();
  await user.click(within(section()).getByRole("button", { name: "Record withdrawal" }));
  const dialog = screen.getByRole("dialog");
  await user.click(within(dialog).getByRole("button", { name: "Record withdrawal" }));
  await waitFor(() => expect(useToastStore.getState().toasts.map((t) => t.message)).toContain("User not found"));
  expect(within(dialog).getByRole("button", { name: "Record withdrawal" })).toBeEnabled();
  expect(within(section()).getByText("Consent to automatic hides and payout holds: not withdrawn.")).toBeInTheDocument();
  expect(useToastStore.getState().toasts.map((t) => t.message)).not.toContain("Withdrawal recorded");
  await user.keyboard("{Escape}");
  expect(dialog).not.toBeInTheDocument();
  expect(within(section()).getByRole("button", { name: "Record withdrawal" })).toHaveFocus();
});

it("another admin action does not make the withdrawal read as not withdrawn", async () => {
  detail = { ...detail, relevanceConsentWithdrawnAt: WITHDRAWN_AT, suspendedAt: "2026-11-01T00:00:00Z", suspensionReason: "Compliance review requested" };
  client.setQueryData(adminUserKeys.detail(admin.id, adminUser.userId), detail);
  // The unsuspend answer is a list row, which never carries the field; the refetch it starts does not answer here.
  const post = vi.spyOn(api, "post").mockResolvedValue({ ...adminUser, suspendedAt: null, suspensionReason: null });
  vi.spyOn(api, "get").mockReturnValue(new Promise<never>(() => {}));
  const user = userEvent.setup(); mount();
  await user.click(screen.getByRole("button", { name: "Unsuspend" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Suspend…" })).toBeInTheDocument());
  expect(post).toHaveBeenCalledTimes(1);
  expect(within(section()).getByText("NOV 12 · 2026")).toBeInTheDocument();
  expect(within(section()).queryByText("Consent to automatic hides and payout holds: not withdrawn.")).not.toBeInTheDocument();
});

it("the other admin actions are locked while the withdrawal call runs", async () => {
  const request = deferred<unknown>();
  vi.spyOn(api, "put").mockReturnValue(request.promise as Promise<never>);
  const user = userEvent.setup(); mount();
  await user.click(within(section()).getByRole("button", { name: "Record withdrawal" }));
  await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Record withdrawal" }));
  const panel = screen.getByRole("complementary", { name: "Admin actions" });
  const buttons = within(panel).getAllByRole("button");
  expect(buttons.map((b) => b.textContent)).toEqual(expect.arrayContaining(["Send back…", "Suspend…", "Reset verification…", "Send message…"]));
  expect(buttons.filter((b) => !b.hasAttribute("disabled")).map((b) => b.textContent)).toEqual([]);
  await act(async () => request.resolve({ photographerId: adminUser.userId, consentWithdrawnAt: WITHDRAWN_AT }));
  await waitFor(() => expect(within(panel).getByRole("button", { name: "Suspend…" })).toBeEnabled());
});

it("the two api functions call the routes part a built", async () => {
  const answer = { photographerId: "p 1", consentWithdrawnAt: WITHDRAWN_AT };
  const put = vi.spyOn(api, "put").mockResolvedValue(answer);
  const del = vi.spyOn(api, "delete").mockResolvedValue({ ...answer, consentWithdrawnAt: null });
  expect(await recordRelevanceConsentWithdrawal("p 1")).toEqual(answer);
  expect(put).toHaveBeenCalledWith("/admin/relevance/photographers/p%201/consent-withdrawal", undefined);
  expect(await clearRelevanceConsentWithdrawal("p 1")).toEqual({ ...answer, consentWithdrawnAt: null });
  expect(del).toHaveBeenCalledWith("/admin/relevance/photographers/p%201/consent-withdrawal");
});
