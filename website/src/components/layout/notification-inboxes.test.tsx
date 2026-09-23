import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { RunnerMessage } from "@/lib/runner-messages";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { useConfirmationStore } from "@/store/confirmation-store";
import { useViewModeStore } from "@/store/view-mode-store";
import { ConfirmationOverlay } from "@/components/ui/confirmation-overlay";
import { NotificationBell } from "./notification-bell";
import { RunnerNotificationBell } from "./runner-notification-bell";

const { push } = vi.hoisted(() => ({ push: vi.fn() }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));
const message: RunnerMessage = { id: "message-1", kind: "admin_message", title: "Account update", body: "A notification from the team.", createdAt: "2026-09-10T00:00:00Z", readAt: null, sourceDecisionId: null, orderId: null };
function deferred<T>() {
  let resolve!: (value: T) => void; let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
let client: QueryClient;
let rows: RunnerMessage[];
beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  rows = [message]; useToastStore.getState().clear(); useViewModeStore.getState().reset();
  vi.spyOn(api, "getWithTotal").mockImplementation(async () => ({ data: rows, total: rows.length }));
});
afterEach(() => { cleanup(); client.clear(); useConfirmationStore.getState().close(false); useAuthStore.getState().logout(); vi.restoreAllMocks(); push.mockClear(); });
function mount(role: "PHOTOGRAPHER" | "RUNNER") {
  useAuthStore.getState().setUser({ id: "owner", role, name: "Owner", email: "owner@example.test", createdAt: "2026-09-10T00:00:00Z" });
  return render(<QueryClientProvider client={client}><NotificationBell /><RunnerNotificationBell /><ConfirmationOverlay /></QueryClientProvider>);
}
describe.each(["PHOTOGRAPHER", "RUNNER"] as const)("%s inbox controls", (role) => {
  it("keeps read-all and rows disabled until confirmation, retaining unread data on failure", async () => {
    const request = deferred<unknown>(); const patch = vi.spyOn(api, "fetch").mockReturnValueOnce(request.promise);
    const user = userEvent.setup(); mount(role);
    await user.click(await screen.findByRole("button", { name: "Inbox · 1 unread" }));
    await user.click(screen.getByRole("button", { name: "Mark all read" }));
    expect(screen.getByRole("button", { name: "Mark all read" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Account update/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Remove notification" })).toBeDisabled();
    expect(screen.getByText("1 unread · 1 total")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Mark all read" })); expect(patch).toHaveBeenCalledTimes(1);
    await act(async () => request.reject(new Error("Failed")));
    await waitFor(() => expect(screen.getByRole("button", { name: "Mark all read" })).toBeEnabled());
    expect(screen.getByText("1 unread · 1 total")).toBeVisible(); expect(useToastStore.getState().toasts[0].kind).toBe("error");
    expect(screen.getByRole("alert")).toHaveTextContent("Couldn't mark as read");
    patch.mockImplementationOnce(async () => { rows = rows.map((m) => ({ ...m, readAt: "2026-09-10T09:00:00Z" })); return { markedRead: 1 }; });
    await user.click(screen.getByRole("button", { name: "Mark all read" }));
    expect(await screen.findByRole("button", { name: "Inbox · no unread messages" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Mark all read" })).not.toBeInTheDocument();
  });
  it("confirms removal, retains the row on failure, and keeps the empty modal open after retry", async () => {
    const user = userEvent.setup(); const request = deferred<unknown>();
    const remove = vi.spyOn(api, "delete").mockReturnValueOnce(request.promise);
    mount(role); await user.click(await screen.findByRole("button", { name: "Inbox · 1 unread" }));
    await user.click(screen.getByRole("button", { name: "Remove notification" }));
    const confirmation = screen.getByRole("dialog", { name: "Remove this notification?" });
    expect(remove).not.toHaveBeenCalled();
    await user.click(within(confirmation).getByRole("button", { name: "Remove" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Remove notification" })).toBeDisabled());
    expect(screen.getByText("Account update")).toBeVisible();
    await act(async () => request.reject(new Error("Failed")));
    await waitFor(() => expect(screen.getByRole("button", { name: "Remove notification" })).toBeEnabled());
    expect(screen.getByText("1 unread · 1 total")).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent("Couldn't remove notification");
    remove.mockImplementationOnce(async () => { rows = []; return { removed: true }; });
    await user.click(screen.getByRole("button", { name: "Remove notification" }));
    await user.click(within(screen.getByRole("dialog", { name: "Remove this notification?" })).getByRole("button", { name: "Remove" }));
    expect(await screen.findByText(/No messages yet/)).toBeVisible(); expect(screen.getByRole("dialog", { name: "Inbox" })).toBeVisible();
    await user.click(within(screen.getByRole("dialog", { name: "Inbox" })).getAllByRole("button", { name: "Close" })[0]);
    expect(screen.queryByRole("button", { name: /Inbox ·/ })).not.toBeInTheDocument();
  });
  it("shows a retryable first-load error instead of hiding the bell or claiming the inbox is empty", async () => {
    vi.mocked(api.getWithTotal).mockRejectedValueOnce(new Error("Offline")).mockRejectedValueOnce(new Error("Still offline"));
    const user = userEvent.setup(); mount(role);
    await user.click(await screen.findByRole("button", { name: /Inbox ·/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't refresh your inbox");
    expect(screen.queryByText(/No messages yet/)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(screen.getByText("Account update")).toBeVisible());
    expect(screen.queryByText(/No messages yet/)).not.toBeInTheDocument();
  });
  it("keeps existing messages during a refresh failure and offers Retry", async () => {
    const user = userEvent.setup(); mount(role);
    await user.click(await screen.findByRole("button", { name: /Inbox ·/ }));
    vi.mocked(api.getWithTotal).mockRejectedValueOnce(new Error("Offline"));
    await act(async () => { await client.invalidateQueries(); });
    expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't refresh your inbox");
    expect(screen.getByText("Account update")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });
  it("Escape cancels only the removal confirmation and leaves the inbox open", async () => {
    const user = userEvent.setup(); const remove = vi.spyOn(api, "delete");
    mount(role); await user.click(await screen.findByRole("button", { name: /Inbox ·/ }));
    await user.click(screen.getByRole("button", { name: "Remove notification" }));
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Remove this notification?" })).not.toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Inbox" })).toBeVisible(); expect(remove).not.toHaveBeenCalled();
  });
});
it("runner receipt navigation waits for mark-read confirmation and remains retryable", async () => {
  rows = [{ ...message, orderId: "order /1", kind: "dispute_resolved" }];
  const user = userEvent.setup(); const request = deferred<unknown>();
  const patch = vi.spyOn(api, "fetch").mockReturnValueOnce(request.promise);
  mount("RUNNER"); await user.click(await screen.findByRole("button", { name: /Inbox ·/ }));
  await user.click(screen.getByRole("button", { name: "View receipt →" }));
  expect(push).not.toHaveBeenCalled(); expect(screen.getByRole("button", { name: "View receipt →" })).toBeDisabled();
  await act(async () => request.reject(new Error("Failed")));
  await waitFor(() => expect(screen.getByRole("button", { name: "View receipt →" })).toBeEnabled());
  expect(push).not.toHaveBeenCalled(); expect(screen.getByRole("dialog", { name: "Inbox" })).toBeVisible();
  patch.mockImplementationOnce(async () => { rows = [{ ...rows[0], readAt: "2026-09-10T09:00:00Z" }]; return rows[0]; });
  await user.click(screen.getByRole("button", { name: /Account update/ }));
  await waitFor(() => expect(push).toHaveBeenCalledWith("/orders?expand=order%20%2F1"));
  expect(screen.queryByRole("dialog", { name: "Inbox" })).not.toBeInTheDocument();
});
