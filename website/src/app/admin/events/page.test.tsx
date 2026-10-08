import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { ConfirmationOverlay } from "@/components/ui/confirmation-overlay";
import { useConfirmationStore } from "@/store/confirmation-store";
import { useToastStore } from "@/store/toast-store";
import { useAuthStore } from "@/store/auth-store";
import { admin, adminEvent } from "@/test/fixtures/admin";
import { adminEventKeys } from "@/hooks/use-admin-events";
import AdminEventsPage from "./page";

let client: QueryClient;
let rows = [adminEvent];
const key = adminEventKeys.list(admin.id);
function mount() { render(<QueryClientProvider client={client}><AdminEventsPage /><ConfirmationOverlay /></QueryClientProvider>); }
function deferred() {
  let resolve!: (value: unknown) => void; let reject!: (error: Error) => void;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
beforeEach(() => {
  useAuthStore.getState().setUser(admin); rows = [adminEvent];
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }); client.setQueryData(key, rows);
  vi.spyOn(api, "get").mockImplementation(async () => ({ items: rows }));
});
afterEach(() => {
  cleanup(); client.clear(); useAuthStore.getState().logout();
  useConfirmationStore.getState().close(useConfirmationStore.getState().active?.id ?? "", false); useToastStore.getState().clear(); vi.restoreAllMocks();
});
it("waits in delete confirmation, retains the event on failure, and allows retry", async () => {
  const user = userEvent.setup(); const request = deferred();
  const remove = vi.spyOn(api, "delete").mockReturnValueOnce(request.promise); mount();
  await user.click(screen.getByRole("button", { name: /delete/i }));
  const dialog = screen.getByRole("dialog", { name: "Delete event?" });
  const submit = within(dialog).getByRole("button", { name: "Delete event" });
  await user.click(submit); await user.click(submit);
  expect(submit).toBeDisabled(); expect(remove).toHaveBeenCalledTimes(1);
  expect(client.getQueryData(key)).toEqual([adminEvent]); expect(useToastStore.getState().toasts).toEqual([]);
  await act(async () => { request.reject(new Error("Event is still in use.")); });
  expect(dialog).toBeInTheDocument(); expect(within(dialog).getByRole("alert")).toHaveTextContent("Event is still in use.");
  expect(submit).toBeEnabled(); expect(useToastStore.getState().toasts.map((t) => t.kind)).toEqual(["error"]);
  remove.mockImplementationOnce(async () => { rows = []; return { removed: true }; });
  await user.click(submit); await waitFor(() => expect(dialog).not.toBeInTheDocument());
  expect(screen.queryByText("Cebu Race")).not.toBeInTheDocument(); expect(client.getQueryData(key)).toEqual([]);
  expect(useToastStore.getState().toasts.map((t) => t.kind)).toEqual(["error", "info"]);
});
it.each(["create", "edit"])("retains %s form input and canonical rows after failure, then retries", async (mode) => {
  const user = userEvent.setup(); const request = deferred();
  const mutation = vi.spyOn(api, mode === "create" ? "post" : "fetch").mockReturnValueOnce(request.promise); mount();
  await user.click(screen.getByRole("button", { name: mode === "create" ? /new event/i : "Edit" }));
  const dialog = screen.getByRole("dialog", { name: mode === "create" ? "Create event" : "Edit event" });
  const title = within(dialog).getByLabelText("Title"); await user.clear(title); await user.type(title, "Draft title");
  if (mode === "create") {
    await user.type(within(dialog).getByLabelText("Location"), "Cebu");
    fireEvent.change(within(dialog).getByLabelText("Date"), { target: { value: adminEvent.date } });
    await user.clear(within(dialog).getByLabelText("Price per photo · PHP"));
    await user.type(within(dialog).getByLabelText("Price per photo · PHP"), "80");
  }
  const submitLabel = mode === "create" ? "Create event" : "Save changes";
  await user.click(within(dialog).getByRole("button", { name: submitLabel }));
  expect(title).toBeDisabled(); expect(within(dialog).getByRole("button", { name: "Cancel" })).toBeDisabled();
  await user.keyboard("{Escape}{Enter}"); expect(mutation).toHaveBeenCalledTimes(1);
  expect(client.getQueryData(key)).toEqual([adminEvent]);
  await act(async () => { request.reject(new Error("Try again")); });
  expect(title).toHaveValue("Draft title"); expect(title).toBeEnabled(); expect(dialog).toBeInTheDocument();
  const confirmed = { ...adminEvent, name: "Backend title", slug: "backend-title" };
  mutation.mockImplementationOnce(async () => { rows = [confirmed]; return confirmed; });
  await user.click(within(dialog).getByRole("button", { name: submitLabel }));
  await waitFor(() => expect(dialog).not.toBeInTheDocument());
  expect(client.getQueryData(key)).toEqual([confirmed]);
  expect(screen.getByRole("link", { name: /view/i })).toHaveAttribute("href", "/events/backend-title");
  if (mode === "edit") {
    const form = (mutation.mock.calls[0][1] as RequestInit).body as FormData;
    expect(form.get("title")).toBe("Draft title"); expect(form.has("pricePerPhoto")).toBe(false);
  }
});
it("shows loading and a retryable query error instead of an empty catalog", async () => {
  client.clear(); vi.mocked(api.get).mockRejectedValueOnce(new Error("Offline")); mount();
  expect(screen.getByRole("status")).toHaveTextContent("Loading events");
  await screen.findByRole("alert"); expect(screen.queryByText("No live events right now.")).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Retry" }));
  await screen.findByRole("button", { name: "Edit" });
});
