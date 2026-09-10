import { act, cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { ConfirmationOverlay } from "@/components/ui/confirmation-overlay";
import { useConfirmationStore } from "@/store/confirmation-store";
import { useAdminEventOverridesStore } from "@/store/admin-event-overrides-store";
import { useToastStore } from "@/store/toast-store";
import AdminEventsPage from "./page";

vi.mock("@/hooks/use-admin-data", () => ({
  useAdminEvents: () => [{
    id: "event-1", slug: "cebu-race", name: "Cebu Race", date: "2026-09-10",
    city: "Cebu", location: "Cebu", state: "live", status: "ACTIVE",
    photoCount: 10, participantCount: 20, createdByHandle: null, createdByName: null,
  }],
}));

afterEach(() => {
  cleanup();
  useAdminEventOverridesStore.getState().clear();
  useConfirmationStore.getState().close(false);
  useToastStore.getState().clear();
  vi.restoreAllMocks();
});

it("waits in the delete confirmation, retains the event on failure, and allows retry", async () => {
  const user = userEvent.setup();
  const client = new QueryClient();
  vi.spyOn(console, "error").mockImplementation(() => {});
  let reject!: (error: Error) => void;
  vi.spyOn(api, "delete").mockReturnValueOnce(new Promise((_, fail) => { reject = fail; }));
  render(<QueryClientProvider client={client}><AdminEventsPage /><ConfirmationOverlay /></QueryClientProvider>);

  await user.click(screen.getByRole("button", { name: /delete/i }));
  const dialog = screen.getByRole("dialog", { name: "Delete event?" });
  const submit = within(dialog).getByRole("button", { name: "Delete event" });
  await user.click(submit);
  expect(submit).toBeDisabled();
  expect(useAdminEventOverridesStore.getState().tombstones).toEqual([]);
  expect(useToastStore.getState().toasts).toEqual([]);

  await act(async () => { reject(new Error("Event is still in use.")); });
  expect(dialog).toBeInTheDocument();
  expect(within(dialog).getByRole("alert")).toHaveTextContent("Event is still in use.");
  expect(submit).toBeEnabled();
  expect(useToastStore.getState().toasts.map((toast) => toast.kind)).toEqual(["error"]);

  vi.mocked(api.delete).mockResolvedValueOnce({});
  await user.click(submit);
  expect(dialog).not.toBeInTheDocument();
  expect(screen.queryByText("Cebu Race")).not.toBeInTheDocument();
  expect(useAdminEventOverridesStore.getState().tombstones).toEqual(["event-1"]);
  expect(useToastStore.getState().toasts.map((toast) => toast.kind)).toEqual(["error", "info"]);
  client.clear();
});
