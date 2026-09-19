import { act, cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { admin, report } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { payoutReportKeys } from "@/hooks/use-payout-reports";
import { FilePayoutReportModal } from "./file-payout-report-modal";

afterEach(() => {
  cleanup();
  useAuthStore.getState().logout();
  useToastStore.getState().clear();
  vi.restoreAllMocks();
});

it("keeps a failed report retryable and only adds the backend-issued report after confirmation", async () => {
  const user = userEvent.setup();
  useAuthStore.getState().setUser({ ...admin, id: report.photographerId, role: "PHOTOGRAPHER" });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const key = payoutReportKeys.photographerList(report.photographerId);
  client.setQueryData(key, []);
  const onClose = vi.fn();
  vi.spyOn(console, "error").mockImplementation(() => {});
  let reject!: (error: Error) => void;
  vi.spyOn(api, "post").mockReturnValueOnce(new Promise((_, fail) => { reject = fail; }));
  render(<QueryClientProvider client={client}><FilePayoutReportModal cycle={{
    id: report.payoutCycleId, weekOf: "2026-09-07", requestedAt: "2026-09-09T00:00:00Z",
    amount: 100, status: "paid", settledAt: "2026-09-10T00:00:00Z", method: "gcash", reference: "reference-1",
  }} onClose={onClose} /></QueryClientProvider>);
  await user.click(screen.getByRole("button", { name: "Report reason" }));
  await user.click(screen.getByRole("menuitem", { name: "Payout not received" }));
  const dialog = screen.getByRole("dialog", { name: "File a report" });
  const note = within(dialog).getByRole("textbox");
  await user.type(note, "Please check the transfer.");
  const send = within(dialog).getByRole("button", { name: "Send report" });
  await user.click(send);
  await user.click(send);
  await user.keyboard("{Escape}");
  expect(api.post).toHaveBeenCalledTimes(1);
  expect(send).toBeDisabled();
  expect(note).toBeDisabled();
  expect(client.getQueryData(key)).toEqual([]);
  expect(onClose).not.toHaveBeenCalled();
  expect(useToastStore.getState().toasts).toEqual([]);

  await act(async () => { reject(new Error("Backend unavailable")); });
  expect(send).toBeEnabled();
  expect(note).toHaveValue("Please check the transfer.");
  expect(client.getQueryData(key)).toEqual([]);
  expect(onClose).not.toHaveBeenCalled();
  expect(useToastStore.getState().toasts.map((toast) => toast.kind)).toEqual(["error"]);

  vi.mocked(api.post).mockResolvedValueOnce(report);
  await user.click(send);
  expect(api.post).toHaveBeenLastCalledWith(`/me/photographer/payouts/${report.payoutCycleId}/report`, {
    reason: "not_received", note: "Please check the transfer.",
  });
  expect(client.getQueryData(key)).toEqual([report]);
  expect(onClose).toHaveBeenCalledTimes(1);
  expect(useToastStore.getState().toasts.map((toast) => toast.kind)).toEqual(["error", "success"]);
  client.clear();
});
