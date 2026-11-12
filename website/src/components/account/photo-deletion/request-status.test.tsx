import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api";
import * as removal from "@/lib/api-photo-removal";
import { RUNNER_REMOVAL_STATUS_LABEL } from "@/lib/runner-removal-status";
import { useAuthStore } from "@/store/auth-store";
import { useConfirmationStore } from "@/store/confirmation-store";
import { ConfirmationOverlay } from "@/components/ui/confirmation-overlay";
import type { RunnerRemovalRequestDetailDto } from "@/types/photo-removal";
import { RequestStatus } from "./request-status";

vi.mock("@/lib/api-photo-removal", async (original) => ({
  ...(await original<typeof import("@/lib/api-photo-removal")>()),
  getRunnerRemovalRequest: vi.fn(),
  withdrawRunnerRemoval: vi.fn(),
}));
const api = vi.mocked(removal);

const DETAIL: RunnerRemovalRequestDetailDto = {
  id: "req-1", eventId: "ev-1", eventName: "Cebu Marathon", status: "PENDING_ADMIN_REVIEW",
  createdAt: "2026-11-12T02:00:00Z", updatedAt: "2026-11-12T02:00:00Z", version: 3, canWithdraw: true,
  counts: { hidden: 4, suggested: 2, removed: 0, kept: 0 },
  timeline: [{ at: "2026-11-12T02:00:00Z", step: "SUBMITTED" }, { at: "2026-11-12T02:00:05Z", step: "PHOTOS_HIDDEN" }],
};

let client: QueryClient;
beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  useAuthStore.getState().setUser({ id: "runner-a", role: "RUNNER", name: "A", email: "a@example.test", createdAt: "2026-09-10" });
  api.getRunnerRemovalRequest.mockResolvedValue(DETAIL);
});
afterEach(() => {
  cleanup(); client.clear();
  useConfirmationStore.getState().close(useConfirmationStore.getState().active?.id ?? "", false);
  useAuthStore.getState().logout(); vi.clearAllMocks();
});
const mount = () => render(<QueryClientProvider client={client}><RequestStatus requestId="req-1" /><ConfirmationOverlay /></QueryClientProvider>);

describe("runner deletion request status", () => {
  it("the detail shows counts and no images", async () => {
    const { container } = mount();
    expect(await screen.findByText("4 photos hidden · 2 more waiting for review · 0 deleted")).toBeVisible();
    expect(screen.getByText("Cebu Marathon")).toBeVisible();
    expect(screen.getByText("Pending admin review")).toBeVisible();
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).not.toContain("req-1");
  });

  it("the detail says later uploads are not included", async () => {
    mount();
    expect(await screen.findByText(/Photos uploaded after your request are not included\. You can send a new request once this one closes\./)).toBeVisible();
  });

  it("withdraw asks for confirmation and sends expectedVersion", async () => {
    const withdrawn = { ...DETAIL, status: "WITHDRAWN" as const, canWithdraw: false, version: 4 };
    api.withdrawRunnerRemoval.mockResolvedValue(withdrawn);
    const user = userEvent.setup(); mount();
    await user.click(await screen.findByRole("button", { name: "Withdraw request" }));
    expect(api.withdrawRunnerRemoval).not.toHaveBeenCalled();
    api.getRunnerRemovalRequest.mockResolvedValue(withdrawn); // the refresh after the withdraw
    await user.click(within(screen.getByRole("dialog", { name: "Withdraw this request?" })).getByRole("button", { name: "Withdraw" }));
    await waitFor(() => expect(api.withdrawRunnerRemoval).toHaveBeenCalledWith("req-1", 3, expect.any(String)));
    expect(await screen.findByText("Withdrawn")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Withdraw request" })).not.toBeInTheDocument();
  });

  it("withdraw is hidden when canWithdraw is false", async () => {
    api.getRunnerRemovalRequest.mockResolvedValue({ ...DETAIL, status: "APPROVED", canWithdraw: false });
    mount();
    expect(await screen.findByText("Approved: deleting")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Withdraw request" })).not.toBeInTheDocument();
  });

  it("a 409 reloads the request", async () => {
    api.withdrawRunnerRemoval.mockRejectedValue(new ApiError([{ code: "VERSION_CONFLICT", message: "changed" }], 409));
    const user = userEvent.setup(); mount();
    await user.click(await screen.findByRole("button", { name: "Withdraw request" }));
    api.getRunnerRemovalRequest.mockResolvedValue({ ...DETAIL, status: "APPROVED", canWithdraw: false, version: 5 });
    await act(async () => {
      await user.click(within(screen.getByRole("dialog", { name: "Withdraw this request?" })).getByRole("button", { name: "Withdraw" }));
    });
    expect(await screen.findByText("Approved: deleting")).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent("This request changed while you were looking. Here is where it stands now.");
    expect(api.getRunnerRemovalRequest).toHaveBeenCalledTimes(2);
  });

  it("after a 409 the runner can withdraw again from the reloaded request", async () => {
    api.withdrawRunnerRemoval.mockRejectedValueOnce(new ApiError([{ code: "VERSION_CONFLICT", message: "changed" }], 409));
    const user = userEvent.setup(); mount();
    await user.click(await screen.findByRole("button", { name: "Withdraw request" }));
    api.getRunnerRemovalRequest.mockResolvedValue({ ...DETAIL, version: 5 });
    await user.click(within(screen.getByRole("dialog", { name: "Withdraw this request?" })).getByRole("button", { name: "Withdraw" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("This request changed");
    api.withdrawRunnerRemoval.mockResolvedValueOnce({ ...DETAIL, status: "WITHDRAWN", canWithdraw: false, version: 6 });
    await user.click(screen.getByRole("button", { name: "Withdraw request" }));
    await user.click(within(screen.getByRole("dialog", { name: "Withdraw this request?" })).getByRole("button", { name: "Withdraw" }));
    await waitFor(() => expect(api.withdrawRunnerRemoval).toHaveBeenLastCalledWith("req-1", 5, expect.any(String)));
    expect(api.withdrawRunnerRemoval).toHaveBeenCalledTimes(2);
    expect(api.withdrawRunnerRemoval.mock.calls[0][2]).not.toBe(api.withdrawRunnerRemoval.mock.calls[1][2]);
  });

  it("every status code has a label", () => {
    const codes = ["PROCESSING", "PENDING_ADMIN_REVIEW", "ADDITIONAL_VERIFICATION_REQUIRED", "NO_MATCHES_FOUND",
      "APPROVED", "REJECTED", "COMPLETED", "WITHDRAWN"] as const; // backend RunnerRemovalStatus
    for (const code of codes) expect(RUNNER_REMOVAL_STATUS_LABEL[code]).toMatch(/^[A-Z][a-z]/);
    expect(Object.keys(RUNNER_REMOVAL_STATUS_LABEL)).toHaveLength(codes.length);
  });

  it("a rejected request says the runner can reply by email", async () => {
    api.getRunnerRemovalRequest.mockResolvedValue({ ...DETAIL, status: "REJECTED", canWithdraw: false });
    mount();
    expect(await screen.findByText("You can reply by email within 14 days.")).toBeVisible();
  });
});
