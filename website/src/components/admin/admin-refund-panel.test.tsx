import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import * as utils from "@/lib/utils";
import type { Dispute } from "@/lib/admin-disputes";
import { adminDisputeKeys } from "@/hooks/use-admin-disputes";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { admin, dispute } from "@/test/fixtures/admin";
import { formatLongDate, formatSalesDate } from "@/lib/format";
import { AdminDisputeActionAside } from "./admin-dispute-action-aside";
import { AdminRefundConfirmModal } from "./admin-refund-confirm-modal";
import { DisputesQueue } from "./disputes-queue";

const fixtures = vi.hoisted(() => ({ search: "" }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
  usePathname: () => "/admin/inbox",
  useSearchParams: () => new URLSearchParams(fixtures.search),
}));

const parked: Dispute = {
  ...dispute,
  status: "escalated",
  resolution: "refund_full",
  refundAmount: 100,
  refundStatus: "manual_review",
  paymentId: "pay_abc123",
  paymentRef: "cs_xyz",
  orderSnapshot: { ...dispute.orderSnapshot, paymentMethod: "qrph" },
};

let client: QueryClient;
beforeEach(() => {
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  fixtures.search = "";
  useAuthStore.getState().setUser(admin);
  useToastStore.getState().clear();
  vi.spyOn(api, "get").mockResolvedValue({ items: [parked], total: 1 });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(adminDisputeKeys.list(admin.id), [parked]);
});
afterEach(() => {
  cleanup();
  client.clear();
  useAuthStore.getState().logout();
  Reflect.deleteProperty(HTMLElement.prototype, "scrollIntoView");
  vi.restoreAllMocks();
});

function renderAside(d: Dispute) {
  return render(
    <QueryClientProvider client={client}>
      <AdminDisputeActionAside dispute={d} runnerHandle={d.runnerHandle} />
    </QueryClientProvider>,
  );
}
const lastToast = () => useToastStore.getState().toasts.at(-1);

describe("manual refund panel", () => {
  it("shows the payment id with a Copy button and copies it", async () => {
    const user = userEvent.setup();
    const copy = vi.spyOn(utils, "copyToClipboard").mockResolvedValue(true);
    renderAside(parked);
    expect(screen.getByText("pay_abc123")).toBeInTheDocument();
    expect(screen.queryByText("Review the payment in PayMongo before another refund attempt.")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Copy" }));
    expect(copy).toHaveBeenCalledWith("pay_abc123");
    expect(lastToast()?.kind).toBe("success");
  });

  it("tells the admin when the copy fails", async () => {
    const user = userEvent.setup();
    vi.spyOn(utils, "copyToClipboard").mockResolvedValue(false);
    renderAside(parked);
    await user.click(screen.getByRole("button", { name: "Copy" }));
    expect(lastToast()?.kind).toBe("error");
  });

  it("falls back to the checkout reference, amount and paid time when there is no payment id", () => {
    renderAside({ ...parked, paymentId: null });
    expect(screen.queryByRole("button", { name: "Copy" })).not.toBeInTheDocument();
    expect(screen.getByText("cs_xyz")).toBeInTheDocument();
    expect(screen.getByText("₱100.00")).toBeInTheDocument();
    expect(screen.getByText(/Search these in PayMongo\./)).toBeInTheDocument();
    expect(screen.getByTestId("refund-paid-at")).not.toHaveTextContent("—");
  });

  it("shows the cancel date sentence only when both dates are set", () => {
    const { unmount } = renderAside({ ...parked, refundWindowClosesAt: "2026-11-20T00:00:00Z", refundCancelsAt: "2026-11-27T00:00:00Z" });
    const closes = formatLongDate("2026-11-20T00:00:00Z");
    const cancels = formatLongDate("2026-11-27T00:00:00Z");
    expect(screen.getByText(`PayMongo window closes ${closes}. QuickPitik cancels this refund on ${cancels} if it is not done.`)).toBeInTheDocument();
    unmount();
    renderAside({ ...parked, refundWindowClosesAt: "2026-11-20T00:00:00Z", refundCancelsAt: null });
    expect(screen.queryByText(/PayMongo window closes/)).not.toBeInTheDocument();
  });

  it("sends the trimmed link, then toasts success", async () => {
    const user = userEvent.setup();
    const post = vi.spyOn(api, "post").mockResolvedValueOnce({ ...parked, refundLinkUrl: "https://pay.example/r/1", refundLinkSentAt: "2026-10-29T01:00:00Z" });
    renderAside(parked);
    const send = screen.getByRole("button", { name: "Send to runner" });
    expect(send).toBeDisabled();
    await user.type(screen.getByLabelText("Refund link"), "  https://pay.example/r/1  ");
    await user.click(send);
    expect(post).toHaveBeenCalledWith(`/admin/disputes/${parked.id}/refund-link`, { url: "https://pay.example/r/1" });
    expect(lastToast()?.kind).toBe("success");
  });

  it("keeps the typed link and shows no success toast when the send fails", async () => {
    const user = userEvent.setup();
    vi.spyOn(api, "post").mockRejectedValueOnce(new Error("Backend unavailable"));
    renderAside(parked);
    await user.type(screen.getByLabelText("Refund link"), "https://pay.example/r/1");
    await user.click(screen.getByRole("button", { name: "Send to runner" }));
    expect(screen.getByLabelText("Refund link")).toHaveValue("https://pay.example/r/1");
    expect(useToastStore.getState().toasts.map((t) => t.kind)).toEqual(["error"]);
  });

  it("shows the saved link as text with a Send a new link button", async () => {
    const user = userEvent.setup();
    renderAside({ ...parked, refundLinkUrl: "https://pay.example/r/1", refundLinkSentAt: "2026-10-29T01:00:00Z" });
    expect(screen.getByText("https://pay.example/r/1")).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Refund link")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Send a new link" }));
    expect(screen.getByLabelText("Refund link")).toBeInTheDocument();
  });

  it("shows the saved link, its sent time and Send a new link after a successful send", async () => {
    const user = userEvent.setup();
    const sent = { ...parked, refundLinkUrl: "https://pay.example/r/1", refundLinkSentAt: "2026-10-29T01:00:00Z" };
    vi.spyOn(api, "post").mockResolvedValueOnce(sent);
    const view = renderAside(parked);
    await user.type(screen.getByLabelText("Refund link"), "https://pay.example/r/1");
    await user.click(screen.getByRole("button", { name: "Send to runner" }));
    view.rerender(
      <QueryClientProvider client={client}>
        <AdminDisputeActionAside dispute={sent} runnerHandle={sent.runnerHandle} />
      </QueryClientProvider>,
    );
    expect(screen.getByText("https://pay.example/r/1")).toBeInTheDocument();
    expect(screen.getByText(formatSalesDate(sent.refundLinkSentAt))).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send a new link" })).toBeInTheDocument();
    expect(screen.queryByLabelText("Refund link")).not.toBeInTheDocument();
  });

  it("keeps the confirm open and shows no success toast when Mark refunded fails", async () => {
    const user = userEvent.setup();
    vi.spyOn(api, "post").mockRejectedValueOnce(new Error("Backend unavailable"));
    renderAside(parked);
    await user.click(screen.getByRole("button", { name: "Mark refunded" }));
    const dialog = screen.getByRole("dialog", { name: "Mark refunded" });
    await user.click(within(dialog).getByRole("button", { name: "Mark refunded" }));
    expect(screen.getByRole("dialog", { name: "Mark refunded" })).toBeInTheDocument();
    expect(useToastStore.getState().toasts.map((t) => t.kind)).toEqual(["error"]);
  });

  it("closes only the Mark refunded confirm on Escape, not the queue drawer", async () => {
    const user = userEvent.setup();
    fixtures.search = `row=${parked.id}`;
    render(<QueryClientProvider client={client}><DisputesQueue /></QueryClientProvider>);
    const drawer = await screen.findByRole("dialog");
    await user.click(within(drawer).getByRole("button", { name: "Mark refunded" }));
    expect(screen.getByRole("dialog", { name: "Mark refunded" })).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Mark refunded" })).not.toBeInTheDocument();
    expect(screen.getByText("pay_abc123")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(screen.queryByText("pay_abc123")).not.toBeInTheDocument();
  });

  it("releases the drawer Escape when the panel disappears with the confirm open", async () => {
    const user = userEvent.setup();
    fixtures.search = `row=${parked.id}`;
    render(<QueryClientProvider client={client}><DisputesQueue /></QueryClientProvider>);
    const drawer = await screen.findByRole("dialog");
    await user.click(within(drawer).getByRole("button", { name: "Mark refunded" }));
    act(() => { client.setQueryData(adminDisputeKeys.list(admin.id), [{ ...parked, status: "resolved", refundStatus: "succeeded" }]); });
    await waitFor(() => expect(screen.queryByText("pay_abc123")).not.toBeInTheDocument());
    await user.keyboard("{Escape}");
    expect(screen.queryByText("pay_abc123")).not.toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("needs the confirm step before Mark refunded calls the API", async () => {
    const user = userEvent.setup();
    const post = vi.spyOn(api, "post").mockResolvedValueOnce({ ...parked, status: "resolved", refundStatus: "refunded" });
    renderAside(parked);
    await user.click(screen.getByRole("button", { name: "Mark refunded" }));
    expect(post).not.toHaveBeenCalled();
    const dialog = screen.getByRole("dialog", { name: "Mark refunded" });
    expect(dialog).toHaveTextContent("Only do this after PayMongo shows this payment as Refunded.");
    await user.click(within(dialog).getByRole("button", { name: "Mark refunded" }));
    expect(post).toHaveBeenCalledWith(`/admin/disputes/${parked.id}/mark-refunded`);
    expect(lastToast()?.kind).toBe("success");
  });

  it("disables every panel button while an action is pending", async () => {
    const user = userEvent.setup();
    let release!: (v: unknown) => void;
    vi.spyOn(api, "post").mockReturnValueOnce(new Promise((yes) => { release = yes; }));
    renderAside(parked);
    await user.type(screen.getByLabelText("Refund link"), "https://pay.example/r/1");
    await user.click(screen.getByRole("button", { name: "Send to runner" }));
    expect(screen.getByRole("button", { name: "Copy" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Mark refunded" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Send/ })).toBeDisabled();
    await act(async () => { release({ ...parked, refundLinkUrl: "https://pay.example/r/1" }); });
  });

  it("appears in the queue drawer in place of the old sentence", async () => {
    fixtures.search = `row=${parked.id}`;
    render(<QueryClientProvider client={client}><DisputesQueue /></QueryClientProvider>);
    const drawer = await screen.findByRole("dialog");
    expect(within(drawer).getByText("pay_abc123")).toBeInTheDocument();
    expect(within(drawer).queryByText("Review the payment in PayMongo before another refund attempt.")).not.toBeInTheDocument();
  });
});

describe("refund confirm modal copy", () => {
  const base = { open: true, onCancel: () => {}, onConfirm: () => {}, amount: 100, runnerHandle: "runner", isFull: true };
  it("explains the PayMongo steps for QR Ph", () => {
    render(<AdminRefundConfirmModal {...base} paymentMethod="qrph" />);
    expect(screen.getByRole("dialog")).toHaveTextContent(
      "After you approve, make the refund link in the PayMongo dashboard and send it here. PayMongo's dashboard can only refund the whole payment.",
    );
  });
  it("keeps the usual text for other payment methods", () => {
    render(<AdminRefundConfirmModal {...base} paymentMethod="gcash" />);
    expect(screen.getByRole("dialog")).toHaveTextContent("The refund goes to the original payment method when the provider confirms it.");
  });
});
