import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { AdminResolveDisputeModal } from "./admin-resolve-dispute-modal";

afterEach(cleanup);

function renderModal(props: Partial<React.ComponentProps<typeof AdminResolveDisputeModal>> = {}) {
  render(
    <AdminResolveDisputeModal open onClose={() => {}} onContinue={() => {}} disputeId="d1"
      orderTotal={300} itemChargeAmount={100} reasonLabel="Too blurred to recognise" {...props} />,
  );
  return screen.getByRole("dialog", { name: "Resolve d1" });
}

it("a QR Ph dispute offers only a full refund", () => {
  const dialog = renderModal({ paymentMethod: "qrph" });
  expect(within(dialog).getAllByRole("radio")).toHaveLength(1);
  expect(within(dialog).queryByRole("radio", { name: /Partial/ })).not.toBeInTheDocument();
  expect(within(dialog).queryByRole("radio", { name: /Deny/ })).not.toBeInTheDocument();
});

it("a dispute that can't be refunded points to Deny instead of Approve", async () => {
  const user = userEvent.setup();
  const onDenyWithReason = vi.fn();
  const dialog = renderModal({
    refundable: false, refundBlockedReason: "bought_together", paymentPhotoCount: 3,
    refundBlockedMessage: "PayMongo can only refund a whole payment.", onDenyWithReason,
  });
  expect(dialog).toHaveTextContent("PayMongo can only refund a whole payment.");
  expect(dialog).toHaveTextContent("The payment covers 3 photos.");
  expect(within(dialog).getByRole("button", { name: "Continue" })).toBeDisabled();
  await user.click(within(dialog).getByRole("button", { name: "Deny with this reason" }));
  expect(onDenyWithReason).toHaveBeenCalledWith("bought_together");
});

it("a no_payment dispute shows the reason and no Deny shortcut", () => {
  const dialog = renderModal({
    refundable: false, refundBlockedReason: "no_payment", refundBlockedMessage: "No settled payment yet.",
    onDenyWithReason: vi.fn(),
  });
  expect(dialog).toHaveTextContent("No settled payment yet.");
  expect(within(dialog).queryByRole("button", { name: "Deny with this reason" })).not.toBeInTheDocument();
  expect(within(dialog).getByRole("button", { name: "Continue" })).toBeDisabled();
});

it("the note starts filled in", () => {
  const dialog = renderModal();
  expect(within(dialog).getByRole("textbox")).toHaveValue("Approved: Too blurred to recognise.");
  expect(within(dialog).getByRole("button", { name: "Continue" })).toBeEnabled();
  expect(dialog).toHaveTextContent("This photo's charge");
});
