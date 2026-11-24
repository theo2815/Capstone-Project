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
  return screen.getByRole("dialog", { name: "Approve refund · d1" });
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
  expect(within(dialog).getByRole("button", { name: "Review refund" })).toBeDisabled();
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
  expect(within(dialog).getByRole("button", { name: "Review refund" })).toBeDisabled();
});

it("the note starts filled in", () => {
  const dialog = renderModal();
  expect(within(dialog).getByRole("textbox")).toHaveValue("Approved: Too blurred to recognise.");
  expect(within(dialog).getByRole("button", { name: "Review refund" })).toBeEnabled();
  expect(dialog).toHaveTextContent("This photo's charge");
});

// FT-18 (C-38): one verb, and why QR Ph has one option. (W18.3 and W18.4 are held: a completed refund shows this note to the runner.)
it("is titled Approve refund and continues with Review refund", () => {
  const dialog = renderModal();
  expect(within(dialog).getByRole("button", { name: "Review refund" })).toBeInTheDocument();
  expect(dialog).not.toHaveTextContent("This payment was made by QR Ph");
});

it("a QR Ph dispute says why only a full refund is offered", () => {
  const dialog = renderModal({ paymentMethod: "qrph" });
  expect(within(dialog).getByText("This payment was made by QR Ph, so only a full refund is possible.")).toBeInTheDocument();
});

// FU-7 (U7.1, U7.2): the note's label and the line under it say who reads it (G2).
it("the note says the runner sees it once the refund is finished, and the photographer does not", () => {
  const dialog = renderModal();
  const note = within(dialog).getByLabelText("Note (required, max 500)");
  expect(note).toHaveAccessibleDescription("The runner sees this note on their order once the refund is finished. The photographer does not. It is also kept in the activity log.");
  expect(within(dialog).queryByText(/Reasoning/)).toBeNull();
});
