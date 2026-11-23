import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { payoutCycle } from "@/test/fixtures/admin";
import { AdminPayoutRow, payoutStatusTone } from "./admin-payout-row";

afterEach(cleanup);

function renderRow() {
  const props = {
    cycle: payoutCycle, submitting: false, selected: false, selectionActive: false, focused: false,
    onToggleSelect: vi.fn(), onActivate: vi.fn(), onOpenDrawer: vi.fn(), onApprove: vi.fn(),
    onHold: vi.fn(), onMarkPaid: vi.fn(), onMarkProcessing: vi.fn(),
  };
  const { container } = render(<AdminPayoutRow {...props} />);
  return { props, container };
}

it("the row opens through a button named by its title, and is not itself a button", async () => {
  const { props, container } = renderRow();
  await userEvent.click(screen.getByRole("button", { name: payoutCycle.photographerName }));
  expect(props.onActivate).toHaveBeenCalledTimes(1);
  expect(container.querySelector('[role="button"]')).toBeNull();
});

it("the checkbox and an inline action do not open the row, by click or Enter", async () => {
  const { props } = renderRow();
  await userEvent.click(screen.getByRole("checkbox", { name: `Select payout ${payoutCycle.id}` }));
  const approve = screen.getByRole("button", { name: "Approve" });
  await userEvent.click(approve);
  fireEvent.keyDown(approve, { key: "Enter" });
  expect(props.onToggleSelect).toHaveBeenCalledTimes(1);
  expect(props.onApprove).toHaveBeenCalledTimes(1);
  expect(props.onActivate).not.toHaveBeenCalled();
});

// FT-16 (C-25): green would read as finished, so no status is fresh; failed and returned are errors.
it("status tones: Approved is ink, Failed and Returned are error, Paid stays slate", () => {
  expect(payoutStatusTone("approved")).toBe("ink");
  expect(payoutStatusTone("failed")).toBe("error");
  expect(payoutStatusTone("returned")).toBe("error");
  expect(payoutStatusTone("paid")).toBe("slate");
  const all = ["pending_review", "approved", "processing", "held", "paid", "failed", "returned", "rejected", "cancelled"] as const;
  expect(all.map(payoutStatusTone)).not.toContain("fresh");
});
