import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { payoutCycle } from "@/test/fixtures/admin";
import { AdminPayoutRow, payoutStatusTone } from "./admin-payout-row";

afterEach(cleanup);

function renderRow(cycle = payoutCycle) {
  const props = {
    cycle, submitting: false, selected: false, selectionActive: false, focused: false,
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
  const approve = screen.getByRole("button", { name: "Approve…" });
  await userEvent.click(approve);
  fireEvent.keyDown(approve, { key: "Enter" });
  expect(props.onToggleSelect).toHaveBeenCalledTimes(1);
  expect(props.onApprove).toHaveBeenCalledTimes(1);
  expect(props.onActivate).not.toHaveBeenCalled();
});

// FT-40 (C-23): the 16 px checkbox sits in a 44 px label, like the Verifications rows.
it("the row checkbox has a 44 px hit area", () => {
  renderRow();
  const label = screen.getByRole("checkbox", { name: `Select payout ${payoutCycle.id}` }).closest("label");
  expect(label?.className).toContain("size-11");
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

// FT-23 (C-15, W23.1): a half-approved payout says so on its row, not only in the drawer.
it("a pending row with a first approval says it waits for a second admin", () => {
  renderRow({ ...payoutCycle, approvedBy: "admin-b" });
  expect(screen.getByText("Waiting for a second admin (1 of 2)")).toBeInTheDocument();
  cleanup();
  renderRow();
  expect(screen.queryByText(/Waiting for a second admin/)).not.toBeInTheDocument();
});

// FT-35 (C-26, C-62): the stop first, the forward step last; a button that opens a dialog ends in "…".
const actionNames = () => screen.getAllByRole("button").map((b) => b.textContent).filter((n) => /^(Hold|Mark|Approve)/.test(n ?? ""));
it("row buttons read stop first and the forward step last, each with its ellipsis", () => {
  renderRow();
  expect(actionNames()).toEqual(["Hold…", "Approve…"]);
  cleanup();
  renderRow({ ...payoutCycle, status: "approved" });
  expect(actionNames()).toEqual(["Hold…", "Mark processing…", "Mark paid…"]);
});
