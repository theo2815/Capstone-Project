import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { AdminPayoutBulkBar } from "./admin-payout-bulk-bar";

afterEach(cleanup);

// FT-35 (C-26): Clear first, Hold next, the green Approve last; both dialogs carry "…".
it("bulk bar buttons read Clear, Hold, then Approve", () => {
  render(<AdminPayoutBulkBar count={2} submitting={false} totalAmount={50000} approveDisabled={false} approveDisabledReason={null} onApprove={vi.fn()} onHold={vi.fn()} onClear={vi.fn()} />);
  expect(screen.getAllByRole("button").map((b) => b.textContent)).toEqual(["Clear", "Hold 2…", "Approve 2…"]);
});
