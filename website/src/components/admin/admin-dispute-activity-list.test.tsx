import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { AdminDisputeActivityList } from "./admin-dispute-activity-list";

it("labels a refund hand-off row instead of leaving it blank", () => {
  render(
    <AdminDisputeActivityList
      activity={[
        {
          id: "a1",
          decidedAt: "2026-10-28T02:00:00Z",
          decision: "refund_sent_to_manual_review",
          resolution: null,
          refundAmount: 127.5,
          reason: null,
        },
      ]}
    />,
  );

  expect(screen.getByText("Sent to manual review")).toBeInTheDocument();
});
