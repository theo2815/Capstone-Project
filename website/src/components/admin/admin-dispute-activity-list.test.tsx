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

  expect(screen.getByText("Refund approved, waiting for PayMongo")).toBeInTheDocument();
});

it("labels the manual refund link and auto-cancel rows", () => {
  const row = { resolution: null, refundAmount: null, reason: null } as const;
  render(
    <AdminDisputeActivityList
      activity={[
        { ...row, id: "a2", decidedAt: "2026-10-28T03:00:00Z", decision: "refund_link_sent" },
        { ...row, id: "a3", decidedAt: "2026-10-28T04:00:00Z", decision: "refund_cancelled" },
      ]}
    />,
  );

  expect(screen.getByText("Refund link sent")).toBeInTheDocument();
  expect(screen.getByText("Refund cancelled: window closed")).toBeInTheDocument();
});
