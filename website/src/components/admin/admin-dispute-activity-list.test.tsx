import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it } from "vitest";
import { formatSalesDate, sentenceCaseDate } from "@/lib/format";
import { AdminDisputeActivityList } from "./admin-dispute-activity-list";

afterEach(cleanup);

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

it("shows 20 entries, then a Show all button for the rest", async () => {
  const activity = Array.from({ length: 25 }, (_, i) => ({
    id: `a${i}`, decidedAt: new Date(Date.UTC(2026, 9, 1, i)).toISOString(), decision: "escalated" as const,
    resolution: null, refundAmount: null, reason: `note ${i}`,
  }));
  render(<AdminDisputeActivityList activity={activity} />);
  expect(screen.getAllByRole("listitem")).toHaveLength(20);
  await userEvent.click(screen.getByRole("button", { name: "Show all 25" }));
  expect(screen.getAllByRole("listitem")).toHaveLength(25);
  expect(screen.queryByRole("button", { name: /Show all/ })).not.toBeInTheDocument();
});

it("dates come from the shared formatter, in Manila time", () => {
  render(<AdminDisputeActivityList activity={[{ id: "a1", decidedAt: "2026-10-28T02:00:00Z", decision: "escalated", resolution: null, refundAmount: null, reason: null }]} />);
  expect(screen.getByRole("listitem")).toHaveTextContent(sentenceCaseDate(formatSalesDate("2026-10-28T02:00:00Z")));
});

// AX-63: the activity time reads as typed, not in CSS capitals.
it("reads the activity time as typed, not in capitals", () => {
  render(<AdminDisputeActivityList activity={[{ id: "a1", decidedAt: "2026-10-28T02:00:00Z", decision: "escalated", resolution: null, refundAmount: null, reason: null }]} />);
  expect(screen.getByText("Oct 28, 2026, 10:00 am").closest(".uppercase")).toBeNull();
});

// FT-18 (C-43): the decision reads as a sentence on its own line, above the date.
it("sets the decision label in sentence case above the date", () => {
  render(<AdminDisputeActivityList activity={[{ id: "a1", decidedAt: "2026-10-28T02:00:00Z", decision: "refund_sent_to_manual_review", resolution: null, refundAmount: null, reason: null }]} />);
  const label = screen.getByText("Refund approved, waiting for PayMongo");
  for (const cls of ["uppercase", "font-mono", "shrink-0"]) expect(label).not.toHaveClass(cls);
  expect(label.compareDocumentPosition(screen.getByText("Oct 28, 2026, 10:00 am")) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
});
