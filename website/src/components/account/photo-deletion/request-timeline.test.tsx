import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { RUNNER_TIMELINE_STEP_LABEL } from "@/lib/runner-removal-status";
import { RequestTimeline } from "./request-timeline";

afterEach(cleanup);

describe("runner deletion timeline", () => {
  it("every timeline step has a label", () => {
    const steps = ["SUBMITTED", "PHOTOS_HIDDEN", "SENT_FOR_REVIEW", "MORE_INFO_NEEDED", "APPROVED", "KEPT", "WITHDRAWN", "CLOSED"] as const; // backend timelineStep()
    for (const step of steps) expect(RUNNER_TIMELINE_STEP_LABEL[step]).toMatch(/^[A-Z][a-z]/);
    expect(Object.keys(RUNNER_TIMELINE_STEP_LABEL)).toHaveLength(steps.length);
  });

  it("shows each step in order with its time", () => {
    render(<RequestTimeline steps={[{ at: "2026-11-12T02:00:00Z", step: "SUBMITTED" }, { at: "2026-11-12T03:30:00Z", step: "PHOTOS_HIDDEN" }]} />);
    const rows = screen.getAllByRole("listitem");
    expect(rows.map((row) => row.querySelector("span:not([aria-hidden])")?.textContent)).toEqual(["Request sent", "Photos hidden"]);
    expect(rows[0].querySelector("time")).toHaveAttribute("datetime", "2026-11-12T02:00:00Z");
  });

  it("an unknown step falls back to Update", () => {
    render(<RequestTimeline steps={[{ at: "2026-11-12T02:00:00Z", step: "SOMETHING_NEW" as never }]} />);
    expect(screen.getByText("Update")).toBeVisible();
  });
});
