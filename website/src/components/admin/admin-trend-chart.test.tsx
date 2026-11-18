import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { AdminTrendChart } from "./admin-trend-chart";

afterEach(cleanup);

const data = [
  { date: "2026-11-17", amount: 0 },
  { date: "2026-11-18", amount: 3 },
  { date: "2026-11-19", amount: 6 },
];

function renderChart() {
  return render(<AdminTrendChart data={data} unitLabel="decisions" ariaLabel="Daily decisions for the last 30 days" />);
}

it("gives screen readers a table with one row per point instead of an image of buttons", () => {
  renderChart();
  expect(screen.queryByRole("img")).toBeNull();
  expect(screen.queryAllByRole("button")).toHaveLength(0);
  const table = screen.getByRole("table", { name: "Daily decisions for the last 30 days" });
  const rows = within(table).getAllByRole("row");
  expect(rows).toHaveLength(data.length + 1); // header row + one per point
  expect(within(rows[1]).getAllByRole("cell")[1].textContent).toBe("0 decisions");
  expect(within(rows[3]).getAllByRole("cell")[1].textContent).toBe("6 decisions");
});

it("draws a zero day at zero height", () => {
  const { container } = renderChart();
  const heights = [...container.querySelectorAll<HTMLElement>("[style*='height']")].map((el) => el.style.height);
  expect(heights).toEqual(["0%", "50%", "100%"]);
});
