import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
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

it("gives screen readers a table with one row per point, and one named button per bar", () => {
  renderChart();
  expect(screen.queryByRole("img")).toBeNull();
  expect(screen.getAllByRole("button")).toHaveLength(data.length);
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

it("clicking or focusing a bar reads its day and value without a mouse hover", () => {
  renderChart();
  const bar = screen.getByRole("button", { name: "Nov 18 · 2026: 3 decisions" });
  expect(screen.getAllByText("3 decisions")).toHaveLength(1); // the table cell only
  fireEvent.click(bar);
  expect(screen.getAllByText("3 decisions")).toHaveLength(2); // and the readout
  fireEvent.focus(screen.getByRole("button", { name: "Nov 19 · 2026: 6 decisions" }));
  expect(screen.getAllByText("6 decisions")).toHaveLength(2);
});

it("names the busiest day under the bars", () => {
  renderChart();
  expect(screen.getByText("Busiest: 6 decisions on Nov 19 · 2026.")).toBeInTheDocument();
});

it("keeps a caller's own day label and adds the unit after a formatted value", () => {
  render(<AdminTrendChart data={[{ date: "2026-11-16", amount: 1200 }]} unitLabel="GMV" ariaLabel="x"
    formatLabel={() => "WK OF NOV 16"} formatValue={(n) => `P${n}`} />);
  expect(screen.getByRole("button", { name: "WK OF NOV 16: P1200 GMV" })).toBeInTheDocument();
  expect(screen.getByText("Busiest: P1200 GMV on WK OF NOV 16.")).toBeInTheDocument();
});

it("prints no busiest day when every day is zero", () => {
  render(<AdminTrendChart data={[{ date: "2026-11-17", amount: 0 }]} unitLabel="decisions" ariaLabel="x" />);
  expect(screen.queryByText(/Busiest/)).toBeNull();
});
