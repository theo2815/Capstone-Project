import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { AdminOverviewTrend } from "./admin-overview-trend";

beforeEach(() => useAuthStore.getState().setUser(admin));
afterEach(() => { cleanup(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

function renderTrend() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><AdminOverviewTrend /></QueryClientProvider>);
}

const point = (date: string, decisions: number, extra: object = {}) => ({ date, decisions, disputes: 0, payouts: 0, ...extra });

it("shows a busy skeleton while the trend loads", () => {
  vi.spyOn(api, "get").mockReturnValue(new Promise(() => {}));
  const { container } = renderTrend();
  expect(container.querySelector("[aria-busy='true']")).not.toBeNull();
});

it("shows an alert with Retry when the trend fails", async () => {
  const get = vi.spyOn(api, "get").mockRejectedValue(new Error("500"));
  renderTrend();
  const alert = await screen.findByRole("alert");
  expect(alert.textContent).toContain("Couldn't load the trend. Please try again.");
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(get).toHaveBeenCalledTimes(2);
});

it("says there was no activity when every day is zero", async () => {
  vi.spyOn(api, "get").mockResolvedValue([point("2026-11-18", 0), point("2026-11-19", 0)]);
  renderTrend();
  expect(await screen.findByText("No activity in the last 30 days.")).toBeInTheDocument();
});

it("does not claim no activity when only uploads happened", async () => {
  vi.spyOn(api, "get").mockResolvedValue([point("2026-11-18", 0, { uploads: 7 }), point("2026-11-19", 0, { uploads: 0 })]);
  renderTrend();
  expect(await screen.findByText(/0 decisions · last 30 days/)).toBeInTheDocument();
  expect(screen.queryByText("No activity in the last 30 days.")).toBeNull();
});

it("has no Uploads toggle when the backend sends no uploads field", async () => {
  vi.spyOn(api, "get").mockResolvedValue([point("2026-11-18", 2), point("2026-11-19", 3)]);
  renderTrend();
  expect(await screen.findByText(/5 decisions/)).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Uploads" })).toBeNull();
});

it("charts the real uploads when the backend sends them", async () => {
  vi.spyOn(api, "get").mockResolvedValue([point("2026-11-18", 2, { uploads: 40 }), point("2026-11-19", 3, { uploads: 2 })]);
  renderTrend();
  fireEvent.click(await screen.findByRole("button", { name: "Uploads" }));
  expect(screen.getByText(/42 uploads/)).toBeInTheDocument();
});
