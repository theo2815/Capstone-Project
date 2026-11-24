import { act, cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useAdminLegendStore } from "@/store/admin-legend-store";
import { KeyboardLegend } from "./keyboard-legend";

beforeEach(() => { act(() => useAdminLegendStore.setState({ open: true })); });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); act(() => useAdminLegendStore.setState({ open: false })); });

it("lists no Suspend shortcut, since no drawer wires S", () => {
  render(<KeyboardLegend />);
  expect(screen.queryByText("Suspend")).not.toBeInTheDocument();
});

it("shows Ctrl K off Apple platforms", () => {
  vi.stubGlobal("navigator", { ...navigator, platform: "Win32", userAgent: "Windows NT 10.0" });
  render(<KeyboardLegend />);
  expect(screen.getByText("Ctrl")).toBeTruthy();
  expect(screen.queryByText("⌘")).not.toBeInTheDocument();
});

it("keeps ⌘ K on a Mac", () => {
  vi.stubGlobal("navigator", { ...navigator, platform: "MacIntel", userAgent: "Macintosh" });
  render(<KeyboardLegend />);
  expect(screen.getByText("⌘")).toBeTruthy();
});

it("FT-36 (W36.1): the verb rows say what each key opens", () => {
  render(<KeyboardLegend />);
  for (const [label, key] of [
    ["Approve a payout (opens a confirmation)", "E"],
    ["Send a profile back (opens the dialog)", "R"],
    ["Hold a payout, or hide a flagged photo", "H"],
  ]) {
    expect(within(screen.getByText(label).closest("li")!).getByText(key, { selector: "kbd" })).toBeInTheDocument();
  }
  for (const old of ["Approve", "Reject", "Hide"]) expect(screen.queryByText(old)).not.toBeInTheDocument();
});
