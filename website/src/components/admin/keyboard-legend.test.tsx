import { act, cleanup, render, screen } from "@testing-library/react";
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
