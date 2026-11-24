import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { AdminStatusTabs } from "./admin-status-tabs";

afterEach(cleanup);
const noop = () => {};

it("a caller that passes none of the Inbox props renders as before: visible names, wrapping row", () => {
  render(<AdminStatusTabs label="Event status" tabs={[{ key: "live", label: "Live now", count: 3 }]} value="live" onChange={noop} />);
  const tab = screen.getByRole("button", { name: /^Live now ?3$/ });
  expect(tab).not.toHaveAttribute("aria-label");
  expect(screen.getByRole("group", { name: "Event status" }).className).toMatch(/\bflex-wrap\b/);
});

it("a dot without a count label still names the count and the attention", () => {
  render(<AdminStatusTabs label="Queues" tabs={[{ key: "d", label: "Disputes", count: 2, dot: true }]} value="d" onChange={noop} />);
  expect(screen.getByRole("button", { name: "Disputes, 2, needs attention" })).toBeTruthy();
});

it("the scroll row sets its own sideways scroll and never scrolls the page", () => {
  const calls: unknown[] = [];
  const original = Element.prototype.scrollIntoView;
  Element.prototype.scrollIntoView = (...a: unknown[]) => { calls.push(a); };
  try {
    render(<AdminStatusTabs label="Queues" tabs={[{ key: "a", label: "A", count: 1 }, { key: "b", label: "B", count: 2 }]} value="b" onChange={noop} scroll />);
    expect(calls).toHaveLength(0);
    expect(screen.getByRole("group", { name: "Queues" }).className).toMatch(/\bflex-nowrap\b/);
  } finally {
    Element.prototype.scrollIntoView = original;
  }
});
