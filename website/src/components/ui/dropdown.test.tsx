import type React from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { Dropdown, DropdownItem, DropdownSelect } from "./dropdown";

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("reports open state and closes portaled options when disabled", () => {
  const change = vi.fn(); const pick = vi.fn();
  const view = render(<Dropdown trigger="Region" onOpenChange={change}><DropdownItem onClick={pick}>Cebu</DropdownItem></Dropdown>);
  fireEvent.click(screen.getByRole("button", { name: "Region" }));
  expect(screen.getByRole("menu").parentElement).toBe(document.body);
  expect(change).toHaveBeenLastCalledWith(true);
  view.rerender(<Dropdown trigger="Region" disabled onOpenChange={change}><DropdownItem onClick={pick}>Cebu</DropdownItem></Dropdown>);
  expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Region" })).toBeDisabled();
  expect(change).toHaveBeenLastCalledWith(false);
  expect(pick).not.toHaveBeenCalled();
});

it("returns focus to its trigger on Escape and exposes field errors on that button", () => {
  render(<Dropdown trigger="Region" id="region" ariaLabel="Region: Cebu" ariaDescribedBy="region-error" onOpenChange={() => {}}><DropdownItem>Cebu</DropdownItem></Dropdown>);
  const trigger = screen.getByRole("button", { name: "Region: Cebu" });
  expect(trigger).toHaveAttribute("aria-describedby", "region-error");
  fireEvent.click(trigger); screen.getByRole("menuitem").focus();
  fireEvent.keyDown(document, { key: "Escape" });
  expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
});

it("keeps a wide panel inside a narrow viewport and opens above a low trigger", () => {
  vi.stubGlobal("innerWidth", 375); vi.stubGlobal("innerHeight", 812);
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
    return (this.getAttribute("role") === "menu"
      ? { width: 343, height: 160, top: 0, bottom: 160, left: 0, right: 343 }
      : { width: 200, height: 40, top: 700, bottom: 740, left: 280, right: 480 }) as DOMRect;
  });
  render(<Dropdown trigger="Region"><DropdownItem>Cebu</DropdownItem></Dropdown>);
  fireEvent.click(screen.getByRole("button", { name: "Region" }));
  const menu = screen.getByRole("menu");
  expect(parseFloat(menu.style.left) + 343).toBeLessThanOrEqual(367);
  expect(parseFloat(menu.style.top) + 160).toBeLessThan(700);
  expect(parseFloat(menu.style.maxHeight)).toBeLessThanOrEqual(320);
});

const SOURCES = [
  { value: "ALL", label: "Runner and support" },
  { value: "RUNNER", label: "Runner requests" },
  { value: "SUPPORT", label: "Support requests" },
  { value: "OLD", label: "Retired source", disabled: true },
] as const;
type Source = (typeof SOURCES)[number]["value"];

function SourceSelect(props: Partial<React.ComponentProps<typeof DropdownSelect<Source>>>) {
  return (
    <>
      <label htmlFor="source">Source</label>
      <DropdownSelect<Source> id="source" value="RUNNER" onChange={() => {}} options={SOURCES} placeholder="Pick a source" {...props} />
    </>
  );
}

it("select mode: a labelled 44px field that announces a list box and its value", () => {
  const view = render(<SourceSelect invalid ariaDescribedBy="source-error" />);
  const trigger = screen.getByLabelText("Source");
  expect(trigger.tagName).toBe("BUTTON");
  expect(trigger).toHaveAttribute("aria-haspopup", "listbox");
  expect(trigger).toHaveAttribute("aria-expanded", "false");
  expect(trigger).toHaveAttribute("aria-invalid", "true");
  expect(trigger).toHaveAttribute("aria-describedby", "source-error");
  expect(trigger.className).toContain("min-h-[44px]");
  expect(trigger).toHaveTextContent("Runner requests");
  view.rerender(<SourceSelect value="" />);
  expect(screen.getByLabelText("Source")).toHaveTextContent("Pick a source");
  expect(screen.getByLabelText("Source")).not.toHaveAttribute("aria-invalid");
});

it("select mode: options carry one aria-selected, and choosing one reports it, closes and refocuses", () => {
  const change = vi.fn();
  render(<SourceSelect onChange={change} />);
  const trigger = screen.getByLabelText("Source");
  fireEvent.click(trigger);
  const list = screen.getByRole("listbox", { name: "Source" });
  expect(list.parentElement).toBe(document.body);
  expect(trigger).toHaveAttribute("aria-expanded", "true");
  const options = screen.getAllByRole("option");
  expect(options).toHaveLength(4);
  expect(options.filter((o) => o.getAttribute("aria-selected") === "true")).toEqual([screen.getByRole("option", { name: "Runner requests" })]);
  fireEvent.click(screen.getByRole("option", { name: "Support requests" }));
  expect(change).toHaveBeenCalledWith("SUPPORT");
  expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
});

it("select mode: arrows, Home, End, a typed letter, Enter and Escape", async () => {
  const user = userEvent.setup();
  const change = vi.fn();
  render(<SourceSelect onChange={change} />);
  const trigger = screen.getByLabelText("Source");
  trigger.focus();
  await user.keyboard("{ArrowDown}");
  expect(screen.getByRole("option", { name: "Runner requests" })).toHaveFocus();
  await user.keyboard("{ArrowDown}");
  expect(screen.getByRole("option", { name: "Support requests" })).toHaveFocus();
  await user.keyboard("{ArrowUp}");
  expect(screen.getByRole("option", { name: "Runner requests" })).toHaveFocus();
  await user.keyboard("{End}");
  expect(screen.getByRole("option", { name: "Support requests" })).toHaveFocus();
  await user.keyboard("{Home}");
  expect(screen.getByRole("option", { name: "Runner and support" })).toHaveFocus();
  await user.keyboard("r");
  expect(screen.getByRole("option", { name: "Runner requests" })).toHaveFocus();
  await user.keyboard("s");
  expect(screen.getByRole("option", { name: "Support requests" })).toHaveFocus();
  await user.keyboard("{Escape}");
  expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
  expect(change).not.toHaveBeenCalled();
  await user.keyboard("{Enter}");
  expect(screen.getByRole("option", { name: "Runner requests" })).toHaveFocus();
  await user.keyboard("{Home}{Enter}");
  expect(change).toHaveBeenCalledWith("ALL");
  expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
});

it("select mode: ArrowDown after a mouse open moves into the list", async () => {
  const user = userEvent.setup();
  render(<SourceSelect />);
  const trigger = screen.getByLabelText("Source");
  await user.click(trigger);
  expect(trigger).toHaveFocus();
  await user.keyboard("{ArrowDown}");
  expect(screen.getByRole("option", { name: "Runner requests" })).toHaveFocus();
});

it("select mode: disabled never opens", () => {
  render(<SourceSelect disabled />);
  const trigger = screen.getByLabelText("Source");
  expect(trigger).toBeDisabled();
  fireEvent.click(trigger);
  expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
});
