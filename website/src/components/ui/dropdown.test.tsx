import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { Dropdown, DropdownItem } from "./dropdown";

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
