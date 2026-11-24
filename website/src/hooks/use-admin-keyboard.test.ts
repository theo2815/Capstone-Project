import { act, cleanup, fireEvent, renderHook } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { useQueueKeyboardNav } from "./use-admin-keyboard";

afterEach(() => { cleanup(); document.body.innerHTML = ""; });

it("Enter activates the J/K row from the page, but not when focus is on a button", () => {
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  const onActivate = vi.fn();
  renderHook(() => useQueueKeyboardNav({ rowIds: ["a", "b"], disabled: false, onActivate }));
  act(() => { fireEvent.keyDown(document.body, { key: "j" }); });

  const button = document.body.appendChild(document.createElement("button"));
  button.focus();
  fireEvent.keyDown(button, { key: "Enter" });
  expect(onActivate).not.toHaveBeenCalled();

  fireEvent.keyDown(document.body, { key: "Enter" });
  expect(onActivate).toHaveBeenCalledExactlyOnceWith("a");
});

it("J/K takes real focus off a row's button, so Enter opens the J/K row, not the old one", () => {
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  document.body.innerHTML = '<article data-row-id="a"><button>A</button></article><article data-row-id="b"><button>B</button></article>';
  const onActivate = vi.fn();
  renderHook(() => useQueueKeyboardNav({ rowIds: ["a", "b"], disabled: false, onActivate }));
  document.querySelector<HTMLButtonElement>('[data-row-id="a"] button')!.focus();
  act(() => { fireEvent.keyDown(document.activeElement!, { key: "j" }); });
  act(() => { fireEvent.keyDown(document.activeElement!, { key: "j" }); });
  fireEvent.keyDown(document.activeElement!, { key: "Enter" });
  expect(onActivate).toHaveBeenCalledExactlyOnceWith("b");
});

it("two lists on one page: J/K move the list that holds focus; from the page body, the primary one", () => {
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  document.body.innerHTML = '<button id="out">Rail</button>'
    + '<section data-kbd-scope=""><ul><li data-row-id="s1"><button id="in">S1</button></li><li data-row-id="s2"></li></ul></section>';
  const scopeRef = { current: document.querySelector<HTMLElement>("[data-kbd-scope]") };
  const onSecond = vi.fn();
  const primary = renderHook(() => useQueueKeyboardNav({ rowIds: ["p1", "p2"], disabled: false, onActivate: vi.fn() }));
  const second = renderHook(() => useQueueKeyboardNav({ rowIds: ["s1", "s2"], disabled: false, onActivate: onSecond, scopeRef }));

  act(() => { fireEvent.keyDown(document.body, { key: "j" }); });
  expect(primary.result.current.focusedId).toBe("p1");
  expect(second.result.current.focusedId).toBeNull();

  // Focus inside the second list: J moves it alone. J blurs the row, so the next keys come from the body and stay with it.
  document.getElementById("in")!.focus();
  act(() => { fireEvent.keyDown(document.activeElement!, { key: "j" }); });
  act(() => { fireEvent.keyDown(document.activeElement!, { key: "j" }); });
  fireEvent.keyDown(document.activeElement!, { key: "Enter" });
  expect(second.result.current.focusedId).toBe("s2");
  expect(onSecond).toHaveBeenCalledExactlyOnceWith("s2");
  expect(primary.result.current.focusedId).toBe("p1");

  // Focus outside every scoped list hands the keys back to the primary one.
  document.getElementById("out")!.focus();
  act(() => { fireEvent.keyDown(document.activeElement!, { key: "j" }); });
  expect(primary.result.current.focusedId).toBe("p2");
  expect(second.result.current.focusedId).toBe("s2");
});

it("a scoped list entered through one of its controls: J takes focus off the control, so Enter opens the row", () => {
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  document.body.innerHTML = '<section data-kbd-scope=""><button id="tab">Needs review</button><ul><li data-row-id="s1"></li></ul></section>';
  const scopeRef = { current: document.querySelector<HTMLElement>("[data-kbd-scope]") };
  const onActivate = vi.fn();
  renderHook(() => useQueueKeyboardNav({ rowIds: ["s1"], disabled: false, onActivate, scopeRef }));
  document.getElementById("tab")!.focus();
  act(() => { fireEvent.keyDown(document.activeElement!, { key: "j" }); });
  fireEvent.keyDown(document.activeElement!, { key: "Enter" });
  expect(onActivate).toHaveBeenCalledExactlyOnceWith("s1");
});

it("keys pressed inside a dialog never move a list behind it, and J in an empty scoped list keeps focus", () => {
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  document.body.innerHTML = '<div role="dialog"><button id="d">Close</button></div><section data-kbd-scope=""><button id="tab">All</button></section>';
  const scopeRef = { current: document.querySelector<HTMLElement>("[data-kbd-scope]") };
  const primary = renderHook(() => useQueueKeyboardNav({ rowIds: ["p1"], disabled: false, onActivate: vi.fn() }));
  renderHook(() => useQueueKeyboardNav({ rowIds: [], disabled: false, onActivate: vi.fn(), scopeRef }));
  document.getElementById("d")!.focus();
  act(() => { fireEvent.keyDown(document.activeElement!, { key: "j" }); });
  expect(primary.result.current.focusedId).toBeNull();
  document.getElementById("tab")!.focus();
  act(() => { fireEvent.keyDown(document.activeElement!, { key: "j" }); });
  expect(document.activeElement?.id).toBe("tab");
});
