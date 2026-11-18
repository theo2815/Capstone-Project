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
