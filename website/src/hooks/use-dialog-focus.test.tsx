import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState, type ReactNode } from "react";
import { afterEach, expect, it } from "vitest";
import { useDialogFocus } from "./use-dialog-focus";

afterEach(cleanup);

function Dialog({ name, open, children }: { name: string; open: boolean; children: ReactNode }) {
  const [el, setEl] = useState<HTMLDivElement | null>(null);
  useDialogFocus(el, open, { trap: true });
  if (!open) return null;
  return (
    <div ref={setEl} role="dialog" aria-label={name}>
      {children}
    </div>
  );
}

function Page() {
  const [first, setFirst] = useState(false);
  const [second, setSecond] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setFirst(true)}>Open first</button>
      <Dialog name="First" open={first}>
        <button type="button">A</button>
        <button type="button" onClick={() => setSecond(true)}>Open second</button>
        <button type="button" onClick={() => setFirst(false)}>Close first</button>
      </Dialog>
      <Dialog name="Second" open={second}>
        <button type="button">X</button>
        <button type="button" onClick={() => setSecond(false)}>Close second</button>
      </Dialog>
    </>
  );
}

function openFirst() {
  const opener = screen.getByRole("button", { name: "Open first" });
  opener.focus();
  fireEvent.click(opener);
  return opener;
}

it("opening moves focus to the first control inside", () => {
  render(<Page />);
  openFirst();
  expect(screen.getByRole("button", { name: "A" })).toHaveFocus();
});

it("Tab on the last control goes to the first, Shift+Tab on the first goes to the last", () => {
  render(<Page />);
  openFirst();
  const first = screen.getByRole("button", { name: "A" });
  const last = screen.getByRole("button", { name: "Close first" });
  last.focus();
  fireEvent.keyDown(last, { key: "Tab" });
  expect(first).toHaveFocus();
  fireEvent.keyDown(first, { key: "Tab", shiftKey: true });
  expect(last).toHaveFocus();
});

it("closing returns focus to the opener", () => {
  render(<Page />);
  const opener = openFirst();
  fireEvent.click(screen.getByRole("button", { name: "Close first" }));
  expect(opener).toHaveFocus();
});

it("with two open, only the top dialog traps and each close steps focus back one layer", () => {
  render(<Page />);
  openFirst();
  const openSecond = screen.getByRole("button", { name: "Open second" });
  openSecond.focus();
  fireEvent.click(openSecond);
  const x = screen.getByRole("button", { name: "X" });
  expect(x).toHaveFocus();
  // Tab on the first dialog's last control is not the first dialog's to handle now.
  const closeFirst = screen.getByRole("button", { name: "Close first" });
  closeFirst.focus();
  fireEvent.keyDown(closeFirst, { key: "Tab" });
  expect(x).toHaveFocus();
  const closeSecond = screen.getByRole("button", { name: "Close second" });
  closeSecond.focus();
  fireEvent.keyDown(closeSecond, { key: "Tab" });
  expect(x).toHaveFocus();
  fireEvent.click(closeSecond);
  expect(openSecond).toHaveFocus();
});

it("a trapping dialog under a newer one leaves Tab alone", () => {
  function Layer({ name, trap }: { name: string; trap: boolean }) {
    const [el, setEl] = useState<HTMLDivElement | null>(null);
    useDialogFocus(el, true, { trap });
    return (
      <div ref={setEl} role="dialog" aria-label={name}>
        <button type="button">{name} first</button>
        <button type="button">{name} last</button>
      </div>
    );
  }
  render(<><Layer name="Lower" trap /><Layer name="Upper" trap={false} /></>);
  const upperLast = screen.getByRole("button", { name: "Upper last" });
  upperLast.focus();
  fireEvent.keyDown(upperLast, { key: "Tab" });
  expect(upperLast).toHaveFocus();
});

it("keeps an autofocused field's focus and still returns focus to the opener", () => {
  function Form() {
    const [open, setOpen] = useState(false);
    const [el, setEl] = useState<HTMLDivElement | null>(null);
    useDialogFocus(el, open, { trap: true });
    return (
      <>
        <button type="button" onClick={() => setOpen(true)}>Open form</button>
        {open && (
          <div ref={setEl} role="dialog">
            <button type="button">Help</button>
            <input aria-label="Name" autoFocus />
            <button type="button" onClick={() => setOpen(false)}>Done</button>
          </div>
        )}
      </>
    );
  }
  render(<Form />);
  const opener = screen.getByRole("button", { name: "Open form" });
  opener.focus();
  act(() => fireEvent.click(opener));
  expect(screen.getByRole("textbox", { name: "Name" })).toHaveFocus();
  fireEvent.click(screen.getByRole("button", { name: "Done" }));
  expect(opener).toHaveFocus();
});
