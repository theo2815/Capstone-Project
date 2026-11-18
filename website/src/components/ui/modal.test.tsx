import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, expect, it } from "vitest";
import { Modal } from "./modal";

afterEach(cleanup);

it("moves focus inside when the panel mounts one render after isOpen turns true", () => {
  const opener = document.createElement("button");
  document.body.appendChild(opener);
  opener.focus();
  // Opened on first render: Modal renders nothing until its mount effect, then the panel.
  const { unmount } = render(
    <Modal isOpen onClose={() => {}} title="Suspend">
      <button type="button">Keep</button>
    </Modal>,
  );
  const dialog = screen.getByRole("dialog", { name: "Suspend" });
  expect(dialog.contains(document.activeElement)).toBe(true);
  unmount();
  expect(opener).toHaveFocus();
  opener.remove();
});

it("a modal mounted only while open, with an autofocused field, still gives focus back", () => {
  function Page() {
    const [open, setOpen] = useState(false);
    return (
      <>
        <button type="button" onClick={() => setOpen(true)}>Mark paid</button>
        {open && (
          <Modal isOpen onClose={() => setOpen(false)} title="Mark paid">
            <input aria-label="Reference" autoFocus />
          </Modal>
        )}
      </>
    );
  }
  render(<Page />);
  const opener = screen.getByRole("button", { name: "Mark paid" });
  opener.focus();
  fireEvent.click(opener);
  expect(screen.getByRole("textbox", { name: "Reference" })).toHaveFocus();
  fireEvent.keyDown(document, { key: "Escape" });
  expect(opener).toHaveFocus();
});

it("traps Tab inside and uses ariaLabel when there is no title", () => {
  function Page() {
    const [open, setOpen] = useState(false);
    return (
      <>
        <button type="button" onClick={() => setOpen(true)}>Open</button>
        <Modal isOpen={open} onClose={() => setOpen(false)} ariaLabel="Search">
          <input aria-label="Query" />
        </Modal>
      </>
    );
  }
  render(<Page />);
  const opener = screen.getByRole("button", { name: "Open" });
  opener.focus();
  fireEvent.click(opener);
  const dialog = screen.getByRole("dialog", { name: "Search" });
  const close = screen.getByRole("button", { name: "Close" });
  const query = screen.getByRole("textbox", { name: "Query" });
  expect(close).toHaveFocus();
  query.focus();
  fireEvent.keyDown(query, { key: "Tab" });
  expect(close).toHaveFocus();
  expect(dialog.contains(document.activeElement)).toBe(true);
  fireEvent.keyDown(document, { key: "Escape" });
  expect(opener).toHaveFocus();
});
