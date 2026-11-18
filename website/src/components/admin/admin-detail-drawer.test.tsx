import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, expect, it } from "vitest";
import { AdminDetailDrawer } from "./admin-detail-drawer";

afterEach(cleanup);

it("names the drawer from a JSX title", () => {
  render(
    <AdminDetailDrawer open onClose={() => {}} title={<span>Payout for <b>Ana Cruz</b></span>}>
      <p>Body</p>
    </AdminDetailDrawer>,
  );
  expect(screen.getByRole("dialog", { name: /Payout for Ana Cruz/ })).toBeInTheDocument();
});

it("takes focus from the row that opened it and gives it back on Escape", () => {
  function Queue() {
    const [open, setOpen] = useState(false);
    return (
      <>
        <button type="button" onClick={() => setOpen(true)}>Row</button>
        <AdminDetailDrawer open={open} onClose={() => setOpen(false)} title="Payout">
          <button type="button">Approve</button>
        </AdminDetailDrawer>
      </>
    );
  }
  render(<Queue />);
  const row = screen.getByRole("button", { name: "Row" });
  row.focus();
  fireEvent.click(row);
  expect(screen.getByRole("dialog", { name: "Payout" }).contains(document.activeElement)).toBe(true);
  fireEvent.keyDown(document, { key: "Escape" });
  expect(row).toHaveFocus();
});
