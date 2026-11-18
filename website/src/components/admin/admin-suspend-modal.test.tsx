import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AdminSuspendModal } from "./admin-suspend-modal";

// AX-28 (PD6, UA5): no reason is ever sent that the admin did not pick.
function setup(onSubmit = vi.fn<(reason: string) => Promise<boolean>>(async () => true)) {
  render(<AdminSuspendModal open onClose={() => {}} onSubmit={onSubmit} photographerName="Ana" />);
  return { onSubmit, submit: () => screen.getByRole("button", { name: "Suspend" }) };
}

describe("AdminSuspendModal (AX-28)", () => {
  afterEach(cleanup);

  it("opens with no reason checked and Suspend disabled", () => {
    const { submit } = setup();
    for (const radio of screen.getAllByRole("radio")) expect(radio).not.toBeChecked();
    expect(submit()).toBeDisabled();
  });

  it("Enter on open does not submit", () => {
    const { onSubmit, submit } = setup();
    fireEvent.submit(submit().closest("form")!);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("Other with an empty note shows the note error and sends nothing", async () => {
    const user = userEvent.setup();
    const { onSubmit, submit } = setup();
    await user.click(screen.getByRole("radio", { name: "Other (see note)" }));
    await user.click(submit());
    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByRole("textbox", { name: /Note \(required/ })).toHaveAttribute("aria-invalid", "true");
  });

  it("a failed request shows the inline error and keeps the reason and note", async () => {
    const user = userEvent.setup();
    const { onSubmit, submit } = setup(vi.fn(async () => false));
    await user.click(screen.getByRole("radio", { name: "Suspected fraudulent uploads" }));
    await user.type(screen.getByRole("textbox"), "Same card on 40 orders");
    await user.click(submit());
    expect(onSubmit).toHaveBeenCalledWith("Suspected fraudulent uploads — Same card on 40 orders");
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/Couldn.t finish/));
    expect(screen.getByRole("radio", { name: "Suspected fraudulent uploads" })).toBeChecked();
    expect(screen.getByRole("textbox")).toHaveValue("Same card on 40 orders");
    expect(submit()).toBeEnabled();
  });
});
