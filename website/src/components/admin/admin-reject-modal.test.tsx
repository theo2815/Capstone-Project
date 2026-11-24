import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AdminRejectModal } from "./admin-reject-modal";

// AX-28 (PD6, UA5): no reason is ever sent that the admin did not pick.
function setup(onSubmit = vi.fn<(reason: string) => Promise<boolean>>(async () => true)) {
  render(<AdminRejectModal open onClose={() => {}} onSubmit={onSubmit} photographerName="Ana" />);
  return { onSubmit, submit: () => screen.getByRole("button", { name: "Send back" }) };
}

describe("AdminRejectModal (AX-28)", () => {
  afterEach(cleanup);

  // FT-33 (C-59, C-31; W33.1, W33.2): the note names who reads it, and the off button says why.
  it("names who reads the note and says why the button is off until a reason is picked", async () => {
    const user = userEvent.setup();
    setup();
    expect(screen.getByRole("textbox")).toHaveAccessibleDescription("Seen by: the photographer.");
    expect(screen.getByText("Choose a reason to continue.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send back" })).toHaveAccessibleDescription("Choose a reason to continue.");
    await user.click(screen.getAllByRole("radio")[0]!);
    expect(screen.queryByText("Choose a reason to continue.")).not.toBeInTheDocument();
  });

  it("opens with no reason checked and Send back disabled", () => {
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
    await user.click(screen.getByRole("radio", { name: "Cover image quality too low" }));
    await user.type(screen.getByRole("textbox"), "Same card on 40 orders");
    await user.click(submit());
    expect(onSubmit).toHaveBeenCalledWith("Cover image quality too low — Same card on 40 orders");
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/Couldn.t finish/));
    expect(screen.getByRole("radio", { name: "Cover image quality too low" })).toBeChecked();
    expect(screen.getByRole("textbox")).toHaveValue("Same card on 40 orders");
    expect(submit()).toBeEnabled();
  });
});

// FT-13 (C-03): the intro says what happens to the profile (wording rows W13.3, W13.15).
it("FT-13: the send-back and reset intros say what state the profile is in afterwards", async () => {
  const { AdminResetVerificationModal } = await import("./admin-reset-verification-modal");
  setup();
  expect(screen.getByRole("dialog")).toHaveTextContent("Ana gets your reason in their inbox. Their profile goes back to Incomplete so they can fix it. When they send it again, it waits here for your approval.");
  cleanup();
  // FU-6 (U6.2): a send-back of two or more speaks of them all; one keeps W13.3.
  render(<AdminRejectModal open onClose={() => {}} onSubmit={async () => true} photographerName="2 photographers" count={2} />);
  expect(screen.getByRole("dialog")).toHaveTextContent("2 photographers get your reason in their inbox. Their profiles go back to Incomplete so they can fix them. When they send them again, they wait here for your approval.");
  cleanup();
  render(<AdminRejectModal open onClose={() => {}} onSubmit={async () => true} photographerName="1 photographer" count={1} />);
  expect(screen.getByRole("dialog")).toHaveTextContent("1 photographer gets your reason in their inbox. Their profile goes back to Incomplete so they can fix it. When they send it again, it waits here for your approval.");
  cleanup();
  render(<AdminResetVerificationModal open onClose={() => {}} onSubmit={async () => true} photographerName="Ana" />);
  expect(screen.getByRole("dialog")).toHaveTextContent("Ana goes back to Incomplete and gets your reason in their inbox. Until they send their profile again and you approve it, they cannot upload and their public page does not open. Photos already sold keep working.");
});
