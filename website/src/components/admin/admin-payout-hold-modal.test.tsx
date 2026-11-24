import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AdminPayoutHoldModal } from "./admin-payout-hold-modal";

// AX-28 (PAY8, UA5): no reason is ever sent that the admin did not pick.
function setup(onSubmit = vi.fn<(reason: string) => Promise<boolean>>(async () => true)) {
  render(<AdminPayoutHoldModal open onClose={() => {}} onSubmit={onSubmit} payoutIds={["PO-1"]} totalAmount={1200} />);
  return { onSubmit, submit: () => screen.getByRole("button", { name: "Hold payout" }) };
}

describe("AdminPayoutHoldModal (AX-28)", () => {
  afterEach(cleanup);

  // FT-33 (C-59, C-31; W33.1, W33.2): the note names who reads it, and the off button says why.
  it("names who reads the note and says why the button is off until a reason is picked", async () => {
    const user = userEvent.setup();
    setup();
    expect(screen.getByRole("textbox")).toHaveAccessibleDescription("Seen by: the photographer, exactly as written.");
    expect(screen.getByText("Choose a reason to continue.")).toBeInTheDocument();
    await user.click(screen.getAllByRole("radio")[0]!);
    expect(screen.queryByText("Choose a reason to continue.")).not.toBeInTheDocument();
  });

  it("opens with no reason checked and Hold payout disabled", () => {
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
    await user.click(screen.getByRole("radio", { name: "We need to review your account first" }));
    await user.type(screen.getByRole("textbox"), "Same card on 40 orders");
    await user.click(submit());
    expect(onSubmit).toHaveBeenCalledWith("We need to review your account first — Same card on 40 orders");
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/Couldn.t finish/));
    expect(screen.getByRole("radio", { name: "We need to review your account first" })).toBeChecked();
    expect(screen.getByRole("textbox")).toHaveValue("Same card on 40 orders");
    expect(submit()).toBeEnabled();
  });

  it("says payout, not cycle", () => {
    setup();
    expect(screen.getByRole("dialog")).not.toHaveTextContent(/cycle/i);
    expect(screen.getByText(/This payout/)).toBeInTheDocument();
  });

  // FT-16 (C-19): the photographer reads the chosen reason word for word, so no internal words.
  it("the reasons are in everyday words, and the intro says the photographer reads them as written", async () => {
    const user = userEvent.setup();
    const { onSubmit, submit } = setup();
    expect(screen.getAllByRole("radio")).toHaveLength(6);
    expect(screen.getByRole("dialog")).not.toHaveTextContent(/chargeback|compliance|payments lead/i);
    expect(screen.getByRole("dialog")).toHaveTextContent(
      "The photographer reads the reason below exactly as it is written, with your note. You can approve or reject the payout later.",
    );
    await user.click(screen.getByRole("radio", { name: "We need to review your account first" }));
    await user.click(submit());
    expect(onSubmit.mock.calls[0]![0]).toMatch(/^We need to review your account first/);
  });

  // FU-9 (FT-33): the server refuses a reason over 255 characters, so the note leaves room for the longest label.
  it("a long note with the longest reason still sends at most 255 characters", async () => {
    const user = userEvent.setup();
    const { onSubmit, submit } = setup();
    await user.click(screen.getByRole("radio", { name: "Your payout account details do not match our records" }));
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "x".repeat(280) } });
    await user.click(submit());
    expect(onSubmit.mock.calls[0]![0].length).toBeLessThanOrEqual(255);
  });

  // FU-9 (FT-11): the "Other" error goes away once the admin picks another reason.
  it("the Other note error clears when the reason changes", async () => {
    const user = userEvent.setup();
    const { submit } = setup();
    await user.click(screen.getByRole("radio", { name: "Other (see note)" }));
    await user.click(submit());
    expect(screen.getByText("Add a note: Other needs one.")).toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: "We need to review your account first" }));
    await user.click(screen.getByRole("radio", { name: "Other (see note)" }));
    expect(screen.queryByText("Add a note: Other needs one.")).not.toBeInTheDocument();
  });
});
