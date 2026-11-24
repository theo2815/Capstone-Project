import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AdminSuspendModal, RUNNER_SUSPEND_REASONS } from "./admin-suspend-modal";

// AX-28 (PD6, UA5): no reason is ever sent that the admin did not pick.
function setup(onSubmit = vi.fn<(reason: string) => Promise<boolean>>(async () => true)) {
  render(<AdminSuspendModal open onClose={() => {}} onSubmit={onSubmit} photographerName="Ana" />);
  return { onSubmit, submit: () => screen.getByRole("button", { name: "Suspend" }) };
}

describe("AdminSuspendModal (AX-28)", () => {
  afterEach(cleanup);

  // FT-33 (C-59, C-31; W33.1, W33.2): the note names who reads it, and the off button says why.
  it("names who reads the note and says why the button is off until a reason is picked", async () => {
    const user = userEvent.setup();
    setup();
    expect(screen.getByRole("textbox")).toHaveAccessibleDescription("Seen by: the suspended person, once they can sign in again.");
    expect(screen.getByText("Choose a reason to continue.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Suspend" })).toHaveAccessibleDescription("Choose a reason to continue.");
    await user.click(screen.getAllByRole("radio")[0]!);
    expect(screen.queryByText("Choose a reason to continue.")).not.toBeInTheDocument();
  });

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

describe("AdminSuspendModal reasons (AX-72)", () => {
  afterEach(cleanup);

  const PHOTOGRAPHER_LABELS = [
    "Watermark or branding violates policy",
    "Suspected fraudulent uploads",
    "Many payment disputes from buyers",
    "Multiple unresolved runner complaints",
    "Compliance review requested",
    "Other (see note)",
  ];
  const radioLabels = () => screen.getAllByRole("radio").map((r) => r.closest("label")!.textContent!.trim());

  it("offers Theo's six runner reasons word for word, and sends the picked label", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn<(reason: string) => Promise<boolean>>(async () => true);
    render(<AdminSuspendModal open onClose={() => {}} onSubmit={onSubmit} photographerName="Ana" reasons={RUNNER_SUSPEND_REASONS} />);
    expect(radioLabels()).toEqual([
      "Abusive or false reports against photographers",
      "Repeated false refund or dispute claims",
      "Suspected payment fraud or chargebacks",
      "Using someone else's selfie or account",
      "Compliance review requested",
      "Other (see note)",
    ]);
    expect(screen.queryByText("Watermark or branding violates policy")).toBeNull();
    await user.click(screen.getByRole("radio", { name: "Repeated false refund or dispute claims" }));
    await user.click(screen.getByRole("button", { name: "Suspend" }));
    expect(onSubmit).toHaveBeenCalledWith("Repeated false refund or dispute claims");
  });

  it("keeps the photographer reasons when no list is passed", () => {
    setup();
    expect(radioLabels()).toEqual(PHOTOGRAPHER_LABELS);
  });
});
