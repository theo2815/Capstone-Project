import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import {
  AdminFieldHint,
  AdminFormSection,
  AdminRadioGroup,
  AdminTextarea,
  AdminTextInput,
} from "./admin-form-fields";

afterEach(cleanup);

const noop = () => {};

it("ties an error to the text input and marks it invalid", () => {
  render(
    <AdminTextInput
      id="name"
      label="Name"
      value=""
      onChange={noop}
      hint="Shown on the event page."
      error="Name is required."
    />,
  );
  const input = screen.getByRole("textbox", { name: /Name/ });
  expect(input).toHaveAttribute("aria-invalid", "true");
  expect(input).toHaveAccessibleDescription(/Shown on the event page\..*Name is required\./);
});

it("leaves a field without an error unmarked", () => {
  render(<AdminTextInput id="name" label="Name" value="" onChange={noop} />);
  const input = screen.getByRole("textbox", { name: "Name" });
  expect(input).not.toHaveAttribute("aria-invalid");
  expect(input).not.toHaveAttribute("aria-describedby");
});

it("marks required fields and says so in the label", () => {
  render(
    <>
      <AdminTextInput id="name" label="Name" value="" onChange={noop} required />
      <AdminTextarea id="note" label="Note" value="" onChange={noop} maxLength={200} required />
    </>,
  );
  expect(screen.getByRole("textbox", { name: "Name (required)" })).toBeRequired();
  expect(screen.getByRole("textbox", { name: "Note (required)" })).toBeRequired();
});

it("ties an error to the textarea", () => {
  render(
    <AdminTextarea
      id="note"
      label="Note"
      value=""
      onChange={noop}
      maxLength={200}
      error="Add a note."
    />,
  );
  const area = screen.getByRole("textbox", { name: "Note" });
  expect(area).toHaveAttribute("aria-invalid", "true");
  expect(area).toHaveAccessibleDescription("Add a note.");
});

// FT-33 (C-59): a textarea's hint sits under it and is read out with it.
it("shows a textarea hint and lists its id in aria-describedby", () => {
  render(<AdminTextarea id="note" label="Note" value="" onChange={noop} maxLength={200} hint="Seen by: the photographer." />);
  const area = screen.getByRole("textbox", { name: "Note" });
  expect(screen.getByText("Seen by: the photographer.")).toHaveAttribute("id", "note-hint");
  expect(area.getAttribute("aria-describedby")?.split(" ")).toContain("note-hint");
  expect(area).toHaveAccessibleDescription("Seen by: the photographer.");
});

it("groups a radio set under its section label", () => {
  render(
    <AdminFormSection label="Reason">
      <AdminRadioGroup
        name="reason"
        value="a"
        onChange={noop}
        options={[
          { value: "a", label: "Spam" },
          { value: "b", label: "Other" },
        ]}
      />
    </AdminFormSection>,
  );
  const group = screen.getByRole("group", { name: "Reason" });
  expect(group.tagName).toBe("FIELDSET");
  expect(screen.getByRole("radio", { name: "Spam" }).closest("label")).toHaveClass("min-h-[44px]");
});

it("FU-13: a text input can be disabled", () => {
  render(<AdminTextInput id="name" label="Name" value="" onChange={noop} disabled />);
  expect(screen.getByRole("textbox", { name: "Name" })).toBeDisabled();
});

it("FU-13: a date input keeps its type and earliest date", () => {
  render(<AdminTextInput id="review" label="Review date" value="" onChange={noop} type="date" min="2026-12-01" />);
  const input = screen.getByLabelText("Review date");
  expect(input).toHaveAttribute("type", "date");
  expect(input).toHaveAttribute("min", "2026-12-01");
});

it("FU-13: a field left empty can be marked invalid without a message", async () => {
  const left: string[] = [];
  const { rerender } = render(<AdminTextInput id="reason" label="Reason" value="" onChange={noop} onBlur={() => left.push("reason")} />);
  const input = screen.getByRole("textbox", { name: "Reason" });
  expect(input).not.toHaveAttribute("aria-invalid");
  input.focus();
  input.blur();
  expect(left).toEqual(["reason"]);
  rerender(<AdminTextInput id="reason" label="Reason" value="" onChange={noop} invalid />);
  expect(screen.getByRole("textbox", { name: "Reason" })).toHaveAttribute("aria-invalid", "true");
});

it("FU-13: a calm hint reads as plain text; a warning hint keeps the warning colour", () => {
  render(
    <>
      <AdminTextInput id="ref" label="Reference" value="" onChange={noop} hint="At least 4 characters." hintTone="calm" />
      <AdminTextInput id="price" label="Price" value="" onChange={noop} hint="Enter a price above zero." />
      <AdminFieldHint id="free" tone="calm">Only admins see this.</AdminFieldHint>
    </>,
  );
  const calm = screen.getByText("At least 4 characters.");
  expect(calm).toHaveClass("font-sans", "text-sm", "text-slate");
  expect(calm).not.toHaveClass("text-warning-text", "uppercase");
  expect(screen.getByRole("textbox", { name: "Reference" })).toHaveAccessibleDescription("At least 4 characters.");
  expect(screen.getByText("Enter a price above zero.")).toHaveClass("text-warning-text", "uppercase");
  expect(screen.getByText("Only admins see this.")).toHaveClass("text-slate");
});
