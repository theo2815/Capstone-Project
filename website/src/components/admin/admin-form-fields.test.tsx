import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import {
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
