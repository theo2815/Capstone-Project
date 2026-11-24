import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { AdminConfirmModal, AdminFormModal } from "./admin-form-modal";

function form(props: Partial<Parameters<typeof AdminFormModal>[0]> = {}) {
  return render(
    <AdminFormModal open onClose={() => {}} onSubmit={() => {}} title="Suspend" submitLabel="Suspend" {...props}>
      <p>Body</p>
    </AdminFormModal>,
  );
}

describe("AdminFormModal footer (AX-3)", () => {
  afterEach(cleanup);

  it("tone danger gives the submit the danger button", () => {
    form({ tone: "danger" });
    const submit = screen.getByRole("button", { name: "Suspend" });
    expect(submit).toHaveClass("text-error", "bg-bone-deep", "min-h-[44px]");
    expect(submit).not.toHaveClass("bg-fresh");
  });

  it("the default tone gives the primary button, and Cancel is 44 px", () => {
    form();
    expect(screen.getByRole("button", { name: "Suspend" })).toHaveClass("bg-fresh", "min-h-[44px]");
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveClass("min-h-[44px]");
  });

  it("an error shows inline as an alert", () => {
    form({ error: "The server said no." });
    expect(screen.getByRole("alert")).toHaveTextContent("The server said no.");
  });

  it("the submit stays disabled while submitting", () => {
    form({ submitting: true, pendingLabel: "Suspending…" });
    expect(screen.getByRole("button", { name: "Suspending…" })).toBeDisabled();
  });
});

describe("AdminConfirmModal footer (AX-3)", () => {
  afterEach(cleanup);

  it("tone danger and error work the same way", () => {
    render(
      <AdminConfirmModal open onClose={() => {}} onConfirm={() => {}} title="Hide" confirmLabel="Confirm hide" tone="danger" error="Failed.">
        <p>Body</p>
      </AdminConfirmModal>,
    );
    expect(screen.getByRole("button", { name: "Confirm hide" })).toHaveClass("text-error", "min-h-[44px]");
    expect(screen.getByRole("alert")).toHaveTextContent("Failed.");
  });
});

describe("AdminFormModal noValidate (FT-22, R-04)", () => {
  afterEach(cleanup);

  it("without the prop the form keeps the browser's own check", () => {
    form();
    expect(document.body.querySelector("form")).not.toHaveAttribute("novalidate");
  });

  it("with the prop the form skips the browser's bubble", () => {
    form({ noValidate: true });
    expect(document.body.querySelector("form")).toHaveAttribute("novalidate");
  });
});
