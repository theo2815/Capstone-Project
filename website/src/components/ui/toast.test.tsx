import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { ToastContainer } from "./toast";
import { Modal } from "./modal";

afterEach(cleanup);

// AX-53: an error toast at the bottom covered the modal's footer buttons.
it("toasts sit at the top while a dialog is open and at the bottom otherwise", () => {
  const { rerender } = render(
    <>
      <ToastContainer />
      <Modal isOpen={false} onClose={() => {}} title="Suspend">
        <button type="button">Suspend</button>
      </Modal>
    </>,
  );
  const region = screen.getByRole("region", { name: "Notifications" });
  expect(region).toHaveClass("bottom-4");
  expect(region).not.toHaveClass("top-4");

  rerender(
    <>
      <ToastContainer />
      <Modal isOpen onClose={() => {}} title="Suspend">
        <button type="button">Suspend</button>
      </Modal>
    </>,
  );
  expect(region).toHaveClass("top-4");
  expect(region).not.toHaveClass("bottom-4");

  rerender(
    <>
      <ToastContainer />
      <Modal isOpen={false} onClose={() => {}} title="Suspend">
        <button type="button">Suspend</button>
      </Modal>
    </>,
  );
  expect(region).toHaveClass("bottom-4");
});
