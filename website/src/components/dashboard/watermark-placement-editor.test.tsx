import { useState } from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { WatermarkPlacementEditor } from "./watermark-placement-editor";
import { DEFAULT_WATERMARK_PLACEMENT } from "@/store/photographer-settings-store";

let previewWidth = 400;
let resize: () => void;

beforeEach(() => {
  previewWidth = 400;
  vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockImplementation(() => previewWidth);
  vi.stubGlobal("ResizeObserver", class {
    constructor(callback: () => void) { resize = callback; }
    observe() {}
    disconnect() {}
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function Editor({ editing = true }: { editing?: boolean }) {
  const [placement, setPlacement] = useState(DEFAULT_WATERMARK_PLACEMENT);
  return <WatermarkPlacementEditor dataUrl="data:image/png;base64,AA==" editing={editing} placement={placement} onChange={setPlacement} />;
}

it("offers focusable native position controls for each orientation and disables them outside Edit", async () => {
  const user = userEvent.setup();
  const { rerender } = render(<Editor />);
  const horizontal = screen.getByRole("slider", { name: "Horizontal position" });
  const vertical = screen.getByRole("slider", { name: "Vertical position" });
  await user.tab(); // landscape
  await user.tab(); // portrait
  await user.tab();
  expect(horizontal).toHaveFocus();
  fireEvent.change(horizontal, { target: { value: "0.25" } });
  fireEvent.change(vertical, { target: { value: "0.75" } });
  expect(horizontal).toHaveAttribute("aria-valuetext", "landscape, 25% from left");

  fireEvent.click(screen.getByRole("tab", { name: "Portrait" }));
  expect(horizontal).toHaveValue("1");
  expect(vertical).toHaveValue("1");
  fireEvent.change(horizontal, { target: { value: "0.5" } });
  fireEvent.click(screen.getByRole("tab", { name: "Landscape" }));
  expect(horizontal).toHaveValue("0.25");
  expect(vertical).toHaveValue("0.75");
  rerender(<Editor editing={false} />);
  expect(horizontal).toBeDisabled();
  expect(vertical).toBeDisabled();
});

it("scales backend pixels for screen size and uses fitted geometry during drag", () => {
  render(<Editor />);
  fireEvent.click(screen.getByRole("tab", { name: "Portrait" }));
  fireEvent.change(screen.getByRole("slider", { name: "Size" }), { target: { value: "0.05" } });
  const logo = screen.getByRole("img", { name: "Watermark position preview" });
  expect(parseFloat(logo.style.width)).toBeCloseTo(48 * 400 / 853);
  act(() => { previewWidth = 200; resize(); });
  expect(parseFloat(logo.style.width)).toBeCloseTo(48 * 200 / 853);

  fireEvent.click(screen.getByRole("tab", { name: "Landscape" }));
  Object.defineProperties(logo, { naturalWidth: { value: 500 }, naturalHeight: { value: 1000 } });
  fireEvent.load(logo);
  fireEvent.change(screen.getByRole("slider", { name: "Size" }), { target: { value: "0.4" } });
  expect(parseFloat(logo.style.width)).toBeCloseTo(394 * 200 / 1280);
  expect(parseFloat(logo.style.height)).toBeCloseTo(789 * 200 / 1280);

  vi.stubGlobal("PointerEvent", MouseEvent);
  logo.setPointerCapture = vi.fn();
  logo.releasePointerCapture = vi.fn();
  fireEvent.pointerDown(logo, { clientX: parseFloat(logo.style.left) + 1, clientY: parseFloat(logo.style.top) + 1 });
  fireEvent.pointerMove(logo, { clientX: 6, clientY: 6 });
  fireEvent.pointerUp(logo);
  expect(screen.getByRole("slider", { name: "Horizontal position" })).toHaveValue("0");
  expect(screen.getByRole("slider", { name: "Vertical position" })).toHaveValue("1");
  expect(parseFloat(logo.style.left)).toBeCloseTo(32 * 200 / 1280);
  expect(parseFloat(logo.style.top)).toBeCloseTo(32 * 200 / 1280);
});
