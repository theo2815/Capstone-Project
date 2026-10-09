import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useConfirmation } from "@/hooks/use-confirmation";
import { useConfirmationStore } from "@/store/confirmation-store";
import { ConfirmationOverlay } from "./confirmation-overlay";

function deferred() {
  let resolve!: () => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<void>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function open(onConfirm?: () => void | Promise<void>, id = "first") {
  const resolve = vi.fn();
  act(() => useConfirmationStore.getState().open({ id, config: { title: id, message: "Keep this draft", confirmLabel: "Submit", onConfirm }, resolve }));
  return resolve;
}

describe("confirmation ID ownership", () => {
  beforeEach(() => { useConfirmationStore.setState({ active: null, loading: false, error: null }); });
  afterEach(() => { cleanup(); useConfirmationStore.setState({ active: null, loading: false, error: null }); });

  it("claims rapid Enter and click once before React renders pending", async () => {
    const response = deferred();
    const execute = vi.fn(() => response.promise);
    render(<ConfirmationOverlay />);
    const resolve = open(execute);
    const button = screen.getByRole("button", { name: "Submit" });
    act(() => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
      button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(execute).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Working…" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Working…" })).toHaveAttribute("aria-busy", "true");
    expect(resolve).not.toHaveBeenCalled();
    await act(async () => { response.resolve(); await response.promise; });
    expect(resolve).toHaveBeenCalledExactlyOnceWith(true);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("retains error and draft after rejection and allows one explicit retry", async () => {
    const response = deferred();
    const execute = vi.fn().mockReturnValueOnce(response.promise).mockResolvedValue(undefined);
    render(<ConfirmationOverlay />);
    const resolve = open(execute);
    fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    await act(async () => { response.reject(new Error("Try again")); });
    expect(screen.getByRole("alert")).toHaveTextContent("Try again");
    expect(screen.getByText("Keep this draft")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Submit" })).toBeEnabled();
    expect(resolve).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    await waitFor(() => expect(resolve).toHaveBeenCalledExactlyOnceWith(true));
    expect(execute).toHaveBeenCalledTimes(2);
  });

  it("holds the claim through a synchronous throw so same-turn Enter and click cannot retry", async () => {
    const execute = vi.fn(() => { throw new Error("Refused immediately"); });
    render(<ConfirmationOverlay />);
    const resolve = open(execute);
    const button = screen.getByRole("button", { name: "Submit" });
    await act(async () => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
      button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(execute).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("alert")).toHaveTextContent("Refused immediately");
    expect(resolve).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    await waitFor(() => expect(execute).toHaveBeenCalledTimes(2));
  });

  it("resolves a second open as cancelled while the first owns execution", async () => {
    const response = deferred();
    const execute = vi.fn(() => response.promise);
    render(<ConfirmationOverlay />);
    const first = open(execute);
    fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    const secondExecute = vi.fn();
    const second = open(secondExecute, "second");
    expect(second).toHaveBeenCalledExactlyOnceWith(false);
    expect(useConfirmationStore.getState().active?.id).toBe("first");
    expect(useConfirmationStore.getState().loading).toBe(true);
    expect(secondExecute).not.toHaveBeenCalled();
    await act(async () => { response.resolve(); await response.promise; });
    expect(first).toHaveBeenCalledExactlyOnceWith(true);
  });

  it.each(["success", "failure"])("ignores stale %s after the active request is reset and replaced", async (outcome) => {
    const response = deferred();
    render(<ConfirmationOverlay />);
    const first = open(() => response.promise);
    fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    // An external reset/replacement must not let an older callback own the new dialog.
    act(() => useConfirmationStore.setState({ active: null, loading: false, error: null }));
    const second = open(undefined, "second");
    await act(async () => {
      if (outcome === "success") response.resolve();
      else response.reject(new Error("Old error"));
    });
    expect(useConfirmationStore.getState().active?.id).toBe("second");
    expect(useConfirmationStore.getState().loading).toBe(false);
    expect(screen.queryByRole("alert")).toBeNull();
    expect(second).not.toHaveBeenCalled();
    expect(first).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    expect(second).toHaveBeenCalledExactlyOnceWith(true);
  });

  it("ID-checks dismissal and prevents cancelling an executing owner", async () => {
    const response = deferred();
    render(<ConfirmationOverlay />);
    const first = open(() => response.promise);
    fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    act(() => useConfirmationStore.getState().close("first", false));
    expect(first).not.toHaveBeenCalled();
    expect(useConfirmationStore.getState().active?.id).toBe("first");
    await act(async () => { response.resolve(); await response.promise; });
    const second = open(undefined, "second");
    act(() => useConfirmationStore.getState().close("first", false));
    expect(second).not.toHaveBeenCalled();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(second).toHaveBeenCalledExactlyOnceWith(false);
  });

  it("settles state before resolving so a continuation can open a new confirmation", () => {
    render(<ConfirmationOverlay />);
    const second = vi.fn();
    act(() => useConfirmationStore.getState().open({ id: "first", config: { title: "First", message: "Continue" }, resolve: vi.fn().mockImplementationOnce(() => {
      useConfirmationStore.getState().open({ id: "second", config: { title: "Second", message: "Next" }, resolve: second });
    }) }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm" }));
    expect(useConfirmationStore.getState().active?.id).toBe("second");
    expect(second).not.toHaveBeenCalled();
  });

  it("preserves the non-admin boolean confirmation contract and focus return", async () => {
    const trigger = document.createElement("button");
    document.body.appendChild(trigger); trigger.focus();
    render(<ConfirmationOverlay />);
    const { result } = renderHook(useConfirmation);
    let decision!: Promise<boolean>;
    act(() => { decision = result.current.confirm({ title: "Leave checkout?", message: "Your QR stays valid", confirmLabel: "Leave" }); });
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await expect(decision).resolves.toBe(false);
    expect(document.activeElement).toBe(trigger);
    act(() => { decision = result.current.confirm({ title: "Leave checkout?", message: "Your QR stays valid", confirmLabel: "Leave" }); });
    fireEvent.click(screen.getByRole("button", { name: "Leave" }));
    await expect(decision).resolves.toBe(true);
    trigger.remove();
  });

  it("leaves Enter on the focused Cancel button to the button so it cancels", () => {
    const execute = vi.fn();
    render(<ConfirmationOverlay />);
    const resolve = open(execute);
    const cancel = screen.getByRole("button", { name: "Cancel" });
    cancel.focus();
    // true = not prevented, so the browser still turns this key press into Cancel's click.
    expect(fireEvent.keyDown(cancel, { key: "Enter" })).toBe(true);
    expect(execute).not.toHaveBeenCalled();
    fireEvent.click(cancel);
    expect(resolve).toHaveBeenCalledExactlyOnceWith(false);
  });

  it("does not confirm on a held Enter key", () => {
    const execute = vi.fn();
    render(<ConfirmationOverlay />);
    const resolve = open(execute);
    fireEvent.keyDown(document, { key: "Enter", repeat: true });
    expect(execute).not.toHaveBeenCalled();
    expect(resolve).not.toHaveBeenCalled();
  });

  it("still confirms once on Enter while focus is on the trigger behind the dialog", async () => {
    const trigger = document.createElement("button");
    document.body.appendChild(trigger); trigger.focus();
    const execute = vi.fn();
    render(<ConfirmationOverlay />);
    const resolve = open(execute);
    expect(fireEvent.keyDown(trigger, { key: "Enter" })).toBe(false);
    await waitFor(() => expect(resolve).toHaveBeenCalledExactlyOnceWith(true));
    expect(execute).toHaveBeenCalledTimes(1);
    trigger.remove();
  });
});
