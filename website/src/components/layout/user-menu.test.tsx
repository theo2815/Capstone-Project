import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { UserMenu } from "@/components/layout/user-menu";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }));
vi.mock("@/hooks/use-auth", () => ({ useAuth: () => ({ logout: vi.fn() }) }));
vi.mock("@/hooks/use-confirmation", () => ({ useConfirmation: () => ({ confirm: vi.fn() }) }));

const runner = { id: "r1", email: "runner@example.test", name: "Runner One", role: "RUNNER" as const, createdAt: "2026-11-01" };

// FU-23: the account menu button is a 44 px target (size-11), not 36 px.
it("the account menu button is 44 px", () => {
  render(<UserMenu user={runner} />);
  expect(screen.getByRole("button", { name: /Account menu/ })).toHaveClass("size-11");
});

// Runner audit P3-6: the menu is announced as a menu, but only Esc worked.
it("moves through the menu with the arrow keys, Home and End", () => {
  render(<UserMenu user={runner} />);
  const trigger = screen.getByRole("button", { name: /Account menu/ });
  fireEvent.click(trigger);
  const items = screen.getAllByRole("menuitem");
  expect(items.length).toBeGreaterThan(2);

  trigger.focus();
  fireEvent.keyDown(document, { key: "ArrowDown" });
  expect(document.activeElement).toBe(items[0]);
  fireEvent.keyDown(document, { key: "ArrowDown" });
  expect(document.activeElement).toBe(items[1]);
  fireEvent.keyDown(document, { key: "ArrowUp" });
  expect(document.activeElement).toBe(items[0]);
  fireEvent.keyDown(document, { key: "ArrowUp" });
  expect(document.activeElement).toBe(items[items.length - 1]); // wraps
  fireEvent.keyDown(document, { key: "Home" });
  expect(document.activeElement).toBe(items[0]);
  fireEvent.keyDown(document, { key: "End" });
  expect(document.activeElement).toBe(items[items.length - 1]);

  fireEvent.keyDown(document, { key: "Escape" });
  expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  expect(document.activeElement).toBe(trigger);
});
