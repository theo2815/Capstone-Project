import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import AdminNotFound from "./not-found";

vi.mock("next/link", () => ({ default: ({ children, href, className }: { children: React.ReactNode; href: string; className?: string }) => <a href={href} className={className}>{children}</a> }));

// FT-4: a bad admin link stays inside the admin and offers the way back.
it("says the admin page does not exist and links to the Inbox", () => {
  render(<AdminNotFound />);
  expect(screen.getByRole("heading", { level: 1, name: "Not found." })).toBeInTheDocument();
  expect(screen.getByText("Admin")).toBeInTheDocument();
  expect(screen.getByText("This admin page does not exist. The link may be old, or the item may have been removed.")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Go to the Inbox" })).toHaveAttribute("href", "/admin/inbox");
});
