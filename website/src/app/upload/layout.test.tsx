import { act, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useAuthStore } from "@/store/auth-store";
import type { User } from "@/types/user";
import UploadLayout from "./layout";

// The role check is ProtectedRoute's own job (tested there); here it lets the page through.
vi.mock("@/components/auth/protected-route", () => ({ ProtectedRoute: ({ children }: { children: ReactNode }) => <>{children}</> }));
vi.mock("@/lib/api-account", () => ({ acceptTerms: vi.fn() }));
vi.mock("@/hooks/use-auth", () => ({ useAuth: () => ({ logout: vi.fn() }) }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace: vi.fn() }) }));

const photographer = (termsAcceptanceRequired: boolean): User => ({
  id: "u-1",
  email: "p@example.com",
  name: "P",
  role: "PHOTOGRAPHER",
  createdAt: "2026-01-01T00:00:00Z",
  termsAcceptanceRequired,
});

afterEach(() => act(() => useAuthStore.getState().logout()));

// RV-L5 review: /upload/[eventId] sits outside the dashboard layout, so it mounts the prompt itself.
describe("UploadLayout", () => {
  it("shows the Terms prompt to a photographer who must accept", () => {
    useAuthStore.getState().setUser(photographer(true));
    render(<UploadLayout><p>upload page</p></UploadLayout>);
    expect(screen.getByRole("dialog")).toHaveTextContent("Please accept the updated Terms");
  });

  it("renders the page with no prompt for a photographer who need not accept", () => {
    useAuthStore.getState().setUser(photographer(false));
    render(<UploadLayout><p>upload page</p></UploadLayout>);
    expect(screen.getByText("upload page")).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
