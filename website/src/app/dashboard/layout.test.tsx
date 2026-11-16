import { act, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useAuthStore } from "@/store/auth-store";
import type { User } from "@/types/user";
import DashboardLayout from "./layout";

// The role check is ProtectedRoute's own job (tested there); the shell's chrome is not under test here.
vi.mock("@/components/auth/protected-route", () => ({ ProtectedRoute: ({ children }: { children: ReactNode }) => <>{children}</> }));
vi.mock("@/components/dashboard/dashboard-shell", () => ({ DashboardShell: ({ children }: { children: ReactNode }) => <main>{children}</main> }));
vi.mock("@/lib/api-account", () => ({ acceptTerms: vi.fn() }));
vi.mock("@/hooks/use-auth", () => ({ useAuth: () => ({ logout: vi.fn() }) }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace: vi.fn() }), usePathname: () => "/dashboard" }));

const photographer = (termsAcceptanceRequired: boolean): User => ({
  id: "u-1",
  email: "p@example.com",
  name: "P",
  role: "PHOTOGRAPHER",
  createdAt: "2026-01-01T00:00:00Z",
  termsAcceptanceRequired,
});

afterEach(() => act(() => useAuthStore.getState().logout()));

// RVR-7: the dashboard layout is where most photographers meet the Terms prompt (RV-L5).
describe("DashboardLayout", () => {
  it("shows the Terms prompt to a photographer who must accept", () => {
    useAuthStore.getState().setUser(photographer(true));
    render(<DashboardLayout><p>dashboard page</p></DashboardLayout>);
    expect(screen.getByRole("dialog")).toHaveTextContent("Please accept the updated Terms");
  });

  it("renders the dashboard with no prompt otherwise", () => {
    useAuthStore.getState().setUser(photographer(false));
    render(<DashboardLayout><p>dashboard page</p></DashboardLayout>);
    expect(screen.getByText("dashboard page")).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByText("dashboard page").closest("[inert]")).toBeNull();
  });

  it("the page behind is inert while the prompt is open", () => {
    useAuthStore.getState().setUser(photographer(true));
    render(<DashboardLayout><p>dashboard page</p></DashboardLayout>);
    expect(screen.getByRole("main").closest("[inert]")).not.toBeNull();
    expect(screen.getByRole("dialog").closest("[inert]")).toBeNull();
  });
});
