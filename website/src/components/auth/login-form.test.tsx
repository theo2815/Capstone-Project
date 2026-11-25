import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { LoginForm } from "@/components/auth/login-form";
import { useAuthStore } from "@/store/auth-store";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

afterEach(() => { cleanup(); useAuthStore.getState().setPendingMfa(null); });

it("gives the sign-in code box a full keyboard, since it also takes a recovery code", () => {
  useAuthStore.getState().setPendingMfa("mfa-token");
  render(<QueryClientProvider client={new QueryClient()}><LoginForm /></QueryClientProvider>);
  expect(screen.getByLabelText("Authentication code")).toHaveAttribute("inputmode", "text");
});
