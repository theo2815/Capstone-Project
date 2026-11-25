import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { SiteHeader } from "@/components/layout/site-header";
import { useAuthStore } from "@/store/auth-store";

const nav = vi.hoisted(() => ({ pathname: "/" }));
vi.mock("next/navigation", () => ({ usePathname: () => nav.pathname }));
vi.mock("@/components/layout/notification-bell", () => ({ NotificationBell: () => null }));
vi.mock("@/components/layout/runner-notification-bell", () => ({ RunnerNotificationBell: () => null }));
vi.mock("@/components/layout/user-menu", () => ({ UserMenu: () => null }));

afterEach(() => {
  nav.pathname = "/";
  useAuthStore.setState({ user: null, isAuthenticated: false, isLoading: false });
});

function signInHref() {
  useAuthStore.setState({ user: null, isAuthenticated: false, isLoading: false });
  render(<QueryClientProvider client={new QueryClient()}><SiteHeader /></QueryClientProvider>);
  return screen.getByRole("link", { name: "Sign in" }).getAttribute("href");
}

it("signing in from an event page comes back to that event", () => {
  nav.pathname = "/events/cebu-run";
  expect(signInHref()).toBe("/login?redirect=%2Fevents%2Fcebu-run");
});

it("signing in from anywhere else keeps the plain sign-in link", () => {
  nav.pathname = "/runners";
  expect(signInHref()).toBe("/login");
});

// FU-23: every header link is a 44 px target at phone and tablet widths (min-h-11 = 44 px).
it("the home, Events and Sign in links are each at least 44 px tall", () => {
  render(<QueryClientProvider client={new QueryClient()}><SiteHeader /></QueryClientProvider>);
  for (const name of ["Back to QuickPitik home", "Events", "Sign in"]) {
    expect(screen.getByRole("link", { name })).toHaveClass("min-h-11");
  }
});

it("the hover underline sits on the word, not the taller link", () => {
  render(<QueryClientProvider client={new QueryClient()}><SiteHeader /></QueryClientProvider>);
  for (const name of ["Events", "Sign in"]) {
    const link = screen.getByRole("link", { name });
    expect(link).not.toHaveClass("nav-link");
    expect(link.querySelector(".nav-link")).toHaveTextContent(name);
  }
});

it("the events list itself keeps the plain sign-in link", () => {
  nav.pathname = "/events";
  expect(signInHref()).toBe("/login");
});
