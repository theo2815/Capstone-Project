import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { ConnectDesktop } from "./connect-desktop";

const mocks = vi.hoisted(() => ({
  params: new URLSearchParams(""),
  mint: vi.fn(),
  navigate: vi.fn(),
  replace: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useSearchParams: () => mocks.params,
  useRouter: () => ({ replace: mocks.replace, push: vi.fn() }),
}));
vi.mock("@/lib/api-account", () => ({
  mintDesktopAuthCode: (...args: unknown[]) => mocks.mint(...args),
}));
vi.mock("@/lib/desktop-connect", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/desktop-connect")>()),
  navigateTo: (url: string) => mocks.navigate(url),
}));
vi.mock("@/components/layout/site-header", () => ({ SiteHeader: () => null }));

const STATE = "s".repeat(43);
const CHALLENGE = "c".repeat(43);
const VALID = `state=${STATE}&challenge=${CHALLENGE}`;

const wrapper = ({ children }: { children: ReactNode }) => (
  <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
);

function signedIn(role: "PHOTOGRAPHER" | "RUNNER") {
  useAuthStore.getState().setUser({ id: "u1", email: "p@test.local", name: "P", role, createdAt: "2026-10-10" });
}

afterEach(() => {
  vi.clearAllMocks();
  useAuthStore.getState().logout();
});

describe("ConnectDesktop", () => {
  it("rejects a malformed link before anything else", () => {
    mocks.params = new URLSearchParams(`state=short&challenge=${CHALLENGE}`);
    signedIn("PHOTOGRAPHER");
    render(<ConnectDesktop />, { wrapper });
    expect(screen.getByText("This connect link is not valid.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Connect" })).not.toBeInTheDocument();
  });

  it("sends a signed-out visitor to login and brings the link back", async () => {
    window.history.pushState({}, "", `/connect/desktop?${VALID}`);
    mocks.params = new URLSearchParams(VALID);
    useAuthStore.getState().logout();
    render(<ConnectDesktop />, { wrapper });
    await waitFor(() =>
      expect(mocks.replace).toHaveBeenCalledWith(`/login?redirect=${encodeURIComponent(`/connect/desktop?${VALID}`)}`),
    );
    expect(screen.queryByRole("button", { name: "Connect" })).not.toBeInTheDocument();
  });

  it("tells a runner to use a photographer account", () => {
    mocks.params = new URLSearchParams(VALID);
    signedIn("RUNNER");
    render(<ConnectDesktop />, { wrapper });
    expect(screen.getByText("Connecting needs a photographer account.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sign out" })).toBeInTheDocument();
    expect(mocks.mint).not.toHaveBeenCalled();
  });

  it("mints a code for a photographer and hands it to the app", async () => {
    mocks.params = new URLSearchParams(VALID);
    signedIn("PHOTOGRAPHER");
    mocks.mint.mockResolvedValue({ code: "k".repeat(43), expiresInSeconds: 120 });
    render(<ConnectDesktop />, { wrapper });

    fireEvent.click(screen.getByRole("button", { name: "Connect" }));

    await waitFor(() =>
      expect(mocks.navigate).toHaveBeenCalledWith(`batchmyphotos://quickpitik/callback?code=${"k".repeat(43)}&state=${STATE}`),
    );
    expect(mocks.mint).toHaveBeenCalledWith(CHALLENGE);
    expect(screen.getByText("Back to BatchMyPhotos.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });

  it("mints a fresh code on Try again", async () => {
    mocks.params = new URLSearchParams(VALID);
    signedIn("PHOTOGRAPHER");
    mocks.mint.mockResolvedValueOnce({ code: "a".repeat(43), expiresInSeconds: 120 })
      .mockResolvedValueOnce({ code: "b".repeat(43), expiresInSeconds: 120 });
    render(<ConnectDesktop />, { wrapper });

    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    fireEvent.click(await screen.findByRole("button", { name: "Try again" }));

    await waitFor(() => expect(mocks.navigate).toHaveBeenCalledTimes(2));
    expect(mocks.navigate.mock.calls[1][0]).toContain(`code=${"b".repeat(43)}`);
  });

  it("shows the rate-limit message and keeps the button usable", async () => {
    mocks.params = new URLSearchParams(VALID);
    signedIn("PHOTOGRAPHER");
    mocks.mint.mockRejectedValue(new ApiError([{ code: "RATE_LIMITED", message: "slow down" }], 429));
    render(<ConnectDesktop />, { wrapper });

    fireEvent.click(screen.getByRole("button", { name: "Connect" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Too many attempts. Wait a minute and try again.");
    expect(screen.getByRole("button", { name: "Connect" })).toBeEnabled();
    expect(mocks.navigate).not.toHaveBeenCalled();
  });
});
