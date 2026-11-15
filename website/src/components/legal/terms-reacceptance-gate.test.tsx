import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api";
import { LEGAL_VERSION } from "@/lib/legal/version";
import { useAuthStore } from "@/store/auth-store";
import type { User } from "@/types/user";
import { TermsReacceptanceGate } from "./terms-reacceptance-gate";

const acceptTerms = vi.fn();
const logout = vi.fn();
const replace = vi.fn();
vi.mock("@/lib/api-account", () => ({ acceptTerms: (v: string) => acceptTerms(v) }));
vi.mock("@/hooks/use-auth", () => ({ useAuth: () => ({ logout }) }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace }) }));

const photographer = (required: boolean): User => ({
  id: "u-1",
  email: "p@example.com",
  name: "P",
  role: "PHOTOGRAPHER",
  createdAt: "2026-01-01T00:00:00Z",
  termsVersion: "1.5",
  legalCurrentVersion: LEGAL_VERSION,
  termsAcceptanceRequired: required,
});

const TITLE = "Please accept the updated Terms";

beforeEach(() => {
  acceptTerms.mockReset();
  logout.mockReset();
  replace.mockReset();
});
afterEach(() => act(() => useAuthStore.getState().logout()));

describe("TermsReacceptanceGate", () => {
  it("shows the prompt when acceptance is required", () => {
    useAuthStore.getState().setUser(photographer(true));
    render(<TermsReacceptanceGate />);
    expect(screen.getByRole("dialog")).toHaveTextContent(TITLE);
    expect(screen.getByRole("dialog")).toHaveTextContent(`accept the new Terms (version ${LEGAL_VERSION}).`);
    expect(screen.getByRole("link", { name: "Read the Terms of Service" })).toHaveAttribute("target", "_blank");
    expect(screen.getByRole("link", { name: "Read the Privacy Policy" })).toHaveAttribute("href", "/privacy");
    expect(screen.getByRole("button", { name: "I accept" })).toBeInTheDocument();
  });

  it("shows nothing when it is not", () => {
    useAuthStore.getState().setUser(photographer(false));
    render(<TermsReacceptanceGate />);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("Accept sends the current legal version and closes after the server answers", async () => {
    useAuthStore.getState().setUser(photographer(true));
    let answer!: (u: User) => void;
    acceptTerms.mockReturnValue(new Promise<User>((r) => (answer = r)));
    render(<TermsReacceptanceGate />);
    fireEvent.click(screen.getByRole("button", { name: "I accept" }));
    expect(acceptTerms).toHaveBeenCalledWith(LEGAL_VERSION);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    await act(async () => answer({ ...photographer(false), termsVersion: LEGAL_VERSION }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(useAuthStore.getState().user?.termsVersion).toBe(LEGAL_VERSION);
  });

  it("a refused acceptance shows the reload message and keeps the prompt", async () => {
    useAuthStore.getState().setUser(photographer(true));
    const message = "The Terms were updated. Please reload the page and read them again.";
    acceptTerms.mockRejectedValue(new ApiError([{ code: "CONSENT_VERSION_MISMATCH", message }], 409));
    render(<TermsReacceptanceGate />);
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "I accept" })));
    expect(screen.getByRole("dialog")).toHaveTextContent(message);
    expect(screen.getByRole("button", { name: "I accept" })).toBeEnabled();
  });

  it("Escape does not close it", () => {
    useAuthStore.getState().setUser(photographer(true));
    render(<TermsReacceptanceGate />);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("Sign out signs the photographer out and goes to the login page", () => {
    useAuthStore.getState().setUser(photographer(true));
    render(<TermsReacceptanceGate />);
    fireEvent.click(screen.getByRole("button", { name: "Sign out" }));
    expect(logout).toHaveBeenCalled();
    expect(replace).toHaveBeenCalledWith("/login");
  });

  it("keeps Tab focus inside the prompt", () => {
    useAuthStore.getState().setUser(photographer(true));
    render(<TermsReacceptanceGate />);
    const signOut = screen.getByRole("button", { name: "Sign out" });
    signOut.focus();
    fireEvent.keyDown(document, { key: "Tab" });
    expect(screen.getByRole("dialog").contains(document.activeElement)).toBe(true);
    expect(document.activeElement).not.toBe(signOut);
  });
});
