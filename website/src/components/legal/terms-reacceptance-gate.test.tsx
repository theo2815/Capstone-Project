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
// The real Modal still renders; the spy records the props the gate hands it.
const modalProps = vi.fn();
vi.mock("@/components/ui/modal", async (importOriginal) => {
  const real = await importOriginal<typeof import("@/components/ui/modal")>();
  return {
    Modal: (props: Parameters<typeof real.Modal>[0]) => {
      modalProps(props);
      return <real.Modal {...props} />;
    },
  };
});

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
  modalProps.mockReset();
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

  it("Accept sends the bundle's version when the server names none, and closes after the server answers", async () => {
    useAuthStore.getState().setUser({ ...photographer(true), legalCurrentVersion: undefined });
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

  it("Accept sends the version the server named", async () => {
    useAuthStore.getState().setUser({ ...photographer(true), legalCurrentVersion: "9.9" });
    acceptTerms.mockReturnValue(new Promise<User>(() => {}));
    render(<TermsReacceptanceGate />);
    expect(screen.getByRole("dialog")).toHaveTextContent("accept the new Terms (version 9.9).");
    expect(screen.getByRole("dialog")).toHaveTextContent("you agree to the Terms of Service version 9.9.");
    expect(screen.getByRole("dialog")).not.toHaveTextContent(`version ${LEGAL_VERSION}`);
    fireEvent.click(screen.getByRole("button", { name: "I accept" }));
    expect(acceptTerms).toHaveBeenCalledWith("9.9");
  });

  it("a runner or an admin never sees the prompt", () => {
    // As the server sends them: only a photographer is ever marked as required.
    for (const role of ["RUNNER", "ADMIN"] as const) {
      useAuthStore.getState().setUser({ ...photographer(false), role, termsVersion: "1.0" });
      const { unmount } = render(<TermsReacceptanceGate />);
      expect(screen.queryByRole("dialog")).toBeNull();
      unmount();
    }
  });

  it("a network or server failure shows its message, keeps the prompt, and Accept works again", async () => {
    useAuthStore.getState().setUser(photographer(true));
    acceptTerms.mockRejectedValueOnce(new Error("Network error. Check your connection."));
    acceptTerms.mockRejectedValueOnce(new ApiError([{ code: "INTERNAL_ERROR", message: "Something went wrong." }], 500));
    acceptTerms.mockResolvedValueOnce({ ...photographer(false), termsVersion: LEGAL_VERSION });
    render(<TermsReacceptanceGate />);
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "I accept" })));
    expect(screen.getByRole("dialog")).toHaveTextContent("Network error. Check your connection.");
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "I accept" })));
    expect(screen.getByRole("dialog")).toHaveTextContent("Something went wrong.");
    expect(screen.getByRole("button", { name: "I accept" })).toBeEnabled();
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "I accept" })));
    expect(acceptTerms).toHaveBeenCalledTimes(3);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("there is no Close button", () => {
    useAuthStore.getState().setUser(photographer(true));
    render(<TermsReacceptanceGate />);
    expect(screen.queryByRole("button", { name: "Close" })).toBeNull();
  });

  it("the prompt tells the modal to ignore Escape", () => {
    useAuthStore.getState().setUser(photographer(true));
    render(<TermsReacceptanceGate />);
    expect(modalProps).toHaveBeenLastCalledWith(expect.objectContaining({ isOpen: true, escDisabled: true, hideClose: true }));
  });

  it("Escape or a click outside accepts nothing and signs nobody out", () => {
    useAuthStore.getState().setUser(photographer(true));
    render(<TermsReacceptanceGate />);
    fireEvent.keyDown(document, { key: "Escape" });
    fireEvent.click(screen.getByRole("dialog").previousElementSibling!);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(acceptTerms).not.toHaveBeenCalled();
    expect(logout).not.toHaveBeenCalled();
    expect(replace).not.toHaveBeenCalled();
  });

  it("focus moves into the page after Accept", async () => {
    useAuthStore.getState().setUser(photographer(true));
    acceptTerms.mockResolvedValue({ ...photographer(false), termsVersion: LEGAL_VERSION });
    render(
      <>
        <main>page</main>
        <TermsReacceptanceGate />
      </>,
    );
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "I accept" })));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.activeElement).toBe(screen.getByRole("main"));
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
