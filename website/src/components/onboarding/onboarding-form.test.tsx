import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { OnboardingForm } from "@/components/onboarding/onboarding-form";
import { LEGAL_COPY } from "@/lib/legal/copy";
import { useAuthStore } from "@/store/auth-store";

const http = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  api: http,
}));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

function renderForm() {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <OnboardingForm />
    </QueryClientProvider>,
  );
}

describe("OnboardingForm terms tick", () => {
  beforeEach(() => {
    http.post.mockReset();
    useAuthStore.getState().setPendingOAuth({
      provider: "GOOGLE",
      sub: "sub-1",
      email: "new@example.com",
      name: "New Runner",
      idToken: "google-id-token",
    });
  });
  afterEach(() => {
    cleanup();
    useAuthStore.getState().setPendingOAuth(null);
  });

  it("blocks the role pick and shows an error when the box is not ticked", async () => {
    const user = userEvent.setup();
    renderForm();
    await user.click(screen.getByRole("radio", { name: /^Runner/ }));
    await user.click(screen.getByRole("button", { name: /Continue/ }));

    expect(await screen.findByText(LEGAL_COPY.signupTickError)).toBeTruthy();
    expect(http.post).not.toHaveBeenCalled();
  });

  it("re-posts the Google token with the role and termsAccepted: true", async () => {
    http.post.mockRejectedValue(new Error("stop here"));
    const user = userEvent.setup();
    renderForm();
    await user.click(screen.getByRole("radio", { name: /^Runner/ }));
    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: /Continue/ }));

    await waitFor(() => expect(http.post).toHaveBeenCalledTimes(1));
    expect(http.post).toHaveBeenCalledWith("/auth/google", {
      idToken: "google-id-token",
      role: "RUNNER",
      termsAccepted: true,
    });
  });
});
