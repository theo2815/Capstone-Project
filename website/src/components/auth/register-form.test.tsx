import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RegisterForm } from "@/components/auth/register-form";
import { LEGAL_COPY } from "@/lib/legal/copy";

const http = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  api: http,
}));
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

function renderForm() {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <RegisterForm />
    </QueryClientProvider>,
  );
}

async function fillFields(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText("Full name"), "Juan dela Cruz");
  await user.type(screen.getByLabelText("Email"), "juan@example.com");
  await user.type(screen.getByLabelText(/^Password/), "password123");
}

describe("RegisterForm terms tick", () => {
  beforeEach(() => {
    http.post.mockReset();
  });
  afterEach(cleanup);

  it("blocks sign-up and shows an error when the box is not ticked", async () => {
    const user = userEvent.setup();
    renderForm();
    await fillFields(user);
    await user.click(screen.getByRole("button", { name: /Create account/ }));

    expect(await screen.findByText(LEGAL_COPY.signupTickError)).toBeTruthy();
    expect(http.post).not.toHaveBeenCalled();
  });

  it("sends termsAccepted: true once the box is ticked", async () => {
    http.post.mockRejectedValue(new Error("stop here"));
    const user = userEvent.setup();
    renderForm();
    await fillFields(user);
    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: /Create account/ }));

    await waitFor(() => expect(http.post).toHaveBeenCalledTimes(1));
    expect(http.post).toHaveBeenCalledWith(
      "/auth/register",
      expect.objectContaining({ email: "juan@example.com", termsAccepted: true }),
    );
  });

  it("opens the Terms and Privacy links in a new tab", () => {
    renderForm();
    for (const name of ["Terms of Service", "Privacy Policy"]) {
      const link = screen.getAllByRole("link", { name }).at(-1)!;
      expect(link.getAttribute("target")).toBe("_blank");
    }
  });
});
