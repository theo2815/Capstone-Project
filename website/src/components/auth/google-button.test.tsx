import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { GoogleButton } from "./google-button";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
}));
vi.mock("@/hooks/use-auth", () => ({
  useAuth: () => ({ googleLogin: vi.fn() }),
}));
vi.mock("@/lib/constants", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/constants")>()),
  GOOGLE_CLIENT_ID: "test-client-id",
}));
vi.mock("next/script", () => ({
  default: ({ src }: { src: string }) => (
    <span data-testid="gsi-script" data-src={src} />
  ),
}));

afterEach(() => vi.restoreAllMocks());

function useUserAgent(ua: string) {
  vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue(ua);
}

describe("GoogleButton", () => {
  it("shows guidance without loading GIS in an in-app browser", async () => {
    useUserAgent(
      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
    );
    render(<GoogleButton />);

    expect(await screen.findByText("In-app browser")).toBeInTheDocument();
    expect(screen.queryByTestId("gsi-script")).not.toBeInTheDocument();
  });

  it("loads GIS in a normal browser", async () => {
    useUserAgent("Mozilla/5.0 Chrome/120.0.0.0 Safari/537.36");
    render(<GoogleButton />);

    await waitFor(() => expect(screen.getByTestId("gsi-script")).toBeInTheDocument());
    expect(screen.queryByText("In-app browser")).not.toBeInTheDocument();
  });
});
