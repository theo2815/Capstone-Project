import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ReportButton } from "@/components/moderation/report-dialog";
import { ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";

// Mock the HTTP client so the real report paths in lib/api-reports.ts are asserted.
const http = vi.hoisted(() => ({ post: vi.fn() }));
vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  api: http,
}));

const EVENT_ID = "11111111-2222-4333-8444-555555555555";
const NOTE = "This race never happened, the photos are copied.";

function signIn() {
  useAuthStore.setState({ isAuthenticated: true, user: { id: "u1" } as never });
}

async function openDialog() {
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: /report/i }));
  return user;
}

beforeEach(() => {
  http.post.mockReset();
  useAuthStore.setState({ isAuthenticated: false, user: null });
  usePhotographerSettingsStore.setState({ handle: "" });
});
afterEach(cleanup);

describe("ReportButton", () => {
  it("hidden when signed out", () => {
    render(<ReportButton target={{ kind: "event", id: EVENT_ID }} />);
    expect(screen.queryByRole("button", { name: /report/i })).toBeNull();
  });

  it("note under 10 characters cannot be sent", async () => {
    signIn();
    render(<ReportButton target={{ kind: "event", id: EVENT_ID }} />);
    const user = await openDialog();
    await user.type(screen.getByLabelText(/what is wrong/i), "too short");
    const send = screen.getByRole("button", { name: /send report/i });
    expect(send).toBeDisabled();
    await user.click(send);
    expect(http.post).not.toHaveBeenCalled();
  });

  it("failure keeps the note", async () => {
    signIn();
    http.post.mockRejectedValue(
      new ApiError([{ code: "RATE_LIMITED", message: "Too many reports. Try again later." }], 429),
    );
    render(<ReportButton target={{ kind: "event", id: EVENT_ID }} />);
    const user = await openDialog();
    await user.type(screen.getByLabelText(/what is wrong/i), NOTE);
    await user.click(screen.getByRole("button", { name: /send report/i }));
    expect(await screen.findByText("Too many reports. Try again later.")).toBeInTheDocument();
    expect(screen.getByLabelText(/what is wrong/i)).toHaveValue(NOTE);
  });

  it("no button on your own profile", () => {
    signIn();
    usePhotographerSettingsStore.setState({ handle: "ana" });
    const { rerender } = render(<ReportButton target={{ kind: "photographer", handle: "ana" }} />);
    expect(screen.queryByRole("button", { name: /report/i })).toBeNull();
    rerender(<ReportButton target={{ kind: "photographer", handle: "ben" }} />);
    expect(screen.getByRole("button", { name: /report/i })).toBeInTheDocument();
  });

  it("sends the note and thanks the user, also when already reported", async () => {
    signIn();
    http.post.mockResolvedValue({ reported: true, alreadyReported: true });
    render(<ReportButton target={{ kind: "photographer", handle: "ben" }} />);
    const user = await openDialog();
    await user.type(screen.getByLabelText(/what is wrong/i), `  ${NOTE}  `);
    await user.click(screen.getByRole("button", { name: /send report/i }));
    await waitFor(() =>
      expect(http.post).toHaveBeenCalledWith("/me/reports/photographers/ben", { note: NOTE }),
    );
    expect(await screen.findByText(/thanks, an admin will look at it/i)).toBeInTheDocument();
  });
});
