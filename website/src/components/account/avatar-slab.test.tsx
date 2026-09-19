import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { AvatarSlab } from "./avatar-slab";
import { useAuthStore } from "@/store/auth-store";
import { useUserMediaStore } from "@/store/user-media-store";
import { uploadAvatar, deleteAvatar } from "@/lib/api-avatar";
import { squareCropToDataUrl } from "@/lib/image-utils";
const mocks = vi.hoisted(() => ({ toast: vi.fn() }));
vi.mock("@/hooks/use-auth", () => ({ useAuth: () => ({ user: useAuthStore((s) => s.user) }) }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ showToast: mocks.toast }) }));
vi.mock("@/lib/api-avatar", () => ({ uploadAvatar: vi.fn(), deleteAvatar: vi.fn() }));
vi.mock("@/lib/image-utils", () => ({ ACCEPTED_IMAGE_MIME: ["image/png"], validateImageFile: () => null, squareCropToDataUrl: vi.fn() }));
const user = { id: "runner", role: "RUNNER" as const, name: "Runner", email: "runner@example.test", createdAt: "2026-09-10", avatarUrl: "/old.jpg" };
const file = new File(["fixture"], "avatar.png", { type: "image/png" });
beforeEach(() => { vi.resetAllMocks(); useAuthStore.getState().setUser(user); useUserMediaStore.getState().setAvatar(null); vi.mocked(squareCropToDataUrl).mockResolvedValue("data:image/png;base64,AA=="); });
afterEach(() => { cleanup(); useAuthStore.getState().logout(); });

it.each(["upload", "remove"])("ignores stale %s success and resets draft/pending state on relogin", async (action) => {
  let finish!: (value: typeof user) => void;
  vi.mocked(action === "upload" ? uploadAvatar : deleteAvatar).mockImplementationOnce(() => new Promise((r) => { finish = r; }));
  render(<AvatarSlab />);
  if (action === "upload") {
    fireEvent.change(screen.getByLabelText(/Replace picture/), { target: { files: [file] } });
    await waitFor(() => expect(screen.getByRole("button", { name: /Use this picture/ })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: /Use this picture/ }));
    expect(screen.getByRole("button", { name: /Saving/ })).toBeDisabled();
  } else {
    fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    expect(screen.queryByRole("button", { name: "Remove" })).not.toBeInTheDocument();
  }
  act(() => { useAuthStore.getState().logout(); useAuthStore.getState().setUser({ ...user, avatarUrl: "/new.jpg" }); });
  await act(async () => { finish({ ...user, avatarUrl: "/stale.jpg" }); });
  expect(useAuthStore.getState().user?.avatarUrl).toBe("/new.jpg");
  expect(screen.getByLabelText(/Replace picture/)).toBeEnabled();
  expect(mocks.toast).not.toHaveBeenCalled();
});

it("keeps failed removal retryable and commits a confirmed removal", async () => {
  vi.mocked(deleteAvatar).mockRejectedValueOnce(new Error("Offline")).mockResolvedValueOnce({ ...user, avatarUrl: undefined });
  render(<AvatarSlab />); fireEvent.click(screen.getByRole("button", { name: "Remove" }));
  await screen.findByText("Could not remove the picture. Try again.");
  expect(useAuthStore.getState().user?.avatarUrl).toBe("/old.jpg");
  fireEvent.click(screen.getByRole("button", { name: "Remove" }));
  await waitFor(() => expect(useAuthStore.getState().user?.avatarUrl).toBeUndefined());
  expect(mocks.toast).toHaveBeenCalledExactlyOnceWith({ kind: "success", message: "Profile picture removed." });
});

it("does not show an old upload failure after logout", async () => {
  let fail!: (error: Error) => void;
  vi.mocked(uploadAvatar).mockImplementationOnce(() => new Promise((_, reject) => { fail = reject; }));
  render(<AvatarSlab />); fireEvent.change(screen.getByLabelText(/Replace picture/), { target: { files: [file] } });
  await waitFor(() => expect(screen.getByRole("button", { name: /Use this picture/ })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: /Use this picture/ }));
  act(() => useAuthStore.getState().logout());
  await act(async () => { fail(new Error("Offline")); });
  expect(useAuthStore.getState().isAuthenticated).toBe(false); expect(mocks.toast).not.toHaveBeenCalled();
  expect(screen.queryByText(/Could not/)).not.toBeInTheDocument();
});
