import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { focusManager, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi, type MockInstance } from "vitest";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { SellerDetailsQueue } from "./seller-details-queue";

vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ showToast: vi.fn(), dismissToast: vi.fn() }) }));

const row = (userId: string, status: string, legalName: string) => ({
  userId, email: `${userId}@example.test`, legalName, idType: "philsys", status, submittedAt: "2026-11-07T01:00:00Z", reviewedAt: null,
});
const RECORD = {
  ...row("u1", "submitted", "Juan Dela Cruz"), address: "Cebu City", phone: "0917 123 4567", hasIdBack: true, hasBirForm: false,
  rejectionReason: null, reviewedBy: null, revision: 3,
  files: { idFront: "https://r2.example/front?sig=1", idBack: "https://r2.example/back?sig=1", selfie: "https://r2.example/selfie?sig=1", birForm: null },
  linksExpireAt: "2026-11-07T01:05:00Z",
};

let client: QueryClient;
let get: MockInstance<typeof api.get>;
const recordCalls = () => get.mock.calls.filter(([path]) => path === "/admin/seller-details/u1").length;

function renderQueue() {
  return render(<QueryClientProvider client={client}><SellerDetailsQueue /></QueryClientProvider>);
}
async function openRecord() {
  renderQueue();
  fireEvent.click(await screen.findByRole("button", { name: /Juan Dela Cruz/ }));
  return screen.findByRole("dialog");
}

beforeEach(() => {
  useAuthStore.getState().logout();
  useAuthStore.getState().setUser({ id: "admin-1", role: "ADMIN", name: "Admin", email: "admin@example.test", createdAt: "2026-09-10" });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  get = vi.spyOn(api, "get").mockImplementation(async (path: string) => {
    if (path.startsWith("/admin/seller-details?")) {
      const items = path.includes("status=submitted")
        ? [row("u1", "submitted", "Juan Dela Cruz")]
        : [row("u1", "submitted", "Juan Dela Cruz"), row("u2", "approved", "Maria Santos")];
      return { items, total: items.length, offset: 0, limit: 10 };
    }
    if (path === "/admin/seller-details/u1") return RECORD;
    if (path === "/admin/users/u1/settings") return { payouts: [{ id: "p1", method: "gcash", accountName: "Juan D. Cruz", accountNumber: "0917", isPrimary: true }] };
    throw new Error(`unexpected GET ${path}`);
  });
});
afterEach(() => { client.clear(); vi.restoreAllMocks(); useAuthStore.getState().logout(); });

it("lists waiting records first and all records on request", async () => {
  renderQueue();
  expect(await screen.findByRole("button", { name: /Juan Dela Cruz/ })).toBeInTheDocument();
  expect(get.mock.calls[0][0]).toContain("status=submitted");
  expect(screen.queryByText("Maria Santos")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "All" }));
  expect(await screen.findByText("Maria Santos")).toBeInTheDocument();
  expect(get.mock.calls.at(-1)![0]).not.toContain("status=");
});

it("opens a record once and shows its photos, the payout names and the checklist", async () => {
  const dialog = await openRecord();
  expect(await within(dialog).findByRole("img", { name: "Front of the ID" })).toHaveAttribute("src", RECORD.files.idFront);
  expect(within(dialog).getByRole("img", { name: "Back of the ID" })).toHaveAttribute("src", RECORD.files.idBack);
  expect(within(dialog).getByRole("img", { name: "Photo holding the ID" })).toHaveAttribute("src", RECORD.files.selfie);
  expect(await within(dialog).findByText("Juan D. Cruz")).toBeInTheDocument();
  expect(within(dialog).getByText(/verify\.philsys\.gov\.ph/)).toBeInTheDocument();
  expect(within(dialog).getByText(/verification\.prc\.gov\.ph/)).toBeInTheDocument();
  expect(recordCalls()).toBe(1);
});

it("approves with the record's revision", async () => {
  const post = vi.spyOn(api, "post").mockResolvedValue({ ...RECORD, status: "approved", files: null });
  const dialog = await openRecord();
  fireEvent.click(await within(dialog).findByRole("button", { name: "Approve" }));
  await waitFor(() => expect(post).toHaveBeenCalledWith("/admin/seller-details/u1/approve", { revision: 3 }));
});

it("needs a reason to reject and sends it with the revision", async () => {
  const post = vi.spyOn(api, "post").mockResolvedValue({ ...RECORD, status: "rejected", files: null });
  const dialog = await openRecord();
  fireEvent.click(await within(dialog).findByRole("button", { name: "Reject" }));
  expect(await within(dialog).findByText("Write the reason the photographer will see.")).toBeInTheDocument();
  expect(post).not.toHaveBeenCalled();
  fireEvent.change(within(dialog).getByLabelText("Reason for rejecting"), { target: { value: "The ID is expired." } });
  fireEvent.click(within(dialog).getByRole("button", { name: "Reject" }));
  await waitFor(() => expect(post).toHaveBeenCalledWith("/admin/seller-details/u1/reject", { revision: 3, reason: "The ID is expired." }));
});

it("asks for a reload when the record changed", async () => {
  vi.spyOn(api, "post").mockRejectedValue(new ApiError([{ code: "INVALID_STATE_TRANSITION", message: "Seller details have changed. Reload before deciding." }], 409));
  const dialog = await openRecord();
  fireEvent.click(await within(dialog).findByRole("button", { name: "Approve" }));
  expect(await within(dialog).findByText("These details changed. Reload before deciding.")).toBeInTheDocument();
});

it("does not open the record again when the window regains focus", async () => {
  const dialog = await openRecord();
  await within(dialog).findByRole("img", { name: "Front of the ID" });
  const later = Date.now() + 5 * 60_000; // past the 4-minute staleTime
  vi.spyOn(Date, "now").mockReturnValue(later);
  act(() => { focusManager.setFocused(false); focusManager.setFocused(true); });
  await new Promise((resolve) => setTimeout(resolve, 20));
  expect(recordCalls()).toBe(1);
  focusManager.setFocused(undefined);
});
