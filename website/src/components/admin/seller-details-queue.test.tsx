import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { focusManager, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi, type MockInstance } from "vitest";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { SellerDetailsQueue } from "./seller-details-queue";

vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ showToast: vi.fn(), dismissToast: vi.fn() }) }));

// `status` keeps the old wire values (manual_review → submitted, auto_approved → approved); the queue reads reviewState.
const WIRE_STATUS = { manual_review: "submitted", auto_approved: "approved", approved: "approved", rejected: "rejected" } as const;
const row = (userId: string, reviewState: keyof typeof WIRE_STATUS, legalName: string) => ({
  userId, email: `${userId}@example.test`, legalName, idType: "philsys", status: WIRE_STATUS[reviewState], reviewState,
  submittedAt: "2026-11-07T01:00:00Z", reviewedAt: null,
  reasonCodes: [] as string[], livenessConfidence: null as number | null, faceSimilarity: null as number | null, autoCheckedAt: null,
});
const RECORD = {
  ...row("u1", "manual_review", "Juan Dela Cruz"), address: "Cebu City", phone: "0917 123 4567", hasIdBack: true, hasBirForm: false,
  rejectionReason: null, reviewedBy: null, revision: 3,
  reasonCodes: ["LIVENESS_LOW", "SOMETHING_NEW"], livenessConfidence: 81.25, faceSimilarity: 99.6, autoCheckedAt: "2026-11-07T01:00:05Z",
  biometricConsentAt: "2026-11-07T00:58:00Z", hasIdNumber: true,
  repeatMatches: [
    { userId: "u9", email: "ana@example.test", matchedOn: ["ID_NUMBER", "PHONE"], suspended: true },
    { userId: "u8", email: "nohandle@example.test", matchedOn: ["PAYOUT_NUMBER"], suspended: false },
  ],
  files: {
    idFront: "https://r2.example/front?sig=1", idBack: "https://r2.example/back?sig=1", selfie: "https://r2.example/selfie?sig=1",
    birForm: null, livenessReference: "https://r2.example/live?sig=1",
  },
  linksExpireAt: "2026-11-07T01:05:00Z",
};
const AUTO_RECORD = { ...RECORD, ...row("u2", "auto_approved", "Maria Santos"), reasonCodes: ["PASSED"], repeatMatches: [], revision: 1 };
const USERS = [
  { userId: "u9", email: "ana@example.test", handle: "ana-shots", role: "PHOTOGRAPHER" },
  { userId: "u8", email: "nohandle@example.test", handle: null, role: "PHOTOGRAPHER" },
];

let client: QueryClient;
let get: MockInstance<typeof api.get>;
const recordCalls = () => get.mock.calls.filter(([path]) => path === "/admin/seller-details/u1").length;
const listCalls = () => get.mock.calls.map(([path]) => path).filter((path) => path.startsWith("/admin/seller-details?"));

function renderQueue() {
  return render(<QueryClientProvider client={client}><SellerDetailsQueue /></QueryClientProvider>);
}
async function openRecord(name: RegExp = /Juan Dela Cruz/) {
  renderQueue();
  fireEvent.click(await screen.findByRole("button", { name }));
  return screen.findByRole("dialog");
}

beforeEach(() => {
  useAuthStore.getState().logout();
  useAuthStore.getState().setUser({ id: "admin-1", role: "ADMIN", name: "Admin", email: "admin@example.test", createdAt: "2026-09-10" });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  get = vi.spyOn(api, "get").mockImplementation(async (path: string) => {
    if (path.startsWith("/admin/seller-details?")) {
      const items = path.includes("status=manual_review") ? [row("u1", "manual_review", "Juan Dela Cruz")]
        : path.includes("status=auto_approved") ? [row("u2", "auto_approved", "Maria Santos")]
        : [row("u1", "manual_review", "Juan Dela Cruz"), row("u2", "auto_approved", "Maria Santos")];
      return { items, total: items.length, offset: 0, limit: 10 };
    }
    if (path === "/admin/seller-details/u1") return RECORD;
    if (path === "/admin/seller-details/u2") return AUTO_RECORD;
    if (path.startsWith("/admin/users?")) return { items: USERS, total: USERS.length, offset: 0, limit: 100 };
    if (path === "/admin/users/u1/settings" || path === "/admin/users/u2/settings") return { payouts: [{ id: "p1", method: "gcash", accountName: "Juan D. Cruz", accountNumber: "0917", isPrimary: true }] };
    throw new Error(`unexpected GET ${path}`);
  });
});
afterEach(() => { client.clear(); vi.restoreAllMocks(); useAuthStore.getState().logout(); });

it("default list asks for manual_review, and labels and actions read reviewState, not status", async () => {
  renderQueue();
  const juan = await screen.findByRole("button", { name: /Juan Dela Cruz/ });
  expect(listCalls()[0]).toContain("status=manual_review");
  expect(within(juan).getByText("Needs review")).toBeInTheDocument();
  expect(screen.queryByText("Maria Santos")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "All" }));
  const maria = await screen.findByRole("button", { name: /Maria Santos/ });
  expect(listCalls().at(-1)).not.toContain("status=");
  // Its wire status is "approved", but the label follows reviewState.
  expect(within(maria).getByText("Auto-approved")).toBeInTheDocument();
  const dialog = await openRecordFrom(/Juan Dela Cruz/);
  expect(await within(dialog).findByRole("button", { name: "Approve" })).toBeInTheDocument();
});

async function openRecordFrom(name: RegExp) {
  fireEvent.click(screen.getByRole("button", { name }));
  return screen.findByRole("dialog");
}

it("auto-approved filter lists auto rows and allows reject", async () => {
  renderQueue();
  await screen.findByRole("button", { name: /Juan Dela Cruz/ });
  fireEvent.click(screen.getByRole("button", { name: "Auto-approved" }));
  await screen.findByRole("button", { name: /Maria Santos/ });
  expect(listCalls().at(-1)).toContain("status=auto_approved");
  const dialog = await openRecordFrom(/Maria Santos/);
  expect(await within(dialog).findByRole("button", { name: "Reject" })).toBeInTheDocument();
  expect(within(dialog).queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();
});

it("shows both scores and the reasons; an unknown reason code renders raw", async () => {
  const dialog = await openRecord();
  expect(await within(dialog).findByText("81.3%")).toBeInTheDocument();
  expect(within(dialog).getByText("99.6%")).toBeInTheDocument();
  expect(within(dialog).getByText("Live check score below the bar; SOMETHING_NEW")).toBeInTheDocument();
  expect(within(dialog).getByText("ID number on file")).toBeInTheDocument();
});

it("reference image is the first photo", async () => {
  const dialog = await openRecord();
  await within(dialog).findByRole("img", { name: "Live selfie (reference)" });
  const photos = within(within(dialog).getByRole("region", { name: "Photos" })).getAllByRole("img");
  expect(photos.map((img) => img.getAttribute("alt"))).toEqual(["Live selfie (reference)", "Front of the ID", "Back of the ID", "Photo holding the ID"]);
  expect(photos[0]).toHaveAttribute("src", RECORD.files.livenessReference);
});

it("repeat match links to the user, names what matched and says suspended", async () => {
  const dialog = await openRecord();
  const matches = within(await within(dialog).findByRole("region", { name: "Other accounts" })).getAllByRole("listitem");
  expect(await within(matches[0]).findByRole("link", { name: "ana@example.test" })).toHaveAttribute("href", "/admin/photographers/ana-shots");
  expect(matches[0]).toHaveTextContent(/ID number, phone/);
  expect(matches[0]).toHaveTextContent(/suspended/i);
  // No handle to link to: the email is plain text.
  expect(within(matches[1]).queryByRole("link")).not.toBeInTheDocument();
  expect(matches[1]).toHaveTextContent(/nohandle@example\.test.*payout number/);
  expect(matches[1]).not.toHaveTextContent(/suspended/i);
});

it("opens a record once and shows its photos, the payout names and the checklist", async () => {
  const dialog = await openRecord();
  expect(await within(dialog).findByRole("img", { name: "Front of the ID" })).toHaveAttribute("src", RECORD.files.idFront);
  expect(within(dialog).getByRole("img", { name: "Back of the ID" })).toHaveAttribute("src", RECORD.files.idBack);
  expect(within(dialog).getByRole("img", { name: "Photo holding the ID" })).toHaveAttribute("src", RECORD.files.selfie);
  expect(await within(dialog).findByText("Juan D. Cruz")).toBeInTheDocument();
  expect(within(dialog).getByText(/live selfie and the photo holding the ID show the person on the ID photo/)).toBeInTheDocument();
  expect(within(dialog).queryByText(/typed legal name, the name on the ID and the payout account names match/)).not.toBeInTheDocument();
  expect(within(dialog).getByText(/verify\.philsys\.gov\.ph/)).toBeInTheDocument();
  expect(within(dialog).getByText(/verification\.prc\.gov\.ph/)).toBeInTheDocument();
  expect(recordCalls()).toBe(1);
});

it("approves with the record's revision", async () => {
  const post = vi.spyOn(api, "post").mockResolvedValue({ ...RECORD, status: "approved", reviewState: "approved", files: null });
  const dialog = await openRecord();
  fireEvent.click(await within(dialog).findByRole("button", { name: "Approve" }));
  await waitFor(() => expect(post).toHaveBeenCalledWith("/admin/seller-details/u1/approve", { revision: 3 }));
});

it("needs a reason to reject and sends it with the revision", async () => {
  const post = vi.spyOn(api, "post").mockResolvedValue({ ...RECORD, status: "rejected", reviewState: "rejected", files: null });
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
