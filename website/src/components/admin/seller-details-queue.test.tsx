import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { focusManager, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi, type MockInstance } from "vitest";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { SellerDetailsQueue } from "./seller-details-queue";

vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ showToast: vi.fn(), dismissToast: vi.fn() }) }));
// A tiny in-memory URL: the filter lives in ?sd=.
const nav = vi.hoisted(() => {
  let search = "";
  const listeners = new Set<() => void>();
  return {
    get: () => search,
    set: (next: string) => { search = next; listeners.forEach((l) => l()); },
    subscribe: (l: () => void) => { listeners.add(l); return () => listeners.delete(l); },
  };
});
vi.mock("next/navigation", async () => {
  const { useSyncExternalStore } = await import("react");
  return {
    usePathname: () => "/admin/verifications",
    useSearchParams: () => new URLSearchParams(useSyncExternalStore(nav.subscribe, nav.get)),
    useRouter: () => ({ replace: (url: string) => nav.set(url.split("?")[1] ?? ""), push: vi.fn() }),
  };
});

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

let kpis: object = { pendingSellerDetails: 4 };
beforeEach(() => {
  nav.set("");
  kpis = { pendingSellerDetails: 4 };
  Element.prototype.scrollIntoView = vi.fn();
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
    if (path === "/admin/kpis") return kpis;
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

it("FT-41: opening a row writes ?sd-row=, and a link with ?sd-row= opens it", async () => {
  await openRecord();
  expect(new URLSearchParams(nav.get()).get("sd-row")).toBe("u1");
  cleanup();
  nav.set("sd-row=u1");
  renderQueue();
  expect(await screen.findByRole("dialog")).toHaveTextContent("Juan Dela Cruz");
});

it("FT-41: the slab number is 04 unless the page gives another", async () => {
  const view = renderQueue();
  expect(within(document.getElementById("seller-details")!).getByText("04")).toBeInTheDocument();
  view.unmount();
  render(<QueryClientProvider client={client}><SellerDetailsQueue number="03" /></QueryClientProvider>);
  expect(within(document.getElementById("seller-details")!).getByText("03")).toBeInTheDocument();
});

it("the filter reads and writes ?sd=, and only Needs review carries the KPI count", async () => {
  nav.set("sd=auto_approved");
  renderQueue();
  const tabs = screen.getByRole("group", { name: "Filter seller details" });
  await screen.findByRole("button", { name: /Maria Santos/ });
  expect(listCalls()[0]).toContain("status=auto_approved");
  expect(within(tabs).getByRole("button", { name: "Auto-approved" })).toHaveAttribute("aria-pressed", "true");
  expect(await within(tabs).findByRole("button", { name: "Needs review 4" })).toBeInTheDocument();
  fireEvent.click(within(tabs).getByRole("button", { name: "All" }));
  expect(nav.get()).toBe("sd=all");
  await waitFor(() => expect(listCalls().at(-1)).not.toContain("status="));
  fireEvent.click(within(tabs).getByRole("button", { name: "Needs review 4" }));
  expect(nav.get()).toBe("");
});

it("an old backend without the KPI field shows no count, not 0", async () => {
  kpis = {};
  renderQueue();
  await screen.findByRole("button", { name: /Juan Dela Cruz/ });
  await waitFor(() => expect(get.mock.calls.some(([path]) => path === "/admin/kpis")).toBe(true));
  expect(screen.getByRole("button", { name: "Needs review" })).toBeInTheDocument();
});

it("shows a busy skeleton while the list loads", () => {
  get.mockImplementation(() => new Promise(() => {}));
  const { container } = renderQueue();
  expect(container.querySelector('[aria-busy="true"]')).not.toBeNull();
  expect(screen.queryByText("Loading…")).not.toBeInTheDocument();
});

it("Reject with no reason brings the reason box into view and focuses it", async () => {
  const dialog = await openRecord();
  fireEvent.click(await within(dialog).findByRole("button", { name: "Reject" }));
  const box = within(dialog).getByLabelText(/Reason for rejecting/);
  expect(box).toHaveFocus();
  expect(box.scrollIntoView).toHaveBeenCalled();
  expect(box).toHaveAttribute("aria-invalid", "true");
  expect(box).toBeRequired();
  expect(within(dialog).getByText("0/500")).toBeInTheDocument();
});

it("Open full size shows the ID photo in its own dialog, and Esc closes only that", async () => {
  const dialog = await openRecord();
  fireEvent.click(await within(dialog).findByRole("button", { name: /^Open full size\W+Front of the ID$/ }));
  const zoom = await screen.findByRole("dialog", { name: "Front of the ID" });
  expect(within(zoom).getByRole("img", { name: "Front of the ID" })).toHaveAttribute("src", RECORD.files.idFront);
  fireEvent.keyDown(document, { key: "Escape" });
  await waitFor(() => expect(screen.queryByRole("dialog", { name: "Front of the ID" })).not.toBeInTheDocument());
  expect(dialog).toBeInTheDocument();
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
  expect(within(dialog).getByText("Live check score below the pass mark; SOMETHING_NEW")).toBeInTheDocument();
  expect(within(dialog).getByText("ID number on file")).toBeInTheDocument();
});

it("the three new reasons show plain labels", async () => {
  const base = get.getMockImplementation()!;
  get.mockImplementation(async (path: string) => path === "/admin/seller-details/u1"
    ? { ...RECORD, reasonCodes: ["ID_FORMAT_UNCHECKED", "ID_NAME_MISMATCH", "REPEATED_FAILED_CHECKS"] }
    : base(path));
  const dialog = await openRecord();
  expect(await within(dialog).findByText(
    "ID number format not checked for this ID type; Legal name not found on the ID; Several failed automatic checks before this one",
  )).toBeInTheDocument();
});

it("a try that failed again with the repeat code never says Passed", async () => {
  const base = get.getMockImplementation()!;
  get.mockImplementation(async (path: string) => path === "/admin/seller-details/u1"
    ? { ...RECORD, reasonCodes: ["FACE_MISMATCH", "REPEATED_FAILED_CHECKS"] }
    : base(path));
  const dialog = await openRecord();
  expect(await within(dialog).findByText(
    "Live face does not match the ID photo; Several failed automatic checks before this one",
  )).toBeInTheDocument();
  expect(within(dialog).queryByText(/Passed/)).not.toBeInTheDocument();
});

it("the failed-try count shows when above zero and is absent at zero", async () => {
  const base = get.getMockImplementation()!;
  const serve = (extra: object) => get.mockImplementation(async (path: string) => path === "/admin/seller-details/u1" ? { ...RECORD, ...extra } : base(path));
  serve({ failedAutoChecks: 2 });
  const first = await openRecord();
  expect(await within(first).findByText("Failed automatic tries")).toBeInTheDocument();
  expect(within(first).getByText("Failed automatic tries").nextElementSibling).toHaveTextContent(/^2$/);
  // The open row is in the URL (FT-41): start each round from a URL with none.
  cleanup(); nav.set("");
  for (const extra of [{ failedAutoChecks: 0 }, {}]) {
    serve(extra);
    const dialog = await openRecord();
    await within(dialog).findByText("ID number on file");
    expect(within(dialog).queryByText("Failed automatic tries")).not.toBeInTheDocument();
    cleanup(); nav.set("");
  }
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
  // FT-14 (C-14): the two checklist addresses are links that open in a new tab.
  for (const [name, href] of [["verify.philsys.gov.ph", "https://verify.philsys.gov.ph"], ["verification.prc.gov.ph", "https://verification.prc.gov.ph"]]) {
    const link = within(dialog).getByRole("link", { name });
    expect(link).toHaveAttribute("href", href);
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  }
  expect(recordCalls()).toBe(1);
});

// FT-14 (C-12, C-13, C-10): what Approve and Reject do, the approved-record confirm, the pass marks, the row reason.
it("the drawer says what Approve and Reject do before the click", async () => {
  const dialog = await openRecord();
  expect(await within(dialog).findByText("Approve: the ID check is passed. If their profile is complete it goes live and they are told. Reject: they are told your reason and their new uploads pause until they send details again. A rejection cannot be undone here.")).toBeInTheDocument();
});

it("rejecting an approved record asks first and sends nothing until confirmed", async () => {
  const post = vi.spyOn(api, "post").mockResolvedValue({ ...AUTO_RECORD, status: "rejected", reviewState: "rejected", files: null });
  nav.set("sd=auto_approved");
  const dialog = await openRecord(/Maria Santos/);
  fireEvent.change(await within(dialog).findByLabelText(/Reason for rejecting/), { target: { value: "The ID photo is edited." } });
  fireEvent.click(within(dialog).getByRole("button", { name: "Reject" }));
  const confirm = await screen.findByRole("dialog", { name: "Reject an approved ID check?" });
  expect(within(confirm).getByText("This photographer is already approved. Rejecting pauses their new uploads until they send details again, and they are told your reason. This cannot be undone here.")).toBeInTheDocument();
  expect(post).not.toHaveBeenCalled();
  fireEvent.click(within(confirm).getByRole("button", { name: "Reject" }));
  await waitFor(() => expect(post).toHaveBeenCalledWith("/admin/seller-details/u2/reject", { revision: 1, reason: "The ID photo is edited." }));
});

it("rejecting a record that needs review stays one click once a reason is typed", async () => {
  const post = vi.spyOn(api, "post").mockResolvedValue({ ...RECORD, status: "rejected", reviewState: "rejected", files: null });
  const dialog = await openRecord();
  fireEvent.change(await within(dialog).findByLabelText(/Reason for rejecting/), { target: { value: "The ID is expired." } });
  fireEvent.click(within(dialog).getByRole("button", { name: "Reject" }));
  await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
  expect(screen.queryByRole("dialog", { name: "Reject an approved ID check?" })).not.toBeInTheDocument();
});

it("scores show their pass mark when the backend sends it, and the bare score when it does not", async () => {
  const base = get.getMockImplementation()!;
  const serve = (extra: object) => get.mockImplementation(async (path: string) => path === "/admin/seller-details/u1" ? { ...RECORD, livenessConfidence: 72.4, faceSimilarity: 99.6, ...extra } : base(path));
  serve({ livenessPassMark: 95, faceMatchPassMark: 99 });
  const first = await openRecord();
  expect(await within(first).findByText("72.4% (needs 95%)")).toBeInTheDocument();
  expect(within(first).getByText("99.6% (needs 99%)")).toBeInTheDocument();
  cleanup(); nav.set("");
  serve({});
  const dialog = await openRecord();
  expect(await within(dialog).findByText("72.4%")).toBeInTheDocument();
  expect(within(dialog).queryByText(/needs/)).not.toBeInTheDocument();
  expect(within(dialog).queryByText(/undefined/)).not.toBeInTheDocument();
});

it("a needs-review row shows its first real reason; an auto-approved row shows none", async () => {
  const base = get.getMockImplementation()!;
  get.mockImplementation(async (path: string) => {
    if (!path.startsWith("/admin/seller-details?")) return base(path);
    const items = [
      { ...row("u1", "manual_review", "Juan Dela Cruz"), reasonCodes: ["PASSED", "AI_DISABLED", "FACE_MISMATCH", "LIVENESS_LOW"] },
      { ...row("u2", "auto_approved", "Maria Santos"), reasonCodes: ["FACE_MISMATCH"] },
    ];
    return { items, total: 2, offset: 0, limit: 10 };
  });
  renderQueue();
  const juan = await screen.findByRole("button", { name: /Juan Dela Cruz/ });
  expect(within(juan).getByText("Live face does not match the ID photo")).toBeInTheDocument();
  expect(within(juan).queryByText(/Live check score/)).not.toBeInTheDocument();
  expect(within(screen.getByRole("button", { name: /Maria Santos/ })).queryByText(/Live face/)).not.toBeInTheDocument();
});

it("approves with the record's revision", async () => {
  const post = vi.spyOn(api, "post").mockResolvedValue({ ...RECORD, status: "approved", reviewState: "approved", files: null });
  const dialog = await openRecord();
  fireEvent.click(await within(dialog).findByRole("button", { name: "Approve" }));
  await waitFor(() => expect(post).toHaveBeenCalledWith("/admin/seller-details/u1/approve", { revision: 3 }));
});

it("a decision refetches the KPIs, so the Needs review count drops at once", async () => {
  vi.spyOn(api, "post").mockResolvedValue({ ...RECORD, status: "approved", reviewState: "approved", files: null });
  const dialog = await openRecord();
  await waitFor(() => expect(get.mock.calls.filter(([path]) => path === "/admin/kpis")).toHaveLength(1));
  fireEvent.click(await within(dialog).findByRole("button", { name: "Approve" }));
  await waitFor(() => expect(get.mock.calls.filter(([path]) => path === "/admin/kpis")).toHaveLength(2));
});

it("needs a reason to reject and sends it with the revision", async () => {
  const post = vi.spyOn(api, "post").mockResolvedValue({ ...RECORD, status: "rejected", reviewState: "rejected", files: null });
  const dialog = await openRecord();
  fireEvent.click(await within(dialog).findByRole("button", { name: "Reject" }));
  expect(await within(dialog).findByText("Write the reason the photographer will see.")).toBeInTheDocument();
  expect(post).not.toHaveBeenCalled();
  fireEvent.change(within(dialog).getByLabelText(/Reason for rejecting/), { target: { value: "The ID is expired." } });
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

// AX-39 (UA7): row values read as typed; mono caps stay for labels only.
it("a row's ID type and date are not upper-cased", async () => {
  renderQueue();
  const juan = await screen.findByRole("button", { name: /Juan Dela Cruz/ });
  expect(within(juan).getByText(/^Nov 0\d · 2026$/).closest(".uppercase")).toBeNull();
});

// AX-57: drawer and detail values read as typed, wrap at 375, and link a no-handle photographer by id.
const notUpperCased = (els: HTMLElement[]) => { expect(els.length).toBeGreaterThan(0); for (const el of els) expect(el.closest(".uppercase")).toBeNull(); };
it("the drawer's email wraps in its own case, and its dates are not upper-cased", async () => {
  const dialog = await openRecord();
  const email = await within(dialog).findByText("u1@example.test");
  expect(email.closest(".uppercase")).toBeNull();
  expect(email).toHaveClass("break-all");
  notUpperCased(within(dialog).getAllByText(/^Nov 0\d · 2026$/));
});

it("an empty Needs review list says the admin is caught up", async () => {
  get.mockImplementation(async (path: string) => {
    if (path.startsWith("/admin/seller-details?")) return { items: [], total: 0, offset: 0, limit: 10 };
    if (path === "/admin/kpis") return { pendingSellerDetails: 0 };
    if (path.startsWith("/admin/users?")) return { items: [], total: 0, offset: 0, limit: 100 };
    throw new Error(`unexpected GET ${path}`);
  });
  renderQueue();
  expect(await screen.findByText("No seller details waiting. You're caught up.")).toBeInTheDocument();
});

it("a failed list load says Couldn't load seller details. with Retry", async () => {
  get.mockRejectedValue(new Error("down"));
  renderQueue();
  const alert = await screen.findByRole("alert");
  expect(within(alert).getByText("Couldn't load seller details.")).toBeInTheDocument();
  expect(within(alert).getByRole("button", { name: "Retry" })).toBeInTheDocument();
});

it("the record loads behind a skeleton, and a failed record says Couldn't load seller details. with Retry", async () => {
  const served = get.getMockImplementation()!;
  let record: () => Promise<never> = () => new Promise(() => {});
  get.mockImplementation(((path: string, ...rest: never[]) =>
    path === "/admin/seller-details/u1" ? record() : served(path, ...rest)) as typeof api.get);
  const dialog = await openRecord();
  expect(dialog.querySelector('[aria-busy="true"] .animate-pulse')).not.toBeNull();
  expect(within(dialog).queryByText(/Loading/)).toBeNull();
  cleanup();
  client.clear();
  record = () => Promise.reject(new Error("down"));
  const failed = await openRecord();
  const alert = await within(failed).findByRole("alert");
  expect(within(alert).getByText("Couldn't load seller details.")).toBeInTheDocument();
  expect(within(alert).getByRole("button", { name: "Retry" })).toBeInTheDocument();
});
