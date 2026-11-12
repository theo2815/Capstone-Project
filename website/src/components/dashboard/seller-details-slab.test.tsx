import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { SellerDetailsSlab } from "./seller-details-slab";

const { showToast } = vi.hoisted(() => ({ showToast: vi.fn() }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ showToast, dismissToast: vi.fn() }) }));
// The real check needs a camera and AWS; these buttons stand in for its three outcomes.
vi.mock("./live-selfie-check", () => ({
  default: (p: { onComplete: () => void; onError: (kind: string) => void; onCancel: () => void }) => (
    <div>
      <button type="button" onClick={p.onComplete}>finish check</button>
      <button type="button" onClick={() => p.onError("camera")}>deny camera</button>
      <button type="button" onClick={p.onCancel}>close check</button>
    </div>
  ),
}));

let client: QueryClient;
const login = () => useAuthStore.getState().setUser({ id: "seller-a", role: "PHOTOGRAPHER", name: "A", email: "a@example.test", createdAt: "2026-09-10" });
const image = (name: string, bytes = 10, type = "image/jpeg") => new File([new Uint8Array(bytes)], name, { type });
const record = (over: Record<string, unknown>) => ({
  status: "submitted", reviewState: "manual_review", legalName: "Juan Dela Cruz", idType: "passport", address: "Cebu City", phone: "0917 123 4567",
  hasIdBack: false, hasBirForm: false, rejectionReason: null, submittedAt: "2026-11-07T01:00:00Z", reviewedAt: null,
  uploadsAllowed: true, identityGateStartsOn: null, ...over,
});
const MISSING = { status: "missing", reviewState: "missing", uploadsAllowed: false, identityGateStartsOn: null };
const SESSION = {
  sessionId: "sess-1", region: "ap-southeast-5", attemptsLeftToday: 2, identityPoolId: null,
  credentials: { accessKeyId: "AK", secretAccessKey: "SK", sessionToken: "TK", expiration: "2026-11-10T10:00:00Z" },
};
const SUBMIT = "/me/photographer/seller-details";
const START = "/me/photographer/seller-details/liveness-session";
const LIVE_CHECK = "/me/photographer/seller-details/live-check";

function renderSlab(details: unknown) {
  vi.spyOn(api, "get").mockResolvedValue(details);
  return render(<QueryClientProvider client={client}><SellerDetailsSlab /></QueryClientProvider>);
}

// Answers by path; an Error value is thrown instead.
function mockPost(over: Record<string, unknown> = {}) {
  return vi.spyOn(api, "post").mockImplementation((async (path: string) => {
    const value = path in over ? over[path] : path === START ? SESSION : record({ status: "approved", reviewState: "auto_approved" });
    if (value instanceof Error) throw value;
    return value;
  }) as never);
}
const calls = (post: ReturnType<typeof mockPost>, path: string) => post.mock.calls.filter((c) => c[0] === path);

async function fillForm({ idType = "passport", phone = "+63 917 123 4567", back = true, bir = false, idNumber = "P1234567A" } = {}) {
  fireEvent.change(await screen.findByLabelText(/Legal name/), { target: { value: "Juan Dela Cruz" } });
  fireEvent.change(screen.getByLabelText("ID type"), { target: { value: idType } });
  if (idNumber) fireEvent.change(screen.getByLabelText(/^ID number/), { target: { value: idNumber } });
  fireEvent.change(screen.getByLabelText("Front of the ID"), { target: { files: [image("front.jpg")] } });
  if (back) fireEvent.change(screen.getByLabelText(/^Back of the/), { target: { files: [image("back.png", 10, "image/png")] } });
  fireEvent.change(screen.getByLabelText("A photo of you holding this ID"), { target: { files: [image("selfie.jpg")] } });
  fireEvent.change(screen.getByLabelText("Address"), { target: { value: "12 Osmeña Blvd, Cebu City" } });
  fireEvent.change(screen.getByLabelText("Phone number"), { target: { value: phone } });
  if (bir) fireEvent.change(screen.getByLabelText(/BIR Form 2303/), { target: { files: [image("bir.webp", 10, "image/webp")] } });
}
const send = () => fireEvent.click(screen.getByRole("button", { name: "Have an admin check instead" }));
const tick = () => fireEvent.click(screen.getByRole("checkbox"));
const sendLive = () => fireEvent.click(screen.getByRole("button", { name: "Take the live selfie and send" }));

beforeEach(() => { useAuthStore.getState().logout(); login(); client = new QueryClient({ defaultOptions: { queries: { retry: false } } }); });
afterEach(() => { client.clear(); vi.restoreAllMocks(); showToast.mockReset(); useAuthStore.getState().logout(); });

it("shows the form when nothing was sent yet", async () => {
  renderSlab(MISSING);
  expect(await screen.findByRole("button", { name: "Take the live selfie and send" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Have an admin check instead" })).toBeInTheDocument();
  expect(api.get).toHaveBeenCalledWith("/me/photographer/seller-details", expect.anything());
});

it("names the gate date until identity passes", async () => {
  renderSlab(record({ identityGateStartsOn: "2099-01-05" }));
  expect(await screen.findByText(/From Monday 5 January 2099, you need a passed identity check/)).toBeInTheDocument();
});

it("no gate date line once identity passed", async () => {
  renderSlab(record({ status: "approved", reviewState: "auto_approved", identityGateStartsOn: "2099-01-05" }));
  expect(await screen.findByText("Verified")).toBeInTheDocument();
  expect(screen.queryByText(/you need a passed identity check/)).not.toBeInTheDocument();
});

it("live path sends the session id and no score", async () => {
  renderSlab(MISSING);
  const post = mockPost();
  await fillForm();
  tick();
  sendLive();
  fireEvent.click(await screen.findByRole("button", { name: "finish check" }));
  await waitFor(() => expect(calls(post, SUBMIT)).toHaveLength(1));
  expect(calls(post, START)[0][1]).toEqual({ biometricConsent: true });
  const form = calls(post, SUBMIT)[0][1] as FormData;
  expect(form.get("livenessSessionId")).toBe("sess-1");
  expect(form.get("idNumber")).toBe("P1234567A");
  expect([...form.keys()].filter((k) => /score|confidence|similarity/i.test(k))).toEqual([]);
  expect(await screen.findByText("Verified")).toBeInTheDocument();
});

it("live path needs the tick; admin path does not and sends no livenessSessionId", async () => {
  renderSlab(MISSING);
  const post = mockPost({ [SUBMIT]: record({}) });
  await fillForm();
  sendLive();
  expect(await screen.findByText("Tick the box to agree before the live selfie.")).toBeInTheDocument();
  expect(post).not.toHaveBeenCalled();
  send();
  await waitFor(() => expect(calls(post, SUBMIT)).toHaveLength(1));
  expect(calls(post, START)).toHaveLength(0);
  expect((calls(post, SUBMIT)[0][1] as FormData).has("livenessSessionId")).toBe(false);
});

it("ID number is required, trimmed, and not shown in the on-file view", async () => {
  renderSlab(MISSING);
  const post = mockPost({ [SUBMIT]: record({}) });
  await fillForm({ idNumber: "" });
  send();
  await waitFor(() => expect(screen.getByLabelText(/^ID number/)).toHaveAttribute("aria-invalid", "true"));
  expect(post).not.toHaveBeenCalled();
  fireEvent.change(screen.getByLabelText(/^ID number/), { target: { value: "  P1234567A  " } });
  send();
  await waitFor(() => expect(calls(post, SUBMIT)).toHaveLength(1));
  expect((calls(post, SUBMIT)[0][1] as FormData).get("idNumber")).toBe("P1234567A");
  expect(await screen.findByText("Juan Dela Cruz")).toBeInTheDocument();
  expect(screen.queryByText("P1234567A")).not.toBeInTheDocument();
  expect(screen.queryByDisplayValue("P1234567A")).not.toBeInTheDocument();
});

it.each([
  ["LIVENESS_ATTEMPTS_EXHAUSTED", 429, "You've used today's 3 tries. Try again tomorrow, or have an admin check instead."],
  ["LIVENESS_UNAVAILABLE", 503, "The live check isn't available right now. Have an admin check instead."],
])("%s keeps the form and offers the admin check", async (code, httpStatus, copy) => {
  renderSlab(MISSING);
  const post = mockPost({ [START]: new ApiError([{ code, message: "server text" }], httpStatus) });
  await fillForm();
  tick();
  sendLive();
  expect(await screen.findByText(copy)).toBeInTheDocument();
  expect(screen.queryByText("server text")).not.toBeInTheDocument();
  expect(screen.getByLabelText(/Legal name/)).toHaveValue("Juan Dela Cruz");
  expect(screen.getByRole("button", { name: "Have an admin check instead" })).toBeEnabled();
  expect(calls(post, SUBMIT)).toHaveLength(0);
});

it("an expired session on submit asks to take the live check again", async () => {
  renderSlab(MISSING);
  mockPost({ [SUBMIT]: new ApiError([{ code: "LIVENESS_SESSION_INVALID", message: "server text" }], 409) });
  await fillForm();
  tick();
  sendLive();
  fireEvent.click(await screen.findByRole("button", { name: "finish check" }));
  expect(await screen.findByText("That live check ran out. Take it again.")).toBeInTheDocument();
  expect(screen.getByLabelText(/^ID number/)).toHaveValue("P1234567A");
});

it("a live-check error (camera denied) offers retry and admin check and keeps the values", async () => {
  renderSlab(MISSING);
  const post = mockPost();
  await fillForm();
  tick();
  sendLive();
  fireEvent.click(await screen.findByRole("button", { name: "deny camera" }));
  expect(await screen.findByText(/couldn't use your camera/)).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "finish check" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Take the live selfie and send" })).toBeEnabled();
  expect(screen.getByRole("button", { name: "Have an admin check instead" })).toBeEnabled();
  expect(screen.getByLabelText(/Legal name/)).toHaveValue("Juan Dela Cruz");
  expect(screen.getByLabelText(/^ID number/)).toHaveValue("P1234567A");
  expect(calls(post, SUBMIT)).toHaveLength(0);
});

it("a late onComplete after sign-out submits nothing", async () => {
  renderSlab(MISSING);
  const post = mockPost();
  await fillForm();
  tick();
  sendLive();
  const finish = await screen.findByRole("button", { name: "finish check" });
  useAuthStore.getState().logout();
  fireEvent.click(finish);
  await new Promise((r) => setTimeout(r, 0));
  expect(calls(post, SUBMIT)).toHaveLength(0);
  expect(calls(post, LIVE_CHECK)).toHaveLength(0);
});

it("manual_review row offers the live check only and calls submitLiveCheck without files", async () => {
  renderSlab(record({}));
  const post = mockPost({ [LIVE_CHECK]: record({ status: "approved", reviewState: "auto_approved" }) });
  await screen.findByText("Juan Dela Cruz");
  expect(screen.queryByRole("button", { name: "Have an admin check instead" })).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Front of the ID")).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText(/^ID number/), { target: { value: "P1234567A" } });
  tick();
  fireEvent.click(screen.getByRole("button", { name: "Take the live selfie now" }));
  fireEvent.click(await screen.findByRole("button", { name: "finish check" }));
  await waitFor(() => expect(calls(post, LIVE_CHECK)).toHaveLength(1));
  expect(calls(post, LIVE_CHECK)[0][1]).toEqual({ livenessSessionId: "sess-1", idNumber: "P1234567A" });
  expect(calls(post, SUBMIT)).toHaveLength(0);
  expect(await screen.findByText("Verified")).toBeInTheDocument();
});

it("a live check that stays in review clears the typed number and says an admin checks", async () => {
  renderSlab(record({}));
  mockPost({ [LIVE_CHECK]: record({}) });
  fireEvent.change(await screen.findByLabelText(/^ID number/), { target: { value: "P1234567A" } });
  tick();
  fireEvent.click(screen.getByRole("button", { name: "Take the live selfie now" }));
  fireEvent.click(await screen.findByRole("button", { name: "finish check" }));
  await waitFor(() => expect(showToast).toHaveBeenCalledWith({ kind: "success", message: "Live check sent. An admin will check your details." }));
  expect(screen.getByLabelText(/^ID number/)).toHaveValue("");
  expect(screen.getByRole("checkbox")).not.toBeChecked();
});

it("requires the back of the card only for a PhilSys ID", async () => {
  renderSlab(MISSING);
  const post = mockPost({ [SUBMIT]: record({}) });
  await fillForm({ idType: "philsys", back: false });
  expect(screen.getByLabelText("Back of the card (the side with the QR code)")).toBeInTheDocument();
  send();
  expect(await screen.findByText("Add the back of the card.")).toBeInTheDocument();
  expect(post).not.toHaveBeenCalled();
  fireEvent.change(screen.getByLabelText("ID type"), { target: { value: "passport" } });
  send();
  await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
  expect((post.mock.calls[0][1] as FormData).has("idBack")).toBe(false);
});

it.each([
  ["a bad phone number", { phone: "12345" }, "Phone number", null],
  ["a too-short ID number", { idNumber: "P1" }, /^ID number/, null],
  ["a PDF", {}, "Front of the ID", image("front.pdf", 10, "application/pdf")],
  ["a 6 MB photo", {}, "Front of the ID", image("huge.jpg", 6 * 1024 * 1024)],
])("shows a field error for %s and sends nothing", async (_name, options, label, file) => {
  renderSlab(MISSING);
  const post = vi.spyOn(api, "post");
  await fillForm(options);
  if (file) fireEvent.change(screen.getByLabelText(label), { target: { files: [file] } });
  send();
  await waitFor(() => expect(screen.getByLabelText(label)).toHaveAttribute("aria-invalid", "true"));
  expect(post).not.toHaveBeenCalled();
});

it("sends all nine parts of a complete form on the admin path", async () => {
  renderSlab(MISSING);
  const post = mockPost({ [SUBMIT]: record({ hasIdBack: true, hasBirForm: true }) });
  await fillForm({ bir: true });
  send();
  await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
  expect(post.mock.calls[0][0]).toBe(SUBMIT);
  const form = post.mock.calls[0][1] as FormData;
  expect([...form.keys()].sort()).toEqual(["address", "birForm", "idBack", "idFront", "idNumber", "idType", "legalName", "phone", "selfie"]);
  expect(form.get("phone")).toBe("+63 917 123 4567");
  expect(await screen.findByText("An admin is checking your details")).toBeInTheDocument();
  expect(showToast).toHaveBeenCalled();
});

it("shows the rejection reason first and the form", async () => {
  renderSlab(record({ status: "rejected", reviewState: "rejected", rejectionReason: "The ID photo is blurry." }));
  expect(await screen.findByText(/The ID photo is blurry\./)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Take the live selfie and send" })).toBeInTheDocument();
});

it("shows a summary and a way to send new details while checking", async () => {
  renderSlab(record({}));
  expect(await screen.findByText("An admin is checking your details")).toBeInTheDocument();
  expect(screen.getByText("Juan Dela Cruz")).toBeInTheDocument();
  expect(screen.getByText("Passport")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Take the live selfie and send" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Send new details" }));
  expect(screen.getByRole("button", { name: "Take the live selfie and send" })).toBeInTheDocument();
});

it.each([
  [true, "You can upload meanwhile."],
  [false, "Uploads open when they approve."],
])("manual review with uploadsAllowed %s says whether uploads wait", async (uploadsAllowed, line) => {
  renderSlab(record({ uploadsAllowed }));
  expect(await screen.findByText(line)).toBeInTheDocument();
  expect(screen.queryByText(/score|confidence/i)).toBeNull();
});

it("a verified record says you can upload", async () => {
  renderSlab(record({ status: "approved", reviewState: "auto_approved" }));
  expect(await screen.findByText("Verified")).toBeInTheDocument();
  expect(screen.getByText("You can upload.")).toBeInTheDocument();
});

it("reads reviewState, not status: an unknown state is shown like manual review", async () => {
  renderSlab(record({ status: "approved", reviewState: "something_new" }));
  expect(await screen.findByText("An admin is checking your details")).toBeInTheDocument();
});

it("puts a server error on the phone field under that field", async () => {
  renderSlab(MISSING);
  vi.spyOn(api, "post").mockRejectedValue(new ApiError([{ code: "VALIDATION_ERROR", message: "This phone number is not accepted.", field: "phone" }], 400));
  await fillForm();
  send();
  await waitFor(() => expect(screen.getByLabelText("Phone number")).toHaveAccessibleDescription("This phone number is not accepted."));
});
