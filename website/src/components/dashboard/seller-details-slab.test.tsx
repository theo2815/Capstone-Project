import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { SellerDetailsSlab } from "./seller-details-slab";

const { showToast } = vi.hoisted(() => ({ showToast: vi.fn() }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ showToast, dismissToast: vi.fn() }) }));

let client: QueryClient;
const login = () => useAuthStore.getState().setUser({ id: "seller-a", role: "PHOTOGRAPHER", name: "A", email: "a@example.test", createdAt: "2026-09-10" });
const image = (name: string, bytes = 10, type = "image/jpeg") => new File([new Uint8Array(bytes)], name, { type });
const record = (over: Record<string, unknown>) => ({
  status: "submitted", legalName: "Juan Dela Cruz", idType: "passport", address: "Cebu City", phone: "0917 123 4567",
  hasIdBack: false, hasBirForm: false, rejectionReason: null, submittedAt: "2026-11-07T01:00:00Z", reviewedAt: null, ...over,
});

function renderSlab(details: unknown) {
  vi.spyOn(api, "get").mockResolvedValue(details);
  return render(<QueryClientProvider client={client}><SellerDetailsSlab /></QueryClientProvider>);
}

async function fillForm({ idType = "passport", phone = "+63 917 123 4567", back = true, bir = false } = {}) {
  fireEvent.change(await screen.findByLabelText(/Legal name/), { target: { value: "Juan Dela Cruz" } });
  fireEvent.change(screen.getByLabelText("ID type"), { target: { value: idType } });
  fireEvent.change(screen.getByLabelText("Front of the ID"), { target: { files: [image("front.jpg")] } });
  if (back) fireEvent.change(screen.getByLabelText(/^Back of the/), { target: { files: [image("back.png", 10, "image/png")] } });
  fireEvent.change(screen.getByLabelText("A photo of you holding this ID"), { target: { files: [image("selfie.jpg")] } });
  fireEvent.change(screen.getByLabelText("Address"), { target: { value: "12 Osmeña Blvd, Cebu City" } });
  fireEvent.change(screen.getByLabelText("Phone number"), { target: { value: phone } });
  if (bir) fireEvent.change(screen.getByLabelText(/BIR Form 2303/), { target: { files: [image("bir.webp", 10, "image/webp")] } });
}
const send = () => fireEvent.click(screen.getByRole("button", { name: "Send seller details" }));

beforeEach(() => { useAuthStore.getState().logout(); login(); client = new QueryClient({ defaultOptions: { queries: { retry: false } } }); });
afterEach(() => { client.clear(); vi.restoreAllMocks(); showToast.mockReset(); useAuthStore.getState().logout(); });

it("shows the form when nothing was sent yet", async () => {
  renderSlab({ status: "missing" });
  expect(await screen.findByRole("button", { name: "Send seller details" })).toBeInTheDocument();
  expect(api.get).toHaveBeenCalledWith("/me/photographer/seller-details", expect.anything());
});

it("requires the back of the card only for a PhilSys ID", async () => {
  renderSlab({ status: "missing" });
  const post = vi.spyOn(api, "post").mockResolvedValue(record({}));
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
  ["a PDF", {}, "Front of the ID", image("front.pdf", 10, "application/pdf")],
  ["a 6 MB photo", {}, "Front of the ID", image("huge.jpg", 6 * 1024 * 1024)],
])("shows a field error for %s and sends nothing", async (_name, options, label, file) => {
  renderSlab({ status: "missing" });
  const post = vi.spyOn(api, "post");
  await fillForm(options);
  if (file) fireEvent.change(screen.getByLabelText(label), { target: { files: [file] } });
  send();
  await waitFor(() => expect(screen.getByLabelText(label)).toHaveAttribute("aria-invalid", "true"));
  expect(post).not.toHaveBeenCalled();
});

it("sends all eight parts of a complete form", async () => {
  renderSlab({ status: "missing" });
  const post = vi.spyOn(api, "post").mockResolvedValue(record({ hasIdBack: true, hasBirForm: true }));
  await fillForm({ bir: true });
  send();
  await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
  expect(post.mock.calls[0][0]).toBe("/me/photographer/seller-details");
  const form = post.mock.calls[0][1] as FormData;
  expect([...form.keys()].sort()).toEqual(["address", "birForm", "idBack", "idFront", "idType", "legalName", "phone", "selfie"]);
  expect(form.get("phone")).toBe("+63 917 123 4567");
  expect(await screen.findByText("We're checking your details. You can upload meanwhile.")).toBeInTheDocument();
  expect(showToast).toHaveBeenCalled();
});

it("shows the rejection reason first and the form", async () => {
  renderSlab(record({ status: "rejected", rejectionReason: "The ID photo is blurry." }));
  expect(await screen.findByText(/The ID photo is blurry\./)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Send seller details" })).toBeInTheDocument();
});

it("shows a summary and a way to send new details while checking", async () => {
  renderSlab(record({}));
  expect(await screen.findByText("We're checking your details. You can upload meanwhile.")).toBeInTheDocument();
  expect(screen.getByText("Juan Dela Cruz")).toBeInTheDocument();
  expect(screen.getByText("Passport")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Send seller details" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Send new details" }));
  expect(screen.getByRole("button", { name: "Send seller details" })).toBeInTheDocument();
});

it("puts a server error on the phone field under that field", async () => {
  renderSlab({ status: "missing" });
  vi.spyOn(api, "post").mockRejectedValue(new ApiError([{ code: "VALIDATION_ERROR", message: "This phone number is not accepted.", field: "phone" }], 400));
  await fillForm();
  send();
  await waitFor(() => expect(screen.getByLabelText("Phone number")).toHaveAccessibleDescription("This phone number is not accepted."));
});
