import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import * as mfa from "@/lib/api-admin-mfa";
import * as utils from "@/lib/utils";
import Page from "./page";
import { expectSectionIntro } from "@/test/admin-section-intro";
import { ROUTES } from "@/lib/constants";
import { ApiError } from "@/lib/api";

// Records exactly what the page hands the QR package.
const qr = vi.hoisted(() => ({ values: [] as string[] }));
vi.mock("qrcode.react", () => ({
  QRCodeSVG: ({ value, ...rest }: { value: string } & React.SVGProps<SVGSVGElement>) => {
    qr.values.push(value);
    return <svg {...rest} />;
  },
}));

let client: QueryClient;
beforeEach(() => {
  useAuthStore.getState().setUser(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(mfa, "fetchMfaStatus").mockResolvedValue({ enabled: false });
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

it("excludes enrollment across mounted views, releases after rejection and never caches secrets", async () => {
  let reject!: (error: Error) => void;
  const enroll = vi.spyOn(mfa, "enrollMfa").mockReturnValue(new Promise((_, no) => { reject = no; }));
  render(<QueryClientProvider client={client}><Page /><Page /></QueryClientProvider>);
  const buttons = await screen.findAllByRole("button", { name: "Turn on two-step verification" });
  act(() => { buttons[0]!.click(); buttons[1]!.click(); });
  await waitFor(() => expect(enroll).toHaveBeenCalledTimes(1));
  await waitFor(() => buttons.forEach((button) => expect(button).toBeDisabled()));
  await act(async () => { reject(new Error("Unavailable")); });
  expect(await screen.findByText("Could not start setup. Please try again.")).toBeInTheDocument();
  enroll.mockResolvedValue({ secret: "private-secret", otpauthUri: "otpauth://test" });
  fireEvent.click(buttons[0]!);
  await screen.findByText("private-secret");
  expect(JSON.stringify(client.getMutationCache().getAll().map((row) => row.state.data))).not.toContain("private-secret");
  expect(sessionStorage.getItem("quickpitik-admin-action-intents") ?? "").not.toContain("private-secret");
});

it("discards enrollment secrets arriving after logout", async () => {
  let resolve!: (value: Awaited<ReturnType<typeof mfa.enrollMfa>>) => void;
  vi.spyOn(mfa, "enrollMfa").mockReturnValue(new Promise((yes) => { resolve = yes; }));
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: "Turn on two-step verification" }));
  await act(async () => { useAuthStore.getState().logout(); resolve({ secret: "old-secret", otpauthUri: "otpauth://test" }); });
  expect(screen.queryByText("old-secret")).not.toBeInTheDocument();
});

it.each(["confirm", "disable"])("%s keeps the code during failure and blocks editing and dismissal while pending", async (action) => {
  vi.mocked(mfa.fetchMfaStatus).mockResolvedValue({ enabled: action === "disable" });
  vi.spyOn(mfa, "enrollMfa").mockResolvedValue({ secret: "local-secret", otpauthUri: "otpauth://test" });
  let reject!: (error: Error) => void;
  const pending = new Promise<never>((_, no) => { reject = no; });
  const request = action === "confirm" ? vi.spyOn(mfa, "confirmMfa").mockReturnValue(pending) : vi.spyOn(mfa, "disableMfa").mockReturnValue(pending);
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: action === "confirm" ? "Turn on two-step verification" : "Turn off" }));
  const input = await screen.findByRole("textbox");
  fireEvent.change(input, { target: { value: "123456" } });
  const submit = screen.getByRole("button", { name: action === "confirm" ? "Turn on" : "Turn off" });
  act(() => { submit.click(); submit.click(); });
  await waitFor(() => expect(request).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(input).toBeDisabled());
  expect(submit).toHaveAttribute("aria-busy", "true");
  expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
  await act(async () => { reject(new Error("Bad code")); });
  await waitFor(() => expect(input).toBeEnabled());
  expect(input).toHaveValue("123456");
  expect(await screen.findByText("That code didn't work. Wait for a new one and try again. If it keeps failing, check that your phone's date and time are right.")).toBeInTheDocument();
});

it.each(["confirm", "disable"])("%s shows the clock hint, not the server's words, for a wrong code", async (action) => {
  vi.mocked(mfa.fetchMfaStatus).mockResolvedValue({ enabled: action === "disable" });
  vi.spyOn(mfa, "enrollMfa").mockResolvedValue({ secret: "local-secret", otpauthUri: "otpauth://test" });
  const wrong = new ApiError([{ code: "INVALID_MFA_CODE", message: "That code is not valid." }], 401);
  if (action === "confirm") vi.spyOn(mfa, "confirmMfa").mockRejectedValue(wrong);
  else vi.spyOn(mfa, "disableMfa").mockRejectedValue(wrong);
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: action === "confirm" ? "Turn on two-step verification" : "Turn off" }));
  fireEvent.change(await screen.findByRole("textbox"), { target: { value: "123456" } });
  fireEvent.click(screen.getByRole("button", { name: action === "confirm" ? "Turn on" : "Turn off" }));
  expect(await screen.findByText("That code didn't work. Wait for a new one and try again. If it keeps failing, check that your phone's date and time are right.")).toBeInTheDocument();
  expect(screen.queryByText("That code is not valid.")).not.toBeInTheDocument();
});

it("keeps the server's words for an error that is not a wrong code", async () => {
  vi.mocked(mfa.fetchMfaStatus).mockResolvedValue({ enabled: true });
  vi.spyOn(mfa, "disableMfa").mockRejectedValue(new ApiError([{ code: "ACCOUNT_LOCKED", message: "Too many tries. Try again later." }], 429));
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: "Turn off" }));
  fireEvent.change(await screen.findByRole("textbox"), { target: { value: "123456" } });
  fireEvent.click(screen.getByRole("button", { name: "Turn off" }));
  expect(await screen.findByText("Too many tries. Try again later.")).toBeInTheDocument();
});

it.each(["confirm", "disable"])("%s strips spaces from the code before sending it", async (action) => {
  vi.mocked(mfa.fetchMfaStatus).mockResolvedValue({ enabled: action === "disable" });
  vi.spyOn(mfa, "enrollMfa").mockResolvedValue({ secret: "local-secret", otpauthUri: "otpauth://test" });
  const request = action === "confirm"
    ? vi.spyOn(mfa, "confirmMfa").mockResolvedValue({ recoveryCodes: ["aaaa-1111"] })
    : vi.spyOn(mfa, "disableMfa").mockResolvedValue(undefined);
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: action === "confirm" ? "Turn on two-step verification" : "Turn off" }));
  const input = await screen.findByLabelText(action === "confirm" ? "6-digit code" : "6-digit code, or a recovery code");
  expect(input).not.toHaveAttribute("maxlength");
  fireEvent.change(input, { target: { value: " 123 456 " } });
  fireEvent.click(screen.getByRole("button", { name: action === "confirm" ? "Turn on" : "Turn off" }));
  await waitFor(() => expect(request).toHaveBeenCalledWith("123456"));
});

it("lists the set-up steps above the button while two-step verification is off", async () => {
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  await screen.findByRole("button", { name: "Turn on two-step verification" });
  const steps = screen.getAllByRole("listitem").map((item) => item.textContent);
  expect(steps).toEqual([
    "Install an authenticator app on your phone, for example Google Authenticator.",
    "Scan the QR code with the app.",
    "Type the 6-digit code the app shows. It changes about every 30 seconds.",
    "Save the recovery codes you are given at the end.",
  ]);
  expect(screen.queryByText(/2-step/)).not.toBeInTheDocument();
});

it("keeps the recovery codes until they are copied, and says what to do without the phone", async () => {
  vi.spyOn(utils, "copyToClipboard").mockResolvedValue(true);
  vi.spyOn(mfa, "enrollMfa").mockResolvedValue({ secret: "local-secret", otpauthUri: "otpauth://test" });
  vi.spyOn(mfa, "confirmMfa").mockResolvedValue({ recoveryCodes: ["aaaa-1111", "bbbb-2222"] });
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: "Turn on two-step verification" }));
  expect(await screen.findByText("In your authenticator app, scan this QR code. If you cannot scan it, add an account by typing the key below. The key is a set-up code, not a password.")).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("6-digit code"), { target: { value: "123456" } });
  fireEvent.click(screen.getByRole("button", { name: "Turn on" }));
  const saved = await screen.findByRole("button", { name: "I've saved them" });
  expect(screen.getByText("Save these recovery codes somewhere safe. This is the only time they are shown. If you do not have your phone, type one of them in the same box as the 6-digit code when you sign in. Each works once.")).toBeInTheDocument();
  expect(screen.getByText("If you lose both your phone and these codes, you cannot get back in by yourself. Ask the owner to reset your two-step verification.")).toBeInTheDocument();
  expect(saved).toBeDisabled();
  expect(screen.getByText("Copy or download the codes first.")).toBeInTheDocument();
  fireEvent.click(saved);
  expect(screen.getByText("aaaa-1111")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Copy codes" }));
  await waitFor(() => expect(saved).toBeEnabled());
  expect(screen.queryByText("Copy or download the codes first.")).not.toBeInTheDocument();
  fireEvent.click(saved);
  expect(await screen.findByText("Sign-in asks for a code from your authenticator app. To get new recovery codes, turn two-step verification off and on again.")).toBeInTheDocument();
  expect(screen.queryByText("aaaa-1111")).not.toBeInTheDocument();
});

it("says two-step verification is off once it is turned off", async () => {
  useToastStore.getState().clear();
  vi.mocked(mfa.fetchMfaStatus).mockResolvedValue({ enabled: true });
  vi.spyOn(mfa, "disableMfa").mockResolvedValue(undefined);
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: "Turn off" }));
  // A recovery code has letters, so a digits-only phone keypad would block it.
  expect(screen.getByLabelText("6-digit code, or a recovery code")).not.toHaveAttribute("inputmode", "numeric");
  fireEvent.change(screen.getByLabelText("6-digit code, or a recovery code"), { target: { value: "aaaa-1111" } });
  fireEvent.click(screen.getByRole("button", { name: "Turn off" }));
  await waitFor(() => expect(useToastStore.getState().toasts[0]).toMatchObject({ kind: "success", message: "Two-step verification is off." }));
  expect(await screen.findByRole("button", { name: "Turn on two-step verification" })).toBeInTheDocument();
});

it("shows an alert with Retry, not Off, when the status cannot load", async () => {
  vi.mocked(mfa.fetchMfaStatus).mockRejectedValueOnce(new Error("down"));
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't load your two-step verification status.");
  expect(screen.queryByText(/Off/)).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Turn on two-step verification" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByRole("button", { name: "Turn on two-step verification" })).toBeInTheDocument();
});

it("submits the code with Enter from the labelled input", async () => {
  vi.spyOn(mfa, "enrollMfa").mockResolvedValue({ secret: "local-secret", otpauthUri: "otpauth://test" });
  const confirm = vi.spyOn(mfa, "confirmMfa").mockResolvedValue({ recoveryCodes: ["aaaa-1111"] });
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: "Turn on two-step verification" }));
  const input = await screen.findByLabelText("6-digit code");
  await userEvent.type(input, "123456{Enter}");
  await waitFor(() => expect(confirm).toHaveBeenCalledWith("123456"));
});

it("copies the secret and the recovery codes, and downloads the codes", async () => {
  const copy = vi.spyOn(utils, "copyToClipboard").mockResolvedValue(true);
  vi.spyOn(mfa, "enrollMfa").mockResolvedValue({ secret: "local-secret", otpauthUri: "otpauth://test" });
  vi.spyOn(mfa, "confirmMfa").mockResolvedValue({ recoveryCodes: ["aaaa-1111", "bbbb-2222"] });
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: "Turn on two-step verification" }));
  fireEvent.click(await screen.findByRole("button", { name: "Copy key" }));
  await waitFor(() => expect(copy).toHaveBeenCalledWith("local-secret"));
  fireEvent.change(screen.getByLabelText("6-digit code"), { target: { value: "123456" } });
  fireEvent.click(screen.getByRole("button", { name: "Turn on" }));
  fireEvent.click(await screen.findByRole("button", { name: "Copy codes" }));
  await waitFor(() => expect(copy).toHaveBeenCalledWith("aaaa-1111\nbbbb-2222"));

  let saved: Blob | undefined;
  URL.createObjectURL = vi.fn((blob: Blob) => { saved = blob; return "blob:codes"; });
  URL.revokeObjectURL = vi.fn();
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    expect(this.download).toBe("quickpitik-recovery-codes.txt");
  });
  fireEvent.click(screen.getByRole("button", { name: "Download codes" }));
  expect(click).toHaveBeenCalledTimes(1);
  // jsdom's Blob has no text(); FileReader reads it.
  const reader = new FileReader();
  const read = new Promise((done) => { reader.onload = () => done(reader.result); });
  reader.readAsText(saved!);
  expect(await read).toContain("aaaa-1111\nbbbb-2222");
  await waitFor(() => expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:codes"));
});

it("selects the key for a manual copy when the clipboard refuses", async () => {
  vi.spyOn(utils, "copyToClipboard").mockResolvedValue(false);
  vi.spyOn(mfa, "enrollMfa").mockResolvedValue({ secret: "local-secret", otpauthUri: "otpauth://test" });
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: "Turn on two-step verification" }));
  fireEvent.click(await screen.findByRole("button", { name: "Copy key" }));
  await waitFor(() => expect(window.getSelection()?.toString()).toBe("local-secret"));
});

it("treats an empty status reply as unknown, not off", async () => {
  vi.mocked(mfa.fetchMfaStatus).mockResolvedValueOnce(null);
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't load your two-step verification status.");
  expect(screen.queryByRole("button", { name: "Turn on two-step verification" })).not.toBeInTheDocument();
});

it("draws the QR from exactly the enroll response's otpauthUri and still shows the key", async () => {
  // Spaces, %-escapes and & prove the value is not trimmed or re-encoded.
  const uri = "otpauth://totp/QuickPitik%20Admin:a%2Bb%40x.ph?secret=JBSWY3DPEHPK3PXP&issuer=QuickPitik%20Admin ";
  qr.values.length = 0;
  vi.spyOn(mfa, "enrollMfa").mockResolvedValue({ secret: "JBSWY3DPEHPK3PXP", otpauthUri: uri });
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: "Turn on two-step verification" }));
  expect(await screen.findByRole("img", { name: "QR code for your authenticator app" })).toBeInTheDocument();
  expect(new Set(qr.values)).toEqual(new Set([uri]));
  expect(screen.getByText("JBSWY3DPEHPK3PXP")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Copy key" })).toBeInTheDocument();
  expect(screen.queryByText(/coming soon/i)).not.toBeInTheDocument();
});

it("shows its guide intro, look-alike links and definitions without hover", () => {
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  expectSectionIntro(ROUTES.ADMIN_SECURITY);
});
