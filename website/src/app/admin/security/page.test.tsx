import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import * as mfa from "@/lib/api-admin-mfa";
import * as utils from "@/lib/utils";
import Page from "./page";
import { expectSectionIntro } from "@/test/admin-section-intro";
import { ROUTES } from "@/lib/constants";

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
  const buttons = await screen.findAllByRole("button", { name: "Enable 2-step verification" });
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
  fireEvent.click(await screen.findByRole("button", { name: "Enable 2-step verification" }));
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
  fireEvent.click(await screen.findByRole("button", { name: action === "confirm" ? "Enable 2-step verification" : "Turn off" }));
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
  expect(await screen.findByText("That code didn't work. Please try again.")).toBeInTheDocument();
});

it("shows an alert with Retry, not Off, when the status cannot load", async () => {
  vi.mocked(mfa.fetchMfaStatus).mockRejectedValueOnce(new Error("down"));
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't load your two-step verification status.");
  expect(screen.queryByText(/Off/)).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Enable 2-step verification" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByRole("button", { name: "Enable 2-step verification" })).toBeInTheDocument();
});

it("submits the code with Enter from the labelled input", async () => {
  vi.spyOn(mfa, "enrollMfa").mockResolvedValue({ secret: "local-secret", otpauthUri: "otpauth://test" });
  const confirm = vi.spyOn(mfa, "confirmMfa").mockResolvedValue({ recoveryCodes: ["aaaa-1111"] });
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: "Enable 2-step verification" }));
  const input = await screen.findByLabelText("6-digit code");
  await userEvent.type(input, "123456{Enter}");
  await waitFor(() => expect(confirm).toHaveBeenCalledWith("123456"));
});

it("copies the secret and the recovery codes, and downloads the codes", async () => {
  const copy = vi.spyOn(utils, "copyToClipboard").mockResolvedValue(true);
  vi.spyOn(mfa, "enrollMfa").mockResolvedValue({ secret: "local-secret", otpauthUri: "otpauth://test" });
  vi.spyOn(mfa, "confirmMfa").mockResolvedValue({ recoveryCodes: ["aaaa-1111", "bbbb-2222"] });
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: "Enable 2-step verification" }));
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
  fireEvent.click(await screen.findByRole("button", { name: "Enable 2-step verification" }));
  fireEvent.click(await screen.findByRole("button", { name: "Copy key" }));
  await waitFor(() => expect(window.getSelection()?.toString()).toBe("local-secret"));
});

it("treats an empty status reply as unknown, not off", async () => {
  vi.mocked(mfa.fetchMfaStatus).mockResolvedValueOnce(null);
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't load your two-step verification status.");
  expect(screen.queryByRole("button", { name: "Enable 2-step verification" })).not.toBeInTheDocument();
});

it("draws the QR from exactly the enroll response's otpauthUri and still shows the key", async () => {
  // Spaces, %-escapes and & prove the value is not trimmed or re-encoded.
  const uri = "otpauth://totp/QuickPitik%20Admin:a%2Bb%40x.ph?secret=JBSWY3DPEHPK3PXP&issuer=QuickPitik%20Admin ";
  qr.values.length = 0;
  vi.spyOn(mfa, "enrollMfa").mockResolvedValue({ secret: "JBSWY3DPEHPK3PXP", otpauthUri: uri });
  render(<QueryClientProvider client={client}><Page /></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: "Enable 2-step verification" }));
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
