import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import * as mfa from "@/lib/api-admin-mfa";
import Page from "./page";

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
