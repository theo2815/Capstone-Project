import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { admin } from "@/test/fixtures/admin";
import { AdminDenyDisputeModal } from "./admin-deny-dispute-modal";

const reasons = [
  { code: "preference", label: "Doesn't like how it looks", runnerText: "The photo arrived as shown.", noteRequired: false },
  { code: "bought_together", label: "Bought together with other photos", runnerText: "Bought together.", noteRequired: false },
  { code: "other", label: "Other (write the reason)", runnerText: "", noteRequired: true },
];

let client: QueryClient;
beforeEach(() => {
  useAuthStore.getState().setUser(admin);
  vi.spyOn(api, "get").mockResolvedValue({ items: reasons, total: reasons.length });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => {
  cleanup();
  client.clear();
  useAuthStore.getState().logout();
  vi.restoreAllMocks();
});

function renderModal(props: Partial<React.ComponentProps<typeof AdminDenyDisputeModal>> = {}) {
  const onSubmit = vi.fn();
  render(
    <QueryClientProvider client={client}>
      <AdminDenyDisputeModal open onClose={() => {}} onSubmit={onSubmit} disputeId="d1" {...props} />
    </QueryClientProvider>,
  );
  return onSubmit;
}

it("no reason is picked until the admin picks one", async () => {
  renderModal();
  const dialog = screen.getByRole("dialog", { name: "Deny d1" });
  const radios = await within(dialog).findAllByRole("radio");
  expect(radios.every((r) => !(r as HTMLInputElement).checked)).toBe(true);
  expect(within(dialog).getByRole("button", { name: "Deny claim" })).toBeDisabled();
});

it("a bought-together dispute opens with Bought together picked", async () => {
  renderModal({ defaultCode: "bought_together" });
  const dialog = screen.getByRole("dialog", { name: "Deny d1" });
  expect(await within(dialog).findByRole("radio", { name: /Bought together/ })).toBeChecked();
  expect(dialog).toHaveTextContent("The runner will read: Bought together.");
  expect(within(dialog).getByRole("button", { name: "Deny claim" })).toBeEnabled();
});

it("Other needs a note", async () => {
  const user = userEvent.setup();
  renderModal();
  const dialog = screen.getByRole("dialog", { name: "Deny d1" });
  await user.click(await within(dialog).findByRole("radio", { name: /Other/ }));
  expect(within(dialog).getByRole("button", { name: "Deny claim" })).toBeDisabled();
  await user.type(within(dialog).getByRole("textbox"), "Photo matches the order.");
  expect(within(dialog).getByRole("button", { name: "Deny claim" })).toBeEnabled();
});

it("confirm sends the code and the note", async () => {
  const user = userEvent.setup();
  const onSubmit = renderModal();
  const dialog = screen.getByRole("dialog", { name: "Deny d1" });
  await user.click(await within(dialog).findByRole("radio", { name: /Doesn't like/ }));
  await user.type(within(dialog).getByRole("textbox"), " Looks fine. ");
  await user.click(within(dialog).getByRole("button", { name: "Deny claim" }));
  expect(onSubmit).toHaveBeenCalledWith("preference", "Looks fine.");
});

it("when the reasons fail to load, the modal shows the error and a Retry, and Confirm stays disabled", async () => {
  const user = userEvent.setup();
  vi.mocked(api.get).mockRejectedValueOnce(new Error("down"));
  renderModal();
  const dialog = screen.getByRole("dialog", { name: "Deny d1" });
  expect(await within(dialog).findByRole("alert")).toHaveTextContent("Couldn't load the deny reasons.");
  expect(within(dialog).getByRole("button", { name: "Deny claim" })).toBeDisabled();
  await user.click(within(dialog).getByRole("button", { name: "Retry" }));
  expect(await within(dialog).findAllByRole("radio")).toHaveLength(3);
});
