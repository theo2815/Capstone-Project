import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as removalApi from "@/lib/api-photo-removal";
import * as adminApi from "@/lib/api-admin";
import { admin, adminEvent } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { RemovalRequestIntake } from "./request-intake";
import { draftCase } from "./removal-fixtures";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: (url: string) => window.history.pushState({}, "", url) }),
}));
vi.mock("@/lib/api-photo-removal");
vi.mock("@/lib/api-admin", async (actual) => ({ ...(await actual<typeof adminApi>()), fetchAdminEvents: vi.fn() }));

let client: QueryClient;
beforeEach(() => {
  window.history.pushState({}, "", "/admin/photo-removal-requests/new");
  useAuthStore.getState().startSession(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  vi.mocked(adminApi.fetchAdminEvents).mockResolvedValue([
    { ...adminEvent, id: "event-public", name: "Cebu Marathon", visibility: "public" },
    { ...adminEvent, id: "event-unlisted", name: "Private Fun Run", visibility: "unlisted" },
  ]);
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.resetAllMocks(); });

function renderIntake() {
  return render(<QueryClientProvider client={client}><RemovalRequestIntake /></QueryClientProvider>);
}

async function fillIntake(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole("radio", { name: /Private Fun Run/ }));
  await user.type(screen.getByLabelText("Support reference"), "TICKET-42");
  await user.type(screen.getByLabelText("Requester contact"), "runner@example.test");
  await user.type(screen.getByLabelText("Reason"), "Runner asked to remove photos of them");
}

it("adminStaysInAdminForUnlistedEvent", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.createRemovalRequest).mockResolvedValue({ ...draftCase, id: "case-new", event: { id: "event-unlisted", name: "Private Fun Run" } });
  renderIntake();
  expect(await screen.findByText("UNLISTED")).toBeInTheDocument();
  expect(screen.getByText("PUBLIC")).toBeInTheDocument();
  await fillIntake(user);
  await user.click(screen.getByRole("button", { name: "Create case" }));
  await waitFor(() => expect(location.pathname).toBe("/admin/photo-removal-requests/case-new"));
  expect(location.pathname).toMatch(/^\/admin\/photo-removal-requests\//);
  expect(removalApi.createRemovalRequest).toHaveBeenCalledWith(expect.objectContaining({
    eventId: "event-unlisted", supportReference: "TICKET-42", requesterContact: "runner@example.test",
    reason: "Runner asked to remove photos of them", commandId: expect.any(String),
  }));
  expect(location.search).toBe("");
  for (const link of document.querySelectorAll("a")) {
    expect(link.getAttribute("target")).not.toBe("_blank");
    expect(link.getAttribute("href")).toMatch(/^\/admin\//);
  }
});

it("keeps input, blocks duplicates and reuses the command when a failed create is retried", async () => {
  const user = userEvent.setup();
  let fail!: (error: Error) => void;
  vi.mocked(removalApi.createRemovalRequest)
    .mockReturnValueOnce(new Promise((_, reject) => { fail = reject; }))
    .mockResolvedValueOnce({ ...draftCase, id: "case-retry" });
  renderIntake();
  await fillIntake(user);
  const submit = screen.getByRole("button", { name: "Create case" });
  await user.click(submit);
  expect(screen.getByRole("button", { name: "Creating…" })).toBeDisabled();
  await act(async () => { fail(new Error("Network down")); });
  expect(await screen.findByText("Network down")).toBeInTheDocument();
  expect(screen.getByLabelText("Support reference")).toHaveValue("TICKET-42");
  expect(location.pathname).toBe("/admin/photo-removal-requests/new");
  await user.click(screen.getByRole("button", { name: "Create case" }));
  await waitFor(() => expect(location.pathname).toBe("/admin/photo-removal-requests/case-retry"));
  const calls = vi.mocked(removalApi.createRemovalRequest).mock.calls;
  expect(calls).toHaveLength(2);
  expect(calls[1][0].commandId).toBe(calls[0][0].commandId);
});

it("lists only events that went live", async () => {
  vi.mocked(adminApi.fetchAdminEvents).mockResolvedValue([
    { ...adminEvent, id: "a", name: "Approved Run", reviewStatus: "approved" },
    { ...adminEvent, id: "c", name: "Repricing Run", reviewStatus: "change_pending" },
    { ...adminEvent, id: "p", name: "Pending Run", reviewStatus: "pending" },
    { ...adminEvent, id: "r", name: "Sent Back Run", reviewStatus: "rejected" },
  ]);
  renderIntake();
  expect(await screen.findByRole("radio", { name: /Approved Run/ })).toBeInTheDocument();
  expect(screen.getByRole("radio", { name: /Repricing Run/ })).toBeInTheDocument();
  expect(screen.queryByText("Pending Run")).not.toBeInTheDocument();
  expect(screen.queryByText("Sent Back Run")).not.toBeInTheDocument();
});

it("explains a runner ID the backend could not read, without sending it", async () => {
  const user = userEvent.setup();
  renderIntake();
  await fillIntake(user);
  const runnerId = screen.getByLabelText("Runner account ID (optional)");
  await user.type(runnerId, "runner@example.test");
  await user.click(screen.getByRole("button", { name: "Create case" }));
  expect(await screen.findByText(/must be the 36-character account ID/)).toBeInTheDocument();
  expect(runnerId).toHaveAttribute("aria-invalid", "true");
  expect(removalApi.createRemovalRequest).not.toHaveBeenCalled();
});
