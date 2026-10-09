import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as removalApi from "@/lib/api-photo-removal";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import type { PhotoRemovalRequestDto, RemovalHistoryDto } from "@/types/photo-removal";
import { RemovalRequestHistory } from "./request-history";
import { openCase } from "./removal-fixtures";

vi.mock("@/lib/api-photo-removal");

const entry = (id: string, extra: Partial<RemovalHistoryDto> = {}): RemovalHistoryDto => ({
  id, requestId: "case-1", itemId: null, reviewId: null, photoIdSnapshot: null, actorType: "ADMIN", actorIdSnapshot: "admin-a",
  action: "VERIFIED", fromState: "DRAFT", toState: "OPEN", reason: "Ticket checked", occurredAt: "2026-10-08T02:00:00Z",
  commandId: `command-${id}`, metadata: { storageKey: "s3://bucket/private" }, ...extra,
});
const guestCase: PhotoRemovalRequestDto = { ...openCase, guestBuyerRemovalNotices: [
  { noticeId: "notice-1", orderId: "order-9", photoId: "photo-a1", occurredAt: "2026-10-09T00:00:00Z", status: "SENT", recipientEmail: "guest@example.test" },
] };
let client: QueryClient;

beforeEach(() => {
  useAuthStore.getState().startSession(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  vi.mocked(removalApi.getAdminRemovalRequest).mockResolvedValue(guestCase);
  // The server returns history oldest first.
  vi.mocked(removalApi.listRemovalHistory).mockImplementation(async (_id, filters = {}) => filters.offset
    ? { items: [entry("h51", { actorType: "SYSTEM", actorIdSnapshot: null, action: "REVIEW_OVERDUE", fromState: null, toState: null, reason: "Later entry", occurredAt: "2026-10-15T03:00:00Z" })], total: 51, offset: 50, limit: 50 }
    : { items: [entry("h1", { action: "CREATED", fromState: null, toState: "DRAFT" })], total: 51, offset: 0, limit: 50 });
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.resetAllMocks(); });

function renderHistory() {
  return render(<QueryClientProvider client={client}><RemovalRequestHistory requestId="case-1" /></QueryClientProvider>);
}

it("pages the immutable history from the server without exposing metadata", async () => {
  const user = userEvent.setup();
  renderHistory();
  expect(await screen.findByText(/CREATED/)).toBeInTheDocument();
  expect(screen.getByText(/admin-a/)).toBeInTheDocument();
  expect(screen.getByText(/Ticket checked/)).toBeInTheDocument();
  expect(screen.queryByText(/s3:\/\//)).not.toBeInTheDocument();
  expect(removalApi.listRemovalHistory).toHaveBeenCalledWith("case-1", { offset: 0, limit: 50 }, expect.anything());
  await user.click(screen.getByRole("button", { name: "Later →" }));
  expect(await screen.findByText(/Later entry/)).toBeInTheDocument();
  expect(screen.getByText(/REVIEW_OVERDUE/)).toBeInTheDocument();
  expect(screen.getByText(/SYSTEM/)).toBeInTheDocument();
  expect(removalApi.listRemovalHistory).toHaveBeenLastCalledWith("case-1", { offset: 50, limit: 50 }, expect.anything());
});

it("records a guest notice follow-up and keeps the note after a failure", async () => {
  const user = userEvent.setup();
  let fail!: (error: Error) => void;
  vi.mocked(removalApi.addRemovalSupportUpdate)
    .mockReturnValueOnce(new Promise((_, reject) => { fail = reject; }))
    .mockResolvedValueOnce(entry("h2", { action: "SUPPORT_UPDATED", actorType: "SUPPORT" }));
  renderHistory();
  expect(await screen.findByRole("heading", { name: "Guest buyer notices" })).toBeInTheDocument();
  expect(screen.getAllByText(/guest@example\.test/).length).toBeGreaterThan(0);
  await user.type(screen.getByLabelText("Support update"), "Emailed the guest buyer about the removal");
  await user.selectOptions(screen.getByLabelText("Guest buyer notice (optional)"), "notice-1");
  await user.click(screen.getByRole("button", { name: "Record update" }));
  expect(screen.getByRole("button", { name: "Recording…" })).toBeDisabled();
  await act(async () => { fail(new Error("Gateway timeout")); });
  expect(await screen.findByText("Gateway timeout")).toBeInTheDocument();
  expect(screen.getByLabelText("Support update")).toHaveValue("Emailed the guest buyer about the removal");
  await user.click(screen.getByRole("button", { name: "Record update" }));
  await waitFor(() => expect(removalApi.addRemovalSupportUpdate).toHaveBeenCalledTimes(2));
  const [first, second] = vi.mocked(removalApi.addRemovalSupportUpdate).mock.calls.map((call) => call[1]);
  expect(second).toEqual(first);
  expect(first).toMatchObject({ note: "Emailed the guest buyer about the removal", noticeId: "notice-1", expectedVersion: 7 });
  expect(await screen.findByText("Support update recorded.")).toBeInTheDocument();
  expect(screen.getByLabelText("Support update")).toHaveValue("");
});
