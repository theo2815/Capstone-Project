import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as removalApi from "@/lib/api-photo-removal";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { RemovalRequestList } from "./request-list";
import { draftCase, openCase, runnerCase } from "./removal-fixtures";

vi.mock("@/lib/api-photo-removal");
let client: QueryClient;
beforeEach(() => { useAuthStore.getState().startSession(admin); client = new QueryClient({ defaultOptions: { queries: { retry: false } } }); });
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.resetAllMocks(); });

it("counts case photos from server phase counts because list rows carry no items", async () => {
  vi.mocked(removalApi.listAdminRemovalRequests).mockResolvedValue({ items: [{ ...draftCase, status: "OPEN", items: [],
    phaseCounts: { ...draftCase.phaseCounts, candidate: 9, verified: 2, excluded: 1, removalAuthorized: 1, removed: 1, retained: 0, retainedRestricted: 0 } }],
    total: 1, offset: 0, limit: 20 });
  render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  expect(await screen.findByText(/4 photos/)).toBeInTheDocument();
});

it("the list shows a Runner badge and filters by source", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.listAdminRemovalRequests).mockResolvedValue({ items: [
    { ...runnerCase, items: [] }, { ...openCase, id: "case-2", items: [] },
  ], total: 2, offset: 0, limit: 20 });
  render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  const [runnerRow, supportRow] = await screen.findAllByRole("link");
  expect(within(runnerRow).getByText("Runner")).toBeVisible();
  expect(within(runnerRow).queryByText("RUNNER-case-1")).toBeNull();
  expect(within(runnerRow).getByText("Open")).toBeVisible();
  expect(within(supportRow).getByText("Support")).toBeVisible();
  expect(within(supportRow).getByText("TICKET-42")).toBeVisible();

  await user.selectOptions(screen.getByLabelText("Source"), "RUNNER");
  await waitFor(() => expect(removalApi.listAdminRemovalRequests).toHaveBeenLastCalledWith({ offset: 0, limit: 20, source: "RUNNER" }, expect.anything()));
  await user.click(screen.getByLabelText("Needs action"));
  await waitFor(() => expect(removalApi.listAdminRemovalRequests).toHaveBeenLastCalledWith({ offset: 0, limit: 20, source: "RUNNER", needsAction: true }, expect.anything()));
});
