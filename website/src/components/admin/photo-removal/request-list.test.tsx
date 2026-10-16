import { cleanup, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as removalApi from "@/lib/api-photo-removal";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { RemovalRequestList } from "./request-list";
import { draftCase } from "./removal-fixtures";

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
