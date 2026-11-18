import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as adminApi from "@/lib/api-admin";
import { adminUserKeys } from "@/hooks/use-admin-users";
import { useAuthStore } from "@/store/auth-store";
import { admin, adminUser } from "@/test/fixtures/admin";
import AdminPhotographersPage from "./page";

vi.mock("@/lib/api-admin", async (orig) => ({ ...(await orig<typeof import("@/lib/api-admin")>()), fetchAdminUsers: vi.fn() }));
vi.mock("next/navigation", () => ({
  usePathname: () => "/admin/photographers",
  useSearchParams: () => new URLSearchParams(""),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
}));

let client: QueryClient;
beforeEach(() => {
  useAuthStore.getState().startSession(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.resetAllMocks(); });

it("keeps the loaded list on screen when a background refresh fails", async () => {
  client.setQueryData(adminUserKeys.list(admin.id), [adminUser]);
  vi.mocked(adminApi.fetchAdminUsers).mockRejectedValue(new Error("Network down"));
  render(<QueryClientProvider client={client}><AdminPhotographersPage /></QueryClientProvider>);
  expect(screen.getByText("Test Studio")).toBeInTheDocument();
  await client.refetchQueries({ queryKey: adminUserKeys.list(admin.id) });
  await waitFor(() => expect(client.getQueryState(adminUserKeys.list(admin.id))?.status).toBe("error"));
  expect(screen.getByText("Test Studio")).toBeInTheDocument();
  expect(screen.queryByText(/Couldn.t load photographers/)).not.toBeInTheDocument();
});

it("shows the error and Retry when the first load fails", async () => {
  vi.mocked(adminApi.fetchAdminUsers).mockRejectedValue(new Error("Network down"));
  render(<QueryClientProvider client={client}><AdminPhotographersPage /></QueryClientProvider>);
  expect(await screen.findByText(/Couldn.t load photographers/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
});
