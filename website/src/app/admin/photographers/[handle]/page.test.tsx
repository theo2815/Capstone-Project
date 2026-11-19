import { cleanup, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { admin, adminUser, adminUserSettings } from "@/test/fixtures/admin";
import AdminPhotographerDetailPage from "./page";

const ID = "3f0c5a2e-8b1d-4c6e-9a7f-2d4b6c8e0a1b";
const params = vi.hoisted(() => ({ handle: "" }));
vi.mock("next/navigation", () => ({
  useParams: () => params,
  usePathname: () => `/admin/photographers/${params.handle}`,
  useSearchParams: () => new URLSearchParams(""),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  notFound: () => { throw new Error("NEXT_NOT_FOUND"); },
}));

const row = { ...adminUser, userId: ID, handle: null };
let client: QueryClient;
beforeEach(() => {
  useAuthStore.getState().startSession(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(api, "get").mockImplementation(async (url: string) => {
    if (url === `/admin/users/${ID}`) return { ...row, decisionLog: [] };
    if (url === `/admin/users/${ID}/settings`) return { ...adminUserSettings, userId: ID, handle: null };
    if (url === `/admin/users/${adminUser.userId}`) return { ...adminUser, decisionLog: [] };
    if (url === `/admin/users/${adminUser.userId}/settings`) return adminUserSettings;
    if (url.startsWith("/admin/users?")) return { items: [adminUser], total: 1, offset: 0, limit: 200 };
    return [];
  });
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

const mount = () => render(<QueryClientProvider client={client}><AdminPhotographerDetailPage /></QueryClientProvider>);

it("opens a photographer with no handle from a user id in the URL", async () => {
  params.handle = ID;
  mount();
  expect(await screen.findByRole("heading", { level: 1, name: /Test Studio/ })).toBeInTheDocument();
});

it("shows Pending in the same warning tone as the directory card", async () => {
  params.handle = adminUser.handle!;
  mount();
  await screen.findByRole("heading", { level: 1, name: /Test Studio/ });
  expect(screen.getAllByText("Pending")[0]).toHaveClass("text-warning-text");
});
