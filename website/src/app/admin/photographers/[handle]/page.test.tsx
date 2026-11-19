import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
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
const record = {
  events: [{ eventId: "e-1", name: "Cebu City Marathon 2026", date: "2026-10-05", visibility: "public", photoCount: 1234 }],
  eventsTotal: 250, sellerDetailsStatus: "auto_approved", salesCount: 12, photoCount: 3400,
};
let detail: Record<string, unknown>;
let client: QueryClient;
beforeEach(() => {
  useAuthStore.getState().startSession(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  detail = { ...adminUser, decisionLog: [] };
  vi.spyOn(api, "get").mockImplementation(async (url: string) => {
    if (url === `/admin/users/${ID}`) return { ...row, decisionLog: [] };
    if (url === `/admin/users/${ID}/settings`) return { ...adminUserSettings, userId: ID, handle: null };
    if (url.endsWith("/record")) return record;
    if (url === `/admin/users/${adminUser.userId}`) return detail;
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

it("lists the events and facts from the admin record, not an empty local registry", async () => {
  params.handle = adminUser.handle!;
  mount();
  fireEvent.click(await screen.findByRole("button", { name: /Events covered/ }));
  expect(await screen.findByText("Cebu City Marathon 2026")).toBeInTheDocument();
  expect(screen.getByText(/Showing 1 of 250/)).toBeInTheDocument();
  expect(screen.getByText("Auto-approved")).toBeInTheDocument();
  expect(screen.getByText("3,400")).toBeInTheDocument();
  expect(screen.getByText("12")).toBeInTheDocument();
});

it("hides the facts and says so when an older backend has no record route", async () => {
  const get = vi.mocked(api.get).getMockImplementation()!;
  vi.mocked(api.get).mockImplementation(async (url: string, ...rest) => {
    if (url.endsWith("/record")) throw new ApiError([{ code: "NOT_FOUND", message: "Not found" }], 404);
    return get(url, ...rest);
  });
  params.handle = adminUser.handle!;
  mount();
  fireEvent.click(await screen.findByRole("button", { name: /Events covered/ }));
  expect(await screen.findByText("Event history isn't available yet.")).toBeInTheDocument();
  expect(screen.queryByText("Seller details")).not.toBeInTheDocument();
});

it("offers Approve for a held profile and sends it to the server", async () => {
  detail = { ...adminUser, adminHold: true, decisionLog: [] };
  const post = vi.spyOn(api, "post").mockResolvedValue({ ...adminUser, verificationStatus: "approved" });
  params.handle = adminUser.handle!;
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Approve" }));
  await waitFor(() => expect(post).toHaveBeenCalledWith(`/admin/users/${adminUser.userId}/approve`, { expectedRevision: 7 }));
});

it("shows no Approve for a held profile that is back to incomplete (the server would refuse it)", async () => {
  detail = { ...adminUser, verificationStatus: "incomplete", adminHold: true, decisionLog: [] };
  params.handle = adminUser.handle!;
  mount();
  await screen.findByRole("button", { name: "Suspend…" });
  expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();
});

it("shows no Approve for a profile that is not held", async () => {
  params.handle = adminUser.handle!;
  mount();
  await screen.findByRole("heading", { level: 1, name: /Test Studio/ });
  expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();
});

it("puts the admin actions before the review sections, so a phone reaches them first", async () => {
  params.handle = adminUser.handle!;
  mount();
  const actions = await screen.findByRole("complementary", { name: "Admin actions" });
  const firstSection = screen.getByRole("button", { name: /Completeness/ });
  expect(actions.compareDocumentPosition(firstSection) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
});

it("shows a styled Retry when the photographer cannot load", async () => {
  vi.mocked(api.get).mockRejectedValue(new Error("down"));
  params.handle = adminUser.handle!;
  mount();
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent(/Couldn.t load this photographer/);
  expect(screen.getByRole("button", { name: "Retry" })).toHaveAttribute("type", "button");
});
