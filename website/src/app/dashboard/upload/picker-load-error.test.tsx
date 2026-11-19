import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import UploadPickerPage from "./page";

vi.mock("@/hooks/use-can-upload", () => ({ useCanUpload: () => ({ kind: "ok" }) }));

let client: QueryClient;
afterEach(() => { cleanup(); client?.clear(); vi.restoreAllMocks(); useAuthStore.getState().logout(); });

it("a failed read of the photographer's own events says so, and Retry loads them", async () => {
  useAuthStore.getState().setUser({ id: "p-1", role: "PHOTOGRAPHER", name: "Jessa", email: "p@example.test", createdAt: "2026-09-01" });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  let mineFails = true;
  vi.spyOn(api, "get").mockImplementation((path: string) => {
    if (path.startsWith("/me/photographer/events")) {
      return mineFails
        ? Promise.reject(new Error("boom"))
        : Promise.resolve({ items: [{ id: "own", name: "Own Run", date: "2026-11-29", location: "SRP, Cebu City", state: "upcoming", photoCount: 0, ownedByMe: true, reviewStatus: "approved" }], total: 1, offset: 0, limit: 200 });
    }
    return Promise.resolve({ items: [], total: 0, offset: 0, limit: 50 });
  });
  render(<QueryClientProvider client={client}><UploadPickerPage /></QueryClientProvider>);

  expect(await screen.findByText("Couldn't load your own events.")).toBeInTheDocument();
  mineFails = false;
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  await waitFor(() => expect(screen.queryByText("Couldn't load your own events.")).not.toBeInTheDocument());
  expect(await screen.findByText("Own Run")).toBeInTheDocument();
});
