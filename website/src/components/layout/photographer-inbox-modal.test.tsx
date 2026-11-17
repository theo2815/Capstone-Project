import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { useInboxNotificationsWs } from "@/hooks/use-inbox-notifications-ws";
import type { PhotographerMessage } from "@/lib/photographer-messages";
import { PhotographerInboxModal } from "./photographer-inbox-modal";

vi.mock("@/lib/auth", () => ({ getAccessToken: () => "fixture-token" }));
vi.mock("@/hooks/use-confirmation", () => ({ useConfirmation: () => ({ confirm: vi.fn() }) }));

class Socket {
  onopen = () => {};
  onclose = () => {};
  onmessage: (event: { data: string }) => void = () => {};
  close = vi.fn();
}
const owner = { id: "owner-a", role: "PHOTOGRAPHER" as const, name: "Owner A", email: "a@example.test", createdAt: "2026-01-01T00:00:00Z" };
const note: PhotographerMessage = {
  id: "message-1", kind: "relevance_notice", title: "Your photos in Cebu Marathon are off sale",
  body: "So our system has taken all your photos in this event off sale. https://evil.example/appeal",
  sourceDecisionId: null, createdAt: "2026-11-10T03:00:00Z", readAt: null, relevanceCaseId: "case-1",
};
let client: QueryClient;
let rows: PhotographerMessage[];
let sockets: Socket[];

beforeEach(() => {
  useAuthStore.getState().startSession(owner);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  rows = [note];
  sockets = [];
  vi.spyOn(api, "getWithTotal").mockImplementation(async () => ({ data: rows, total: rows.length }));
  vi.stubGlobal("WebSocket", vi.fn(function () { const socket = new Socket(); sockets.push(socket); return socket; }));
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

function wrap(ui: ReactNode) {
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}
function LiveInbox() {
  useInboxNotificationsWs("PHOTOGRAPHER", true);
  return <PhotographerInboxModal isOpen onClose={() => {}} />;
}

it("the note links to the case page", async () => {
  rows = [note, { ...note, id: "message-2", relevanceCaseId: null }];
  wrap(<PhotographerInboxModal isOpen onClose={() => {}} />);
  expect((await screen.findAllByText("Uploads"))[0]).toBeVisible();
  // Only the message's case id builds a link; nothing in the body becomes one, and a note without an id has none.
  const links = screen.getAllByRole("link");
  expect(links).toHaveLength(1);
  expect(links[0]).toHaveAccessibleName(/View details/);
  expect(links[0]).toHaveAttribute("href", "/dashboard/relevance/case-1");
});

it("a note pushed live carries the link without a reload", async () => {
  rows = [];
  wrap(<LiveInbox />);
  expect(await screen.findByText(/No messages yet/)).toBeVisible();
  rows = [note];
  // The push only triggers the inbox's own refresh (REST stays authoritative); the photographer does nothing.
  await act(async () => sockets[0].onmessage({ data: JSON.stringify({ type: "message.created", message: note }) }));
  await waitFor(() => expect(screen.getByRole("link", { name: /View details/ })).toHaveAttribute("href", "/dashboard/relevance/case-1"));
});
