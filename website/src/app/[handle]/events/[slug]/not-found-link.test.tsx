import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import * as eventsApi from "@/lib/api-events";
import * as photographerApi from "@/lib/api-photographer-public";
import HandleEventPage from "./page";

vi.mock("next/navigation", () => ({
  useParams: () => ({ handle: "/evil.example", slug: "no-such-event" }),
  notFound: vi.fn(),
}));
vi.mock("@/components/layout/site-header", () => ({ SiteHeader: () => null }));
vi.mock("@/components/profile-shell", () => ({ ProfileShellFooter: () => null }));

it("the portfolio link on a missing event stays on the site", async () => {
  vi.spyOn(eventsApi, "fetchEventDetail").mockResolvedValue(null);
  vi.spyOn(photographerApi, "fetchPublicPhotographer").mockResolvedValue(null);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><HandleEventPage /></QueryClientProvider>);
  expect(await screen.findByRole("link", { name: /View their portfolio/ }))
    .toHaveAttribute("href", "/%2Fevil.example");
});
