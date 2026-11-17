import { expect, it, vi } from "vitest";
import Page from "./page";

const redirect = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({ redirect }));

it("the old detail URL redirects to the queue drawer", async () => {
  await Page({ params: Promise.resolve({ id: "dispute-1" }) });
  expect(redirect).toHaveBeenCalledWith("/admin/disputes?row=dispute-1");
});
