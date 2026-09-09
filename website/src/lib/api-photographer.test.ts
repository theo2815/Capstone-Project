import { beforeEach, expect, it, vi } from "vitest";
import { permanentDeletePhotographerPhotos, restorePhotographerPhotos, trashPhotographerPhotos } from "./api-photographer";

const { post } = vi.hoisted(() => ({ post: vi.fn() }));
vi.mock("@/lib/api", () => ({ api: { post }, refreshAccessToken: vi.fn() }));
beforeEach(() => { post.mockReset(); });
const ids = (count: number) => Array.from({ length: count }, (_, i) => `photo-${i}`);

it("batches all selected pages within the 500-photo mutation limit", async () => {
  post.mockImplementation(async (_path, body) => ({ affected: body.photoIds.length }));
  const result = await trashPhotographerPhotos("event", ids(1001));
  expect(post.mock.calls.map(([, body]) => body.photoIds.length)).toEqual([500, 500, 1]);
  expect(result).toEqual({ affected: 1001, failed: [] });
});

it("keeps completed restore counts and identifies a failed batch for retry", async () => {
  post.mockResolvedValueOnce({ affected: 500 }).mockRejectedValueOnce(new Error("offline"))
    .mockResolvedValueOnce({ affected: 1 });
  const result = await restorePhotographerPhotos("event", ids(1001));
  expect(result).toEqual({ affected: 501, failed: ids(1001).slice(500, 1000) });
});

it("combines deletion results, purchase holds, and failed requests without losing IDs", async () => {
  const selected = ids(251);
  post.mockResolvedValueOnce({ deleted: selected.slice(0, 90), blockedSold: selected.slice(90, 100) })
    .mockRejectedValueOnce(new Error("timeout"))
    .mockResolvedValueOnce({ deleted: selected.slice(200, 250), blockedSold: [], blockedPending: [selected[250]] });
  const result = await permanentDeletePhotographerPhotos("event", selected);
  expect(post.mock.calls.map(([, body]) => body.photoIds.length)).toEqual([100, 100, 51]);
  expect(result.deleted).toEqual([...selected.slice(0, 90), ...selected.slice(200, 250)]);
  expect(result.blockedSold).toEqual(selected.slice(90, 100));
  expect(result.blockedPending).toEqual([selected[250]]);
  expect(result.failed).toEqual(selected.slice(100, 200));
});
