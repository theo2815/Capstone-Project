import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { getCatalogWithOverrides } from "@/lib/event-catalog";
import { useAdminEventOverridesStore } from "./admin-event-overrides-store";
import { useToastStore } from "./toast-store";

beforeEach(() => {
  useAdminEventOverridesStore.getState().clear();
  useToastStore.getState().clear();
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => vi.restoreAllMocks());

describe("delete event", () => {
  const event = {
    id: "event-1", slug: "cebu-race", name: "Cebu Race", date: "2026-09-10",
    city: "Cebu", location: "Cebu", state: "live" as const, status: "ACTIVE" as const,
    photoCount: 10, participantCount: 20,
  };

  it.each([false, true])("retains the catalog and reports an error on failure (session submission: %s)", async (submitted) => {
    useAdminEventOverridesStore.setState({
      submissions: submitted ? [event] : [],
      overrides: { [event.id]: { name: "Updated race name" } },
    });
    const before = useAdminEventOverridesStore.getState();
    const catalog = getCatalogWithOverrides([event]);
    vi.spyOn(api, "delete").mockRejectedValueOnce(new Error("Cannot delete event"));

    await expect(before.deleteEvent(event.id)).rejects.toThrow("Cannot delete event");

    expect(useAdminEventOverridesStore.getState()).toBe(before);
    expect(getCatalogWithOverrides([event])).toEqual(catalog);
    expect(useToastStore.getState().toasts).toEqual([
      expect.objectContaining({ kind: "error", message: expect.stringContaining("delete event") }),
    ]);
  });

  it("keeps a created/refetched event visible until deletion is confirmed", async () => {
    useAdminEventOverridesStore.setState({ submissions: [event] });
    let confirm!: (value: unknown) => void;
    vi.spyOn(api, "delete").mockReturnValue(new Promise((resolve) => { confirm = resolve; }));

    const outcome = useAdminEventOverridesStore.getState().deleteEvent(event.id);
    expect(getCatalogWithOverrides([event]).map((row) => row.id)).toContain(event.id);
    expect(useAdminEventOverridesStore.getState().tombstones).toEqual([]);

    confirm({});
    await outcome;
    expect(api.delete).toHaveBeenCalledWith(`/admin/events/${event.id}`);
    expect(useAdminEventOverridesStore.getState().submissions).toEqual([]);
    expect(useAdminEventOverridesStore.getState().tombstones).toEqual([event.id]);
    expect(getCatalogWithOverrides([event])).toEqual([]);
  });
});
