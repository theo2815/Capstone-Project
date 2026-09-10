import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { getCatalogWithOverrides } from "@/lib/event-catalog";
import { useAdminDisputeStore } from "./admin-dispute-store";
import { useAdminPayoutReportStore } from "./admin-payout-report-store";
import { useAdminEventOverridesStore } from "./admin-event-overrides-store";
import { useToastStore } from "./toast-store";

const disputeId = "dispute-1";
const reportId = "report-1";
const reason = "Reviewed the supporting evidence.";
const resolution = { resolution: "refund_partial" as const, refundAmount: 100, reason };

const decisions = [
  {
    label: "resolve dispute",
    method: "post" as const,
    endpoint: `/admin/disputes/${disputeId}/resolve`,
    payload: resolution,
    store: useAdminDisputeStore,
    run: () => useAdminDisputeStore.getState().resolve(disputeId, resolution),
    id: disputeId,
    status: "resolved",
  },
  {
    label: "deny dispute",
    method: "post" as const,
    endpoint: `/admin/disputes/${disputeId}/deny`,
    payload: { reason },
    store: useAdminDisputeStore,
    run: () => useAdminDisputeStore.getState().deny(disputeId, reason),
    id: disputeId,
    status: "denied",
  },
  {
    label: "escalate dispute",
    method: "post" as const,
    endpoint: `/admin/disputes/${disputeId}/escalate`,
    payload: { reason },
    store: useAdminDisputeStore,
    run: () => useAdminDisputeStore.getState().escalate(disputeId, reason),
    id: disputeId,
    status: "escalated",
  },
  {
    label: "acknowledge payout report",
    method: "fetch" as const,
    endpoint: `/admin/payouts/reports/${reportId}/acknowledge`,
    payload: { method: "PATCH", body: JSON.stringify({ reply: reason }) },
    store: useAdminPayoutReportStore,
    run: () => useAdminPayoutReportStore.getState().acknowledge(reportId, reason),
    id: reportId,
    status: "acknowledged",
  },
  {
    label: "resolve payout report",
    method: "fetch" as const,
    endpoint: `/admin/payouts/reports/${reportId}/resolve`,
    payload: { method: "PATCH", body: JSON.stringify({ resolutionNote: reason }) },
    store: useAdminPayoutReportStore,
    run: () => useAdminPayoutReportStore.getState().resolve(reportId, reason),
    id: reportId,
    status: "resolved",
  },
];

beforeEach(() => {
  useAdminDisputeStore.getState().clear();
  useAdminPayoutReportStore.getState().clear();
  useAdminEventOverridesStore.getState().clear();
  useToastStore.getState().clear();
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => vi.restoreAllMocks());

describe.each(decisions)("$label", ({ method, endpoint, payload, store, run, id, status }) => {
  it("preserves the prior record and decision log when the backend rejects", async () => {
    // Retain a prior confirmed decision too, rather than merely an empty store.
    vi.spyOn(api, method).mockResolvedValueOnce({}).mockRejectedValueOnce(new Error("Backend unavailable"));
    expect(await run()).toBe(true);
    const before = store.getState();

    expect(await run()).toBe(false);

    expect(store.getState()).toBe(before);
    expect(useToastStore.getState().toasts).toEqual([
      expect.objectContaining({ kind: "error", message: expect.stringContaining("please try again") }),
    ]);
  });

  it("waits for confirmation before committing exactly one decision", async () => {
    let confirm!: (value: unknown) => void;
    const response = new Promise((resolve) => { confirm = resolve; });
    const request = vi.spyOn(api, method).mockReturnValue(response);
    const before = store.getState();

    const outcome = run();
    expect(request).toHaveBeenCalledWith(endpoint, payload);
    expect(store.getState()).toBe(before);
    expect(useToastStore.getState().toasts).toEqual([]);

    confirm({});
    expect(await outcome).toBe(true);
    expect(store.getState().overrides[id]?.status).toBe(status);
    expect(store.getState().log).toHaveLength(1);
    expect(store.getState().log[0].decision).toBe(status);
  });
});

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
