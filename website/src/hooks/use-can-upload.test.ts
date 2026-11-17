import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCanUpload } from "@/hooks/use-can-upload";
import type { SellerDetails } from "@/lib/api-seller-details";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";

// The seller-details read (AA-I15): undefined = still loading or failed.
const seller = vi.hoisted(() => ({ data: undefined as Partial<SellerDetails> | undefined }));
vi.mock("@/hooks/use-seller-details", () => ({ useSellerDetails: () => ({ query: { data: seller.data } }) }));

// The completeness gate is now derived purely from the backend verification
// poll (status + missing[]), never recomputed from field stores. These pin the
// branch order the banner + railguard depend on.
describe("useCanUpload", () => {
  beforeEach(() => {
    seller.data = undefined;
    usePhotographerSettingsStore.setState({
      verificationStatus: "incomplete",
      verificationMissing: null,
      verificationMissingCodes: null,
      verificationAwaitingAdmin: false,
      suspendedAt: null,
      suspensionReason: null,
    });
  });

  it("suspended beats every other state, even approved", () => {
    usePhotographerSettingsStore.setState({
      verificationStatus: "approved",
      verificationMissing: [],
      suspendedAt: "2026-09-01T00:00:00Z",
      suspensionReason: "under review",
    });
    const { result } = renderHook(() => useCanUpload());
    expect(result.current).toEqual({ kind: "suspended", reason: "under review" });
  });

  it("approved and un-suspended is ok", () => {
    usePhotographerSettingsStore.setState({
      verificationStatus: "approved",
      verificationMissing: [],
    });
    const { result } = renderHook(() => useCanUpload());
    expect(result.current).toEqual({ kind: "ok" });
  });

  it("a non-empty backend missing[] is incomplete and carries every item", () => {
    usePhotographerSettingsStore.setState({
      verificationStatus: "incomplete",
      verificationMissing: ["profile picture", "payout account"],
    });
    const { result } = renderHook(() => useCanUpload());
    expect(result.current).toEqual({
      kind: "incomplete",
      missing: ["profile picture", "payout account"],
      missingLabel: "profile picture",
    });
  });

  it("all fields filled but not yet approved is pending", () => {
    usePhotographerSettingsStore.setState({
      verificationStatus: "pending",
      verificationMissing: [],
    });
    const { result } = renderHook(() => useCanUpload());
    expect(result.current).toEqual({ kind: "pending" });
  });

  it("a held pending profile tells the photographer an admin is reviewing it", () => {
    usePhotographerSettingsStore.setState({ verificationStatus: "pending", verificationMissing: [], verificationAwaitingAdmin: true });
    const { result } = renderHook(() => useCanUpload());
    expect(result.current).toEqual({ kind: "pending", awaitingAdmin: true });
  });

  // Withdraw and the settings page's local flip both move the status away from pending without a new server answer.
  it("a stale hold flag is ignored once the status is no longer pending", () => {
    usePhotographerSettingsStore.setState({ verificationStatus: "incomplete", verificationMissing: [], verificationAwaitingAdmin: true });
    const { result } = renderHook(() => useCanUpload());
    expect(result.current).toEqual({ kind: "pending" });
  });

  it("approved profile + uploadsAllowed false + manual_review → identity", () => {
    usePhotographerSettingsStore.setState({ verificationStatus: "approved", verificationMissing: [] });
    seller.data = { status: "submitted", reviewState: "manual_review", uploadsAllowed: false };
    const { result } = renderHook(() => useCanUpload());
    expect(result.current).toEqual({ kind: "identity", reviewState: "manual_review" });
  });

  it("approved profile + uploadsAllowed true (gate off, in review) stays ok", () => {
    usePhotographerSettingsStore.setState({ verificationStatus: "approved", verificationMissing: [] });
    seller.data = { status: "submitted", reviewState: "manual_review", uploadsAllowed: true };
    const { result } = renderHook(() => useCanUpload());
    expect(result.current).toEqual({ kind: "ok" });
  });

  it("seller query loading → ok (no flash block)", () => {
    usePhotographerSettingsStore.setState({ verificationStatus: "approved", verificationMissing: [] });
    const { result } = renderHook(() => useCanUpload());
    expect(result.current).toEqual({ kind: "ok" });
  });

  it("suspended beats identity", () => {
    usePhotographerSettingsStore.setState({
      verificationStatus: "approved",
      verificationMissing: [],
      suspendedAt: "2026-09-01T00:00:00Z",
      suspensionReason: null,
    });
    seller.data = { status: "missing", reviewState: "missing", uploadsAllowed: false };
    const { result } = renderHook(() => useCanUpload());
    expect(result.current).toEqual({ kind: "suspended", reason: null });
  });

  it("pending beats identity", () => {
    usePhotographerSettingsStore.setState({ verificationStatus: "pending", verificationMissing: [] });
    seller.data = { status: "submitted", reviewState: "manual_review", uploadsAllowed: false };
    const { result } = renderHook(() => useCanUpload());
    expect(result.current).toEqual({ kind: "pending" });
  });

  it("before the poll lands (missing null) it never flashes ok/ready", () => {
    usePhotographerSettingsStore.setState({
      verificationStatus: "incomplete",
      verificationMissing: null,
    });
    const { result } = renderHook(() => useCanUpload());
    expect(result.current.kind).toBe("incomplete");
  });
});
