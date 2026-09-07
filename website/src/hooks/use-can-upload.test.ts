import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { useCanUpload } from "@/hooks/use-can-upload";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";

// The completeness gate is now derived purely from the backend verification
// poll (status + missing[]), never recomputed from field stores. These pin the
// branch order the banner + railguard depend on.
describe("useCanUpload", () => {
  beforeEach(() => {
    usePhotographerSettingsStore.setState({
      verificationStatus: "incomplete",
      verificationMissing: null,
      verificationMissingCodes: null,
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

  it("before the poll lands (missing null) it never flashes ok/ready", () => {
    usePhotographerSettingsStore.setState({
      verificationStatus: "incomplete",
      verificationMissing: null,
    });
    const { result } = renderHook(() => useCanUpload());
    expect(result.current.kind).toBe("incomplete");
  });
});
