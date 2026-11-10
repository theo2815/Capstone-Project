import { render, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import LiveSelfieCheck from "@/components/dashboard/live-selfie-check";

// The detector chunk fails to download (bad network, stale deploy).
vi.mock("@/components/dashboard/live-selfie-detector", () => {
  throw new Error("chunk load failed");
});

it("reports onError('load') when the detector cannot download", async () => {
  const onError = vi.fn();
  render(
    <LiveSelfieCheck
      sessionId="session-1"
      region="ap-southeast-5"
      credentials={{ accessKeyId: "a", secretAccessKey: "s", sessionToken: "t", expiration: "2026-11-10T08:15:00Z" }}
      onComplete={vi.fn()}
      onError={onError}
      onCancel={vi.fn()}
    />,
  );
  await waitFor(() => expect(onError).toHaveBeenCalledWith("load"));
});
