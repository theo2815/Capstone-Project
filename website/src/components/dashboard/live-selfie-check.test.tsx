import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ErrorState, FaceLivenessDetectorCoreProps } from "@aws-amplify/ui-react-liveness";
import LiveSelfieCheck from "@/components/dashboard/live-selfie-check";
import { LIVENESS_BINARY_PATH } from "@/components/dashboard/live-selfie-detector";

vi.mock("@aws-amplify/ui-react/styles.css", () => ({}));
// The fake detector exposes the callbacks it was given so a test can fire them.
const detector = vi.hoisted(() => ({ props: null as null | FaceLivenessDetectorCoreProps }));
vi.mock("@aws-amplify/ui-react-liveness", () => ({
  FaceLivenessDetectorCore: (props: FaceLivenessDetectorCoreProps) => {
    detector.props = props;
    return <div data-testid="detector" />;
  },
}));

const credentials = {
  accessKeyId: "AKIA-test",
  secretAccessKey: "secret-test",
  sessionToken: "token-test",
  expiration: "2026-11-10T08:15:00Z",
};

function renderCheck(container?: HTMLElement) {
  const handlers = { onComplete: vi.fn(), onError: vi.fn(), onCancel: vi.fn() };
  render(
    <LiveSelfieCheck sessionId="session-1" region="ap-southeast-5" credentials={credentials} {...handlers} />,
    container ? { container } : undefined,
  );
  return handlers;
}

describe("LiveSelfieCheck", () => {
  beforeEach(() => {
    detector.props = null;
  });

  it("calls onComplete once when the analysis completes", async () => {
    const h = renderCheck();
    await screen.findByTestId("detector");
    await detector.props!.onAnalysisComplete();
    expect(h.onComplete).toHaveBeenCalledTimes(1);
    expect(h.onError).not.toHaveBeenCalled();
  });

  it("maps a camera-permission error to onError('camera')", async () => {
    const h = renderCheck();
    await screen.findByTestId("detector");
    detector.props!.onError!({ state: "CAMERA_ACCESS_ERROR", error: new Error("Permission denied") });
    expect(h.onError).toHaveBeenCalledWith("camera");
  });

  it.each([
    ["DEFAULT_CAMERA_NOT_FOUND_ERROR", "camera"],
    ["TIMEOUT", "timeout"],
    ["FRESHNESS_TIMEOUT", "timeout"],
    ["CONNECTION_TIMEOUT", "connection"],
    ["SERVER_ERROR", "connection"],
    ["MULTIPLE_FACES_ERROR", "other"],
  ])("maps %s to onError('%s')", async (state, kind) => {
    const h = renderCheck();
    await screen.findByTestId("detector");
    detector.props!.onError!({ state: state as ErrorState, error: new Error("x") });
    expect(h.onError).toHaveBeenCalledWith(kind);
  });

  it("renders into document.body", async () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    renderCheck(host);
    const dialog = await screen.findByRole("dialog");
    expect(dialog.closest("body")).toBe(document.body);
    expect(host.contains(dialog)).toBe(false);
  });

  it("gives the detector the server's session, region, keys and our self-hosted files", async () => {
    renderCheck();
    await screen.findByTestId("detector");
    expect(detector.props!.sessionId).toBe("session-1");
    expect(detector.props!.region).toBe("ap-southeast-5");
    expect(detector.props!.config!.binaryPath).toBe(LIVENESS_BINARY_PATH);
    await expect(detector.props!.config!.credentialProvider!()).resolves.toEqual({
      accessKeyId: "AKIA-test",
      secretAccessKey: "secret-test",
      sessionToken: "token-test",
      expiration: new Date("2026-11-10T08:15:00Z"),
    });
  });

  it("calls onCancel on Escape and on the detector's own cancel", async () => {
    const h = renderCheck();
    await screen.findByTestId("detector");
    await userEvent.keyboard("{Escape}");
    detector.props!.onUserCancel!();
    await waitFor(() => expect(h.onCancel).toHaveBeenCalledTimes(2));
  });
});

describe("self-hosted liveness WASM", () => {
  it("matches the installed @tensorflow/tfjs-backend-wasm byte for byte", () => {
    const pkg = join(process.cwd(), "node_modules/@tensorflow/tfjs-backend-wasm");
    const version = JSON.parse(readFileSync(join(pkg, "package.json"), "utf8")).version;
    expect(LIVENESS_BINARY_PATH).toBe(`/liveness/tfjs-backend-wasm-${version}/`);
    const served = join(process.cwd(), "public", LIVENESS_BINARY_PATH);
    const wasm = readdirSync(join(pkg, "dist")).filter((f) => f.endsWith(".wasm"));
    expect(wasm.length).toBeGreaterThan(0);
    for (const f of wasm) {
      expect(readFileSync(join(served, f)).equals(readFileSync(join(pkg, "dist", f))), f).toBe(true);
    }
  });
});
