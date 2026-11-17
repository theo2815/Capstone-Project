import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FaceLivenessDetectorCoreProps } from "@aws-amplify/ui-react-liveness";
import { ApiError } from "@/lib/api";
import * as removal from "@/lib/api-photo-removal";
import { LEGAL_VERSION } from "@/lib/legal/version";
import { useAuthStore } from "@/store/auth-store";
import { RequestForm } from "./request-form";

vi.mock("@aws-amplify/ui-react/styles.css", () => ({}));
// The real live check runs with a fake AWS detector; tests fire its callbacks.
const detector = vi.hoisted(() => ({ props: null as null | FaceLivenessDetectorCoreProps }));
vi.mock("@aws-amplify/ui-react-liveness", () => ({
  FaceLivenessDetectorCore: (props: FaceLivenessDetectorCoreProps) => {
    detector.props = props;
    return <div data-testid="detector" />;
  },
}));
vi.mock("@/lib/api-photo-removal", async (original) => ({
  ...(await original<typeof import("@/lib/api-photo-removal")>()),
  listRunnerRemovalReasons: vi.fn(),
  getRunnerRemovalEligibility: vi.fn(),
  startRunnerLiveness: vi.fn(),
  submitRunnerRemoval: vi.fn(),
}));
const events = vi.hoisted(() => ({ filters: [] as unknown[] }));
vi.mock("@/hooks/use-public-events", () => ({
  usePublicEvents: (filters: unknown) => {
    events.filters.push(filters);
    return [{ id: "ev-1", slug: "cebu-marathon", name: "Cebu Marathon", date: "2026-10-01" }];
  },
}));
vi.mock("@/lib/api-events", () => ({
  fetchEventDetail: vi.fn(async (slug: string) => (slug === "cebu-marathon" ? { id: "ev-1", slug, name: "Cebu Marathon" } : null)),
}));
const push = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({ useRouter: () => ({ push, replace: vi.fn() }) }));
vi.mock("@/hooks/use-auth", () => ({
  useAuth: () => ({ user: useAuthStore((s) => s.user), logout: vi.fn() }),
}));

const api = vi.mocked(removal);
const ELIGIBLE = { eligible: true, blockedReason: null, attemptsLeftToday: 3, emailVerified: true, openRequestId: null };
const SESSION = {
  sessionId: "sess-1", region: "ap-southeast-5", attemptsLeftToday: 2, identityPoolId: null,
  credentials: { accessKeyId: "AK", secretAccessKey: "SK", sessionToken: "TK", expiration: "2026-11-12T10:00:00Z" },
};
const REASONS = [{ code: "PRIVACY", label: "I don't want photos of me online" }, { code: "OTHER", label: "Something else" }];
const apiError = (code: string, status = 409) => new ApiError([{ code, message: code }], status);

let client: QueryClient;
const login = (role: "RUNNER" | "PHOTOGRAPHER" = "RUNNER") =>
  useAuthStore.getState().setUser({ id: "runner-a", role, name: "A", email: "a@example.test", createdAt: "2026-09-10" });

function renderForm(eventSlug: string | null = "cebu-marathon") {
  return render(<QueryClientProvider client={client}><RequestForm eventSlug={eventSlug} /></QueryClientProvider>);
}
const tick = async () => fireEvent.click(await screen.findByRole("checkbox"));
const startButton = () => screen.findByRole("button", { name: "Start live check" });
async function pickReason() {
  fireEvent.click(await screen.findByRole("button", { name: "Reason" }));
  fireEvent.click(await screen.findByRole("menuitem", { name: REASONS[0].label }));
}
async function start() {
  await pickReason();
  await tick();
  fireEvent.click(await startButton());
  await screen.findByTestId("detector");
}
const emailRoute = () => document.querySelector('a[href^="mailto:"]');

beforeEach(() => {
  useAuthStore.getState().logout();
  login();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  detector.props = null;
  events.filters = [];
  api.listRunnerRemovalReasons.mockResolvedValue(REASONS);
  api.getRunnerRemovalEligibility.mockResolvedValue(ELIGIBLE);
  api.startRunnerLiveness.mockResolvedValue(SESSION);
  api.submitRunnerRemoval.mockResolvedValue({ requestId: "req-1", status: "PENDING_ADMIN_REVIEW", hiddenCount: 2, suggestedCount: 1 });
});
afterEach(() => {
  client.clear();
  vi.clearAllMocks();
  useAuthStore.getState().logout();
});

describe("RequestForm", () => {
  it("start is disabled until the tick is set", async () => {
    renderForm();
    await pickReason();
    expect(await startButton()).toBeDisabled();
    await tick();
    expect(await startButton()).toBeEnabled();
  });

  it("start sends consentVersion equal to LEGAL_VERSION", async () => {
    renderForm();
    await start();
    expect(api.startRunnerLiveness).toHaveBeenCalledWith({ eventId: "ev-1", biometricConsent: true, consentVersion: LEGAL_VERSION });
  });

  it("the page has no file input", async () => {
    const { container } = renderForm();
    await start();
    expect(container.querySelector('input[type="file"]')).toBeNull();
    expect(document.querySelector('input[type="file"]')).toBeNull();
  });

  it("the live check modal renders into document.body", async () => {
    const { container } = renderForm();
    await start();
    const node = screen.getByTestId("detector");
    expect(container.contains(node)).toBe(false);
    expect(document.body.contains(node)).toBe(true);
  });

  it("reasons come from the API", async () => {
    renderForm();
    fireEvent.click(await screen.findByRole("button", { name: "Reason" }));
    expect(await screen.findByRole("menuitem", { name: "Something else" })).toBeInTheDocument();
    expect(api.listRunnerRemovalReasons).toHaveBeenCalled();
  });

  it("completing the check submits and opens the request", async () => {
    renderForm();
    await start();
    await detector.props!.onAnalysisComplete();
    await waitFor(() => expect(push).toHaveBeenCalledWith("/account/photo-deletion/req-1"));
    expect(api.submitRunnerRemoval).toHaveBeenCalledWith(
      { sessionId: "sess-1", eventId: "ev-1", reasonCode: "PRIVACY", details: undefined }, expect.any(String));
  });

  it("a retry reuses the same Idempotency-Key", async () => {
    api.submitRunnerRemoval.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    renderForm();
    await start();
    await detector.props!.onAnalysisComplete();
    fireEvent.click(await screen.findByRole("button", { name: "Send again" }));
    await waitFor(() => expect(api.submitRunnerRemoval).toHaveBeenCalledTimes(2));
    const [first, second] = api.submitRunnerRemoval.mock.calls;
    expect(second[1]).toBe(first[1]);
    expect(second[0].sessionId).toBe("sess-1");
  });

  it("a retry answered LIVENESS_SESSION_INVALID links to the request list", async () => {
    api.submitRunnerRemoval.mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockRejectedValueOnce(apiError("LIVENESS_SESSION_INVALID"));
    renderForm();
    await start();
    await detector.props!.onAnalysisComplete();
    fireEvent.click(await screen.findByRole("button", { name: "Send again" }));
    expect(await screen.findByText(/We may already have your request/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Your deletion requests/ })).toHaveAttribute("href", "/account/photo-deletion");
  });

  it("a first LIVENESS_SESSION_INVALID asks to take the check again", async () => {
    api.submitRunnerRemoval.mockRejectedValueOnce(apiError("LIVENESS_SESSION_INVALID"));
    renderForm();
    await start();
    await detector.props!.onAnalysisComplete();
    expect(await screen.findByText("That live check ran out. Take it again.")).toBeInTheDocument();
  });

  it("a submit answered UPLOAD_STORAGE_UNAVAILABLE links to the request list to withdraw", async () => {
    api.submitRunnerRemoval.mockRejectedValueOnce(apiError("UPLOAD_STORAGE_UNAVAILABLE", 503));
    renderForm();
    await start();
    await detector.props!.onAnalysisComplete();
    expect(await screen.findByText(/Your check could not be saved. Open your request to withdraw it/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Your deletion requests/ })).toHaveAttribute("href", "/account/photo-deletion");
  });

  it("LIVENESS_UNAVAILABLE shows try again in a minute and the email route", async () => {
    api.startRunnerLiveness.mockRejectedValueOnce(apiError("LIVENESS_UNAVAILABLE", 503));
    renderForm();
    await pickReason();
    await tick();
    fireEvent.click(await startButton());
    expect(await screen.findByText(/Busy right now. Try again in a minute./)).toBeInTheDocument();
    expect(emailRoute()).not.toBeNull();
  });

  it("RATE_LIMITED reads like LIVENESS_UNAVAILABLE", async () => {
    api.startRunnerLiveness.mockRejectedValueOnce(apiError("RATE_LIMITED", 429));
    renderForm();
    await pickReason();
    await tick();
    fireEvent.click(await startButton());
    expect(await screen.findByText(/Busy right now/)).toBeInTheDocument();
  });

  it("LIVENESS_FAILED shows the tries left from this check's session", async () => {
    api.submitRunnerRemoval.mockRejectedValueOnce(apiError("LIVENESS_FAILED", 422));
    renderForm();
    await start();
    await detector.props!.onAnalysisComplete();
    expect(await screen.findByText("We couldn't confirm a live check. You have 2 tries left today.")).toBeInTheDocument();
  });

  it("LIVENESS_ATTEMPTS_EXHAUSTED shows the daily-limit copy", async () => {
    api.startRunnerLiveness.mockRejectedValueOnce(apiError("LIVENESS_ATTEMPTS_EXHAUSTED", 429));
    renderForm();
    await pickReason();
    await tick();
    fireEvent.click(await startButton());
    expect(await screen.findByText(/You've used today's checks. Try again tomorrow or email us./)).toBeInTheDocument();
    expect(emailRoute()).not.toBeNull();
  });

  it("an old consent version asks for a reload", async () => {
    api.startRunnerLiveness.mockRejectedValueOnce(apiError("CONSENT_VERSION_MISMATCH", 400));
    renderForm();
    await pickReason();
    await tick();
    fireEvent.click(await startButton());
    expect(await screen.findByText("Our privacy notice changed. Reload the page and tick again.")).toBeInTheDocument();
  });

  it.each(["AI_API_UNAVAILABLE", "REMOVAL_INTAKE_CLOSED", "FACE_SEARCH_ENDED", "SOMETHING_NEW"])(
    "%s sends the runner to the email route", async (code) => {
      api.submitRunnerRemoval.mockRejectedValueOnce(apiError(code, 503));
      renderForm();
      await start();
      await detector.props!.onAnalysisComplete();
      await waitFor(() => expect(emailRoute()).not.toBeNull());
      expect(screen.queryByRole("button", { name: "Start live check" })).toBeNull();
    });

  it("a camera error shows the email route", async () => {
    renderForm();
    await start();
    detector.props!.onError!({ state: "CAMERA_ACCESS_ERROR", error: new Error("denied") });
    expect(await screen.findByText(/Allow camera access/)).toBeInTheDocument();
    expect(emailRoute()).not.toBeNull();
  });

  it("a cancelled check offers another try and the email route", async () => {
    renderForm();
    await start();
    detector.props!.onUserCancel!();
    expect(await screen.findByText(/The live check didn't finish/)).toBeInTheDocument();
    expect(await startButton()).toBeInTheDocument();
    expect(emailRoute()).not.toBeNull();
  });

  it("an open request links to it", async () => {
    api.getRunnerRemovalEligibility.mockResolvedValue({ ...ELIGIBLE, eligible: false, blockedReason: "REQUEST_OPEN", openRequestId: "req-9" });
    renderForm();
    const link = await screen.findByRole("link", { name: /Open your request/ });
    expect(link).toHaveAttribute("href", "/account/photo-deletion/req-9");
  });

  it("a blocked event shows the email route, not the camera", async () => {
    api.getRunnerRemovalEligibility.mockResolvedValue({ ...ELIGIBLE, eligible: false, blockedReason: "FACE_DATA_EXPIRED" });
    renderForm();
    await waitFor(() => expect(emailRoute()).not.toBeNull());
    expect(screen.queryByRole("button", { name: "Start live check" })).toBeNull();
    expect(screen.queryByRole("checkbox")).toBeNull();
  });

  it("an unverified email is a note, not a block", async () => {
    api.getRunnerRemovalEligibility.mockResolvedValue({ ...ELIGIBLE, emailVerified: false });
    renderForm();
    expect(await screen.findByText(/Verify your email so matching photos can be hidden at once/)).toBeInTheDocument();
    expect(await startButton()).toBeInTheDocument();
  });

  it("a photographer sees Runner accounts only and the email route", async () => {
    login("PHOTOGRAPHER");
    renderForm();
    expect(await screen.findByText("Runner accounts only")).toBeInTheDocument();
    expect(emailRoute()).not.toBeNull();
    expect(api.getRunnerRemovalEligibility).not.toHaveBeenCalled();
  });

  it("shows who is signed in before anything else", async () => {
    renderForm();
    expect(await screen.findByText(/Signed in as/)).toHaveTextContent("a@example.test");
    expect(screen.getByRole("button", { name: "Sign out" })).toBeInTheDocument();
  });

  it("the event search includes listed photographer events", async () => {
    renderForm(null);
    fireEvent.change(await screen.findByLabelText("Find your event"), { target: { value: "Cebu" } });
    const option = await screen.findByRole("button", { name: /Cebu Marathon/ });
    expect(events.filters.some((f) => (f as { includeListed?: boolean }).includeListed === true)).toBe(true);
    fireEvent.click(option);
    await waitFor(() => expect(api.getRunnerRemovalEligibility).toHaveBeenCalledWith("ev-1", expect.anything()));
  });

  it("details are optional and capped at 500 characters", async () => {
    renderForm();
    const box = await screen.findByLabelText(/Details/);
    expect(box).toHaveAttribute("maxLength", "500");
    fireEvent.change(box, { target: { value: "abc" } });
    expect(within(box.parentElement!).getByText("3/500")).toBeInTheDocument();
  });
});
