import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { act, cleanup, render, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { api, setStepUpHandler } from "@/lib/api";
import { setTokens, clearTokens } from "@/lib/auth";
import { StepUpProvider } from "./step-up-provider";
import { useAuth } from "@/hooks/use-auth";
import { useAuthStore } from "@/store/auth-store";
import { useAdminPayouts, usePayoutActions } from "@/hooks/use-admin-payouts";
import { admin, payoutCycle } from "@/test/fixtures/admin";

const mocks = vi.hoisted(() => ({
  resolve: null as null | ((token: string | null) => void),
  open: false,
  status: "pending_review",
}));
vi.mock("@/lib/auth", () => ({ getAccessToken: () => "access", getRefreshToken: () => "refresh", setTokens: vi.fn(), clearTokens: vi.fn() }));
vi.mock("@/lib/auth-reset", () => ({ captureGuestBuffer: () => () => {}, resetUserScopedStores: vi.fn() }));
vi.mock("@/lib/api-admin", () => ({
  fetchAdminPayouts: vi.fn(async () => [{ ...payoutCycle, id: "PAY-TEST", status: mocks.status, approvedBy: "admin-a" }]),
  approvePayout: vi.fn(async () => ({ ...payoutCycle, id: "PAY-TEST", approvedBy: "admin-a" })),
  holdPayout: vi.fn(), markPayoutPaid: vi.fn(), rejectPayout: vi.fn(),
  markPayoutProcessing: vi.fn(), markPayoutFailed: vi.fn(), markPayoutReturned: vi.fn(),
  bulkApprovePayouts: vi.fn(async () => ({ groupId: "group", results: [{ id: "PAY-TEST", ok: true }, { id: "missing", ok: false }] })),
  bulkHoldPayouts: vi.fn(),
}));
vi.mock("./step-up-modal", () => ({ StepUpModal: (props: { open: boolean; onResolve: (token: string | null) => void }) => {
  mocks.resolve = props.onResolve;
  mocks.open = props.open;
  return null;
} }));

beforeEach(() => {
  vi.clearAllMocks();
  useAuthStore.getState().logout();
  mocks.status = "pending_review";
  mocks.open = false;
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); setStepUpHandler(null); });

function response(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), { status });
}
function queryWrapper(client: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

test.each(["/auth/mfa/verify", "/admin/mfa/confirm", "/admin/mfa/disable"])("wrong code at %s stays an inline error without refreshing or logging out", async (path) => {
  const fetch = vi.fn(async () => response({ success: false, errors: [{ code: "INVALID_MFA_CODE", message: "That code is not valid." }] }, 401));
  vi.stubGlobal("fetch", fetch);
  await expect(api.post(path, { code: "invalid" })).rejects.toThrow("That code is not valid.");
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(clearTokens).not.toHaveBeenCalled();
});

test("first approval stays pending and subsequent provider state is visible", async () => {
  useAuthStore.getState().setUser(admin);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const { result } = renderHook(() => ({ payouts: useAdminPayouts(), actions: usePayoutActions() }), { wrapper: queryWrapper(client) });
  await waitFor(() => expect(result.current.payouts.isSuccess).toBe(true));
  await act(async () => { await result.current.actions.approve("PAY-TEST"); });
  expect(result.current.payouts.data?.[0].status).toBe("pending_review");
  expect(result.current.payouts.data?.[0].approvedBy).toBe(admin.id);
  expect(result.current.payouts.data?.[0].secondApprovedBy).toBeNull();
  mocks.status = "paid";
  await act(async () => { await client.invalidateQueries({ queryKey: ["admin", "payouts"] }); });
  await waitFor(() => expect(result.current.payouts.data?.[0].status).toBe("paid"));
  await act(async () => { expect(await result.current.actions.bulkApprove(["PAY-TEST", "missing"])).toBe(false); });
  expect(result.current.payouts.data?.[0].status).toBe("paid");
  client.clear();
});

test.each(["success", "cancel", "unmount"])("simultaneous step-up requests all settle on %s", async (outcome) => {
  const view = render(<StepUpProvider />);
  vi.stubGlobal("fetch", vi.fn(async (_url: string, opts: RequestInit) => {
    const proof = (opts.headers as Record<string, string>)["X-Step-Up"];
    return proof ? response({ success: true, data: { saved: true } }) : response({ success: false, errors: [{ code: "STEP_UP_REQUIRED", message: "Reauthenticate" }] }, 403);
  }));
  const settled = Promise.allSettled([api.post("/me/photographer/payouts/a", {}), api.post("/me/photographer/payouts/b", {})]);
  await waitFor(() => expect(mocks.open).toBe(true));
  await act(async () => {
    if (outcome === "unmount") view.unmount();
    else mocks.resolve?.(outcome === "success" ? "proof" : null);
  });
  expect((await settled).map((r) => r.status)).toEqual(outcome === "success" ? ["fulfilled", "fulfilled"] : ["rejected", "rejected"]);
});

test("Google MFA withholds the session until the code is verified", async () => {
  const user = { id: "admin-a", role: "ADMIN", name: "Admin", email: "admin@test.local" };
  vi.stubGlobal("fetch", vi.fn(async (url: string) => response({ success: true, data: url.endsWith("/auth/google")
    ? { mfaRequired: true, mfaToken: "challenge", accessToken: "", refreshToken: "", user }
    : { accessToken: "verified-access", refreshToken: "verified-refresh", user } })));
  const client = new QueryClient();
  const { result } = renderHook(useAuth, { wrapper: queryWrapper(client) });
  await act(async () => { expect((await result.current.googleLogin("google-token"))?.kind).toBe("mfa"); });
  expect(useAuthStore.getState().pendingMfa).toBe("challenge");
  expect(setTokens).not.toHaveBeenCalled();
  expect(useAuthStore.getState().isAuthenticated).toBe(false);
  await act(async () => { await result.current.verifyMfa("challenge", "123456"); });
  expect(setTokens).toHaveBeenCalledWith("verified-access", "verified-refresh");
  expect(useAuthStore.getState().pendingMfa).toBeNull();
  client.clear();
});
