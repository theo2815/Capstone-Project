import { api } from "@/lib/api";

// Client half of POST /me/reauth — exchange the current password for a
// short-lived step-up token, sent as X-Step-Up on sensitive payout actions.
// The api client calls this transparently (via the registered step-up handler)
// when a request 403s with STEP_UP_REQUIRED.
export async function reauthForStepUp(password: string): Promise<string> {
  const res = await api.post<{ stepUpToken: string }>("/me/reauth", {
    password,
  });
  return res.stepUpToken;
}
