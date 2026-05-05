import { WireframePage } from "../_components/WireframePage";
import { BrowserFrame } from "../_components/Frames";
import { Annot, Box, Btn, Caption, Field, Placeholder, Tag } from "../_components/primitives";

export default function M31() {
  return (
    <WireframePage
      module="M3"
      ucId="UC-M3-3.1"
      title="Register / Login"
      tracesTo="(supports GO3) · RA 10173"
      mustShow={[
        "Login form",
        "Sign-up form with consent dialog and selfie-capture step (runner only)",
        "Forgot-password flow",
        "Rate-limit error state",
        "Success-redirect to events list",
      ]}
    >
      <BrowserFrame url="quickpitik.ph/auth">
        <div className="grid grid-cols-12 gap-0">
          {/* login pane */}
          <div className="col-span-4 border-r border-neutral-200 p-6">
            <Caption>Pane A · login</Caption>
            <h3 className="mt-2 font-display text-xl font-semibold">Welcome back.</h3>
            <p className="mt-1 text-[11px] text-neutral-600">
              POST /v1/auth/login · HTTPS · bcrypt-verified
            </p>
            <div className="mt-4 space-y-3">
              <Field label="Email" placeholder="runner@example.com" />
              <Field label="Password" value="••••••••" />
              <Btn primary>Log in</Btn>
              <div className="flex items-center justify-between text-[11px]">
                <a className="text-neutral-700 underline">Forgot password (A2)</a>
                <a className="text-neutral-700">No account? Sign up →</a>
              </div>
            </div>

            <Box className="mt-5 !bg-rose-50 !border-rose-700">
              <Caption>E1 · wrong credentials</Caption>
              <div className="mt-1 text-[11px] text-neutral-800">
                Incorrect email or password. <span className="text-neutral-500">(generic — no account-existence disclosure)</span>
              </div>
            </Box>

            <Box className="mt-2 !bg-amber-50 !border-amber-700">
              <Caption>E2 · rate limit</Caption>
              <div className="mt-1 text-[11px] text-neutral-800">
                5 failed attempts · account locked for 15 min · HTTP 429.
              </div>
            </Box>
          </div>

          {/* sign-up pane */}
          <div className="col-span-5 p-6">
            <Caption>Pane B · sign-up (A1)</Caption>
            <h3 className="mt-2 font-display text-xl font-semibold">Create your QuickPitik account.</h3>
            <div className="mt-4 grid grid-cols-2 gap-3">
              <Field label="Full name" placeholder="Maria Dela Cruz" />
              <Field label="Email" placeholder="maria@example.com" />
              <Field label="Password" value="••••••••" />
              <Field label="Confirm" value="••••••••" />
            </div>

            <div className="mt-4">
              <Caption>Role</Caption>
              <div className="mt-1 flex gap-2">
                <span className="inline-flex items-center gap-1.5 border border-neutral-900 bg-neutral-900 px-2 py-1 text-[11px] text-white">
                  <span className="h-2 w-2 rounded-full bg-white" /> Runner
                </span>
                <span className="inline-flex items-center gap-1.5 border border-neutral-300 px-2 py-1 text-[11px]">
                  <span className="h-2 w-2 rounded-full border border-neutral-700" /> Photographer
                </span>
              </div>
            </div>

            <Box className="mt-4 !bg-neutral-50">
              <div className="flex items-center justify-between">
                <Caption>RA 10173 consent · biometric processing</Caption>
                <Tag tone="warn">required for selfie search</Tag>
              </div>
              <p className="mt-1 text-[11px] text-neutral-700">
                I consent to QuickPitik processing my facial embedding for the
                purpose of finding my own photos. I may revoke at any time.
              </p>
              <div className="mt-2 flex items-center gap-2">
                <span className="inline-block h-3 w-3 border border-neutral-900 bg-neutral-900" />
                <span className="text-[11px] text-neutral-800">I agree</span>
              </div>
            </Box>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <Box dashed>
                <Caption>Selfie capture</Caption>
                <Placeholder label="CAMERA PREVIEW" height="h-24" />
                <div className="mt-2 flex gap-2">
                  <Btn small primary>Capture</Btn>
                  <Btn small>Upload</Btn>
                </div>
              </Box>
              <Box>
                <Caption>Enrolment status</Caption>
                <div className="mt-1 space-y-1 text-[11px] text-neutral-700">
                  <div>· Encrypted in transit (HTTPS)</div>
                  <div>· Embedding stored, not photo</div>
                  <div>· Tagged event_id + api_key_id</div>
                </div>
                <Tag tone="info">ai-api · /faces/enroll</Tag>
              </Box>
            </div>

            <div className="mt-4 flex gap-2">
              <Btn primary>Create account</Btn>
              <Btn>Cancel</Btn>
            </div>
          </div>

          {/* annotations + post-success + reset */}
          <div className="col-span-3 border-l border-neutral-200 p-6 bg-neutral-50">
            <Caption>Annotations</Caption>
            <div className="mt-3 space-y-3">
              <Annot n={1}>HTTPS-only password transit · server stores bcrypt hash.</Annot>
              <Annot n={2}>RA 10173 consent dialog gates face enrolment (runner only).</Annot>
              <Annot n={3}>Selfie capture → ai-api /faces/enroll via backend.</Annot>
              <Annot n={4}>Forgot password (A2) — reset link expires after 1 h.</Annot>
              <Annot n={5}>Post-success redirect → events list (UC-M3-3.2).</Annot>
            </div>

            <Box className="mt-5">
              <Caption>A2 · forgot password</Caption>
              <Field label="Email" placeholder="you@example.com" className="!mt-2" />
              <div className="mt-2"><Btn small primary>Send reset link</Btn></div>
            </Box>

            <Box className="mt-4 !bg-emerald-50 !border-emerald-700">
              <Caption>Success → /events</Caption>
              <div className="mt-1 text-[11px] text-neutral-800">
                JWT in HTTP-only cookie · session active.
              </div>
            </Box>
          </div>
        </div>
      </BrowserFrame>
    </WireframePage>
  );
}
