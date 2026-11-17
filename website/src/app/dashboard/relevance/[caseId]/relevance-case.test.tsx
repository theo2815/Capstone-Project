import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import type { PhotographerRelevanceCase } from "@/lib/api-photographer";
import { RelevanceCase } from "./relevance-case";

const owner = { id: "owner-a", role: "PHOTOGRAPHER" as const, name: "Owner A", email: "a@example.test", createdAt: "2026-01-01T00:00:00Z" };
const base: PhotographerRelevanceCase = {
  caseId: "case-1", eventId: "event-1", eventName: "Cebu Marathon", verdict: "JUNK_DUMP", action: "HIDE_ALL", current: true,
  batchPhotos: 400, sampleSize: 300, junkCount: 240, hiddenPhotos: 400, scanMeasured: null, scanBlurry: null,
  noticeAt: "2026-11-10T03:00:00Z", appealDeadline: "2026-11-24", appeal: null, eventDate: "2026-11-08", closeReason: null,
};
let client: QueryClient;
let serverCase: PhotographerRelevanceCase;

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(new Date("2026-11-16T04:00:00Z"));
  useAuthStore.getState().startSession(owner);
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  serverCase = base;
  vi.spyOn(api, "get").mockImplementation(async () => serverCase);
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); vi.useRealTimers(); });

const renderCase = () => render(<QueryClientProvider client={client}><RelevanceCase caseId="case-1" /></QueryClientProvider>);
const reasonField = () => screen.getByLabelText("Why do you think this decision is wrong?");
const sendButton = () => screen.getByRole("button", { name: "Send appeal" });

it("shows the approved reason, the numbers and the appeal form with its deadline", async () => {
  renderCase();
  expect(await screen.findByRole("heading", { name: "Your photos in Cebu Marathon are off sale" })).toBeVisible();
  expect(screen.getByText(/We checked 300 of your 400 photos in Cebu Marathon, and 240 of them look blurry/)).toBeVisible();
  expect(screen.getByText(/You can appeal until NOV 24 · 2026, 14 days from our message\./)).toBeVisible();
  expect(api.get).toHaveBeenCalledWith("/me/photographer/relevance/cases/case-1", expect.anything());
});

it("the appeal form refuses an empty or over-long reason before sending", async () => {
  const post = vi.spyOn(api, "post");
  renderCase();
  await screen.findByRole("heading", { name: /off sale/ });
  expect(sendButton()).toBeDisabled();
  fireEvent.change(reasonField(), { target: { value: "   " } });
  expect(sendButton()).toBeDisabled();
  fireEvent.change(reasonField(), { target: { value: "x".repeat(1001) } });
  expect(screen.getByText("1,001/1,000")).toBeVisible();
  expect(reasonField()).toHaveAttribute("aria-invalid", "true");
  expect(sendButton()).toBeDisabled();
  fireEvent.click(sendButton());
  expect(post).not.toHaveBeenCalled();
});

it("a filed appeal shows its status and hides the form", async () => {
  vi.useRealTimers();
  const user = userEvent.setup();
  const filed = { ...base, appeal: { id: "appeal-1", status: "OPEN" as const, createdAt: "2026-11-16T04:00:00Z", decidedAt: null } };
  const post = vi.spyOn(api, "post").mockImplementation(async () => { serverCase = filed; return filed; });
  renderCase();
  await screen.findByRole("heading", { name: /off sale/ });
  await user.type(reasonField(), "  These are my race photos.  ");
  await user.click(sendButton());
  expect(post).toHaveBeenCalledWith("/me/photographer/relevance/cases/case-1/appeal", { reason: "These are my race photos." });
  expect(await screen.findByText("Appeal received")).toBeVisible();
  expect(screen.getByText(/Until then, the photos stay off sale\./)).toBeVisible();
  expect(screen.queryByLabelText("Why do you think this decision is wrong?")).not.toBeInTheDocument();
});

it("a refused appeal shows the server's message and keeps the reason", async () => {
  vi.useRealTimers();
  const user = userEvent.setup();
  vi.spyOn(api, "post").mockRejectedValue(new ApiError([{ code: "APPEAL_ALREADY_OPEN",
    message: "You already have an open appeal for your photos in this event. We will tell you in your inbox when it is decided." }], 409));
  renderCase();
  await screen.findByRole("heading", { name: /off sale/ });
  await user.type(reasonField(), "Mine");
  await user.click(sendButton());
  expect(await screen.findByText(/You already have an open appeal for your photos in this event\./)).toBeVisible();
  expect(reasonField()).toHaveValue("Mine");
});

it("the form is gone after the deadline", async () => {
  vi.setSystemTime(new Date("2026-11-24T16:30:00Z")); // Nov 25, 00:30 in the Philippines
  renderCase();
  expect(await screen.findByText("The time to appeal this decision has ended.")).toBeVisible();
  expect(screen.queryByLabelText("Why do you think this decision is wrong?")).not.toBeInTheDocument();
});

it("the form stays open through the last day in Philippine time", async () => {
  vi.setSystemTime(new Date("2026-11-24T15:30:00Z")); // Nov 24, 23:30 in the Philippines
  renderCase();
  expect(await screen.findByLabelText("Why do you think this decision is wrong?")).toBeVisible();
});

it("some photos hidden shows the Mixed text", async () => {
  serverCase = { ...base, verdict: "MIXED", action: "HIDE_JUNK", junkCount: 60, hiddenPhotos: 70 };
  renderCase();
  expect(await screen.findByRole("heading", { name: "Some of your photos are off sale" })).toBeVisible();
  expect(screen.getByText(/Your other photos stay on sale\./)).toBeVisible();
});

it("photos shown again after a won appeal", async () => {
  serverCase = { ...base, action: "HIDE_JUNK", appeal: { id: "appeal-1", status: "WON", createdAt: "2026-11-12T00:00:00Z", decidedAt: "2026-11-13T00:00:00Z" } };
  renderCase();
  expect(await screen.findByRole("heading", { name: "Appeal accepted" })).toBeVisible();
  expect(screen.getByText(/We are now checking every photo and putting the good ones back on sale\./)).toBeVisible();
  expect(screen.queryByLabelText("Why do you think this decision is wrong?")).not.toBeInTheDocument();
});

it("photos shown again after a full check", async () => {
  serverCase = { ...base, verdict: "FITS", action: "NONE", hiddenPhotos: 0 };
  renderCase();
  expect(await screen.findByRole("heading", { name: "Photos back on sale" })).toBeVisible();
  expect(screen.queryByLabelText("Why do you think this decision is wrong?")).not.toBeInTheDocument();
});

it("an unknown case shows not found", async () => {
  vi.mocked(api.get).mockRejectedValue(new ApiError([{ code: "NOT_FOUND", message: "Case not found" }], 404));
  renderCase();
  expect(await screen.findByText("Case not found")).toBeVisible();
});

it("the page never shows the model's reason", async () => {
  serverCase = { ...base, adminReason: "MODEL REASON: dump of lens-cap shots", summarySent: "SUMMARY SENT" } as PhotographerRelevanceCase;
  renderCase();
  await screen.findByRole("heading", { name: /off sale/ });
  await waitFor(() => expect(document.body.textContent).not.toMatch(/MODEL REASON|SUMMARY SENT/));
});

it("a second click while the appeal is sending does nothing, and nothing changes before the server answers", async () => {
  let answer!: (value: PhotographerRelevanceCase) => void;
  const filed = { ...base, appeal: { id: "appeal-1", status: "OPEN" as const, createdAt: "2026-11-16T04:00:00Z", decidedAt: null } };
  const post = vi.spyOn(api, "post").mockImplementation(() => new Promise((resolve) => { answer = resolve as typeof answer; }));
  renderCase();
  await screen.findByRole("heading", { name: /off sale/ });
  fireEvent.change(reasonField(), { target: { value: "Mine" } });
  act(() => { sendButton().click(); sendButton().click(); });
  await waitFor(() => expect(post).toHaveBeenCalled());
  await act(async () => {}); // let a second send, if any, reach the network
  expect(post).toHaveBeenCalledTimes(1);
  expect(screen.queryByText("Appeal received")).not.toBeInTheDocument();
  await waitFor(() => expect(screen.getByRole("button", { name: "Sending…" })).toBeDisabled());
  serverCase = filed;
  await act(async () => answer(filed));
  expect(await screen.findByText("Appeal received")).toBeVisible();
});

it("wrong event names the event's date, and shows no rule sentence without it", async () => {
  serverCase = { ...base, verdict: "WRONG_EVENT" };
  renderCase();
  expect(await screen.findByText(/^Many of your 400 photos in Cebu Marathon seem to be from a different event: their camera dates are far from the event's date \(NOV 08 · 2026\)/)).toBeVisible();
  cleanup(); client.clear();
  serverCase = { ...base, verdict: "WRONG_EVENT", eventDate: null };
  renderCase();
  expect(await screen.findByRole("heading", { name: "Your photos in Cebu Marathon are off sale" })).toBeVisible();
  expect(document.body.textContent).not.toMatch(/So our system/);
});

it("a Fits case a re-check lifted to junk only never says all photos are off sale", async () => {
  serverCase = { ...base, verdict: "FITS", action: "HIDE_JUNK", noticeAt: null, appealDeadline: null };
  renderCase();
  expect(await screen.findByRole("heading", { name: "Photos back on sale" })).toBeVisible();
  expect(screen.getByText(/except those that still look blurry, very dark or not of a race\./)).toBeVisible();
  expect(document.body.textContent).not.toMatch(/all your photos/);
});

it("a case a re-check lifted shows the note its close sent", async () => {
  serverCase = { ...base, current: false, closeReason: "RECHECK_LIFTED" };
  renderCase();
  expect(await screen.findByRole("heading", { name: "Photos back on sale" })).toBeVisible();
  expect(screen.getByText(/^We checked your photos in Cebu Marathon again after your new uploads\./)).toBeVisible();
  expect(screen.queryByText("Closed")).not.toBeInTheDocument();
});

it("a case closed after a won appeal shows the appeal accepted, not Closed", async () => {
  serverCase = { ...base, current: false, action: "HIDE_JUNK", closeReason: "APPEAL_WON",
    appeal: { id: "appeal-1", status: "WON", createdAt: "2026-11-12T00:00:00Z", decidedAt: "2026-11-13T00:00:00Z" } };
  renderCase();
  expect(await screen.findByRole("heading", { name: "Appeal accepted" })).toBeVisible();
  expect(screen.queryByText("Closed")).not.toBeInTheDocument();
});

it("a case closed for a reason with no approved text shows only its status word", async () => {
  serverCase = { ...base, current: false, closeReason: "SUPERSEDED",
    appeal: { id: "appeal-1", status: "OPEN", createdAt: "2026-11-12T00:00:00Z", decidedAt: null } };
  renderCase();
  expect(await screen.findByRole("heading", { name: "Closed" })).toBeVisible();
  expect(screen.queryByText("Appeal received")).not.toBeInTheDocument();
  expect(document.body.textContent).not.toMatch(/off sale/);
});

it("a Fits case shows the blurry text only when its check found blurry photos", async () => {
  serverCase = { ...base, verdict: "FITS", action: "NONE", noticeAt: null, appealDeadline: null, scanMeasured: 400, scanBlurry: 3 };
  renderCase();
  expect(await screen.findByRole("heading", { name: "3 photos look blurry" })).toBeVisible();
  cleanup(); client.clear();
  serverCase = { ...base, verdict: "FITS", action: "NONE", noticeAt: null, appealDeadline: null };
  renderCase();
  expect(await screen.findByText("Case not found")).toBeVisible();
  expect(document.body.textContent).not.toMatch(/blurry/);
});

it("shows no off-sale count, which the server stamps once and does not keep up to date", async () => {
  serverCase = { ...base, verdict: "MIXED", action: "HIDE_JUNK", hiddenPhotos: 0 };
  renderCase();
  await screen.findByRole("heading", { name: "Some of your photos are off sale" });
  expect(screen.queryByText("Off sale")).not.toBeInTheDocument();
});

// RVF-18 (RVR-25): W1, approved 2026-11-17.
const TOO_LONG = "Your reason is too long. Please keep it to 1,000 characters.";

it("an over-long reason shows why under the box and keeps Send off", async () => {
  renderCase();
  await screen.findByRole("heading", { name: /off sale/ });
  fireEvent.change(reasonField(), { target: { value: "x".repeat(1001) } });
  const message = screen.getByText(TOO_LONG);
  expect(message).toBeVisible();
  expect(reasonField()).toHaveAttribute("aria-invalid", "true");
  expect(reasonField().getAttribute("aria-describedby")).toContain(message.id);
  expect(sendButton()).toBeDisabled();
});

it("the message goes at 1,000 characters", async () => {
  renderCase();
  await screen.findByRole("heading", { name: /off sale/ });
  fireEvent.change(reasonField(), { target: { value: "x".repeat(1001) } });
  expect(screen.getByText(TOO_LONG)).toBeVisible();
  // The counter and the check count the same trimmed length: spaces around the reason do not count.
  fireEvent.change(reasonField(), { target: { value: `  ${"x".repeat(1000)}  ` } });
  expect(screen.queryByText(TOO_LONG)).not.toBeInTheDocument();
  expect(screen.getByText("1,000/1,000")).toBeVisible();
  expect(reasonField()).toHaveAttribute("aria-invalid", "false");
  expect(sendButton()).toBeEnabled();
});

it("a current case that hides nothing and was never hidden by a check reads as not found", async () => {
  serverCase = { ...base, verdict: "CANT_TELL", action: "NONE" };
  renderCase();
  expect(await screen.findByText("Case not found")).toBeVisible();
  expect(document.body.textContent).not.toMatch(/back on sale/);
});
