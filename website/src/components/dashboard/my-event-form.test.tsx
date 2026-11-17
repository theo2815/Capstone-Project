import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { raceOutcomeMessage } from "@/lib/photographer-events";
import { eventRegions } from "@/test/fixtures/admin";
import { MyEventForm } from "./my-event-form";

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

const TYPES = [
  { code: "ROAD_RACE", label: "Road race", hint: "5K to 42K", shared: true },
  { code: "OTHER", label: "Other", hint: "Your own event", shared: false },
];

const candidate = (over: Record<string, unknown> = {}) => ({
  id: "race-1", slug: "cebu-marathon-2027", name: "Cebu Marathon 2027", date: "2027-01-10",
  location: "IT Park, Cebu City", regionCode: "region-vii", provinceCode: "cebu", organizerName: "",
  eventType: "ROAD_RACE", bannerUrl: null, reviewStatus: "approved", strength: "strong", ...over,
});

function mount(selfServeEnabled = true) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(["regions"], eventRegions);
  client.setQueryData(["race-types"], { selfServeEnabled, types: TYPES });
  const onDone = vi.fn(); const onRaceDone = vi.fn();
  render(<QueryClientProvider client={client}><MyEventForm onDone={onDone} onCancel={vi.fn()} onRaceDone={onRaceDone} /></QueryClientProvider>);
  return { onDone, onRaceDone };
}

function fill() {
  fireEvent.change(screen.getByLabelText("Title"), { target: { value: "Cebu Marathon 2027" } });
  fireEvent.change(screen.getByLabelText("Location · Venue, City"), { target: { value: "IT Park, Cebu City" } });
  fireEvent.change(screen.getByLabelText("Date"), { target: { value: "2027-01-10" } });
}

async function pick(field: string, option: string) {
  await userEvent.click(screen.getByRole("button", { name: new RegExp(`^${field}:`) }));
  await userEvent.click(screen.getByRole("menuitem", { name: option }));
}

async function raceReady() {
  const view = mount(); fill();
  fireEvent.click(screen.getByRole("radio", { name: /^Road race/ }));
  await pick("Region", eventRegions[0].name);
  await pick("Province", "Cebu");
  return view;
}

const submit = () => fireEvent.click(screen.getByRole("button", { name: "Create race" }));

it("race type requires region and province", async () => {
  mount(); fill();
  fireEvent.click(screen.getByRole("radio", { name: /^Road race/ }));
  expect(screen.queryByLabelText("Price per photo · PHP")).toBeNull();
  expect(screen.getByRole("button", { name: "Create race" })).toBeDisabled();
  await pick("Region", eventRegions[0].name);
  expect(screen.getByRole("button", { name: "Create race" })).toBeDisabled();
  await pick("Province", "Cebu");
  expect(screen.getByRole("button", { name: "Create race" })).toBeEnabled();
});

it("Other keeps the pricing trio", () => {
  mount(); fill();
  fireEvent.click(screen.getByRole("radio", { name: /^Other/ }));
  expect(screen.getByRole("radio", { name: /^Paid/ })).toBeChecked();
  expect(screen.getByLabelText("Price per photo · PHP")).toBeInTheDocument();
  expect(screen.getByRole("radio", { name: /^Public/ })).toBeChecked();
  expect(screen.getByRole("button", { name: "Submit for review" })).toBeInTheDocument();
});

it("switch off shows only today's form", async () => {
  const post = vi.spyOn(api, "post").mockResolvedValue({ id: "e1" });
  const { onDone } = mount(false); fill();
  expect(screen.queryByRole("radio", { name: /^Road race/ })).toBeNull();
  expect(screen.queryByRole("button", { name: /^Region:/ })).toBeNull();
  fireEvent.change(screen.getByLabelText("Price per photo · PHP"), { target: { value: "150" } });
  fireEvent.click(screen.getByRole("button", { name: "Submit for review" }));
  await waitFor(() => expect(onDone).toHaveBeenCalled());
  expect(post).toHaveBeenCalledWith("/me/photographer/events", expect.any(FormData));
});

it("match answer offers Join and Mine is different", async () => {
  vi.spyOn(api, "post").mockResolvedValue({ outcome: "match", candidates: [candidate()], checkReasons: [] });
  await raceReady(); submit();
  expect(await screen.findByRole("button", { name: "Join Cebu Marathon 2027" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Mine is different" })).toBeInTheDocument();
});

it("mine is different on a match sends confirmDifferentFrom and shows sent for review", async () => {
  const pending = { outcome: "pending_review" as const, event: null, candidates: [], checkReasons: ["POSSIBLE_DUPLICATE"] };
  const post = vi.spyOn(api, "post")
    .mockResolvedValueOnce({ outcome: "match", candidates: [candidate()], checkReasons: [] })
    .mockResolvedValueOnce(pending);
  const { onRaceDone } = await raceReady(); submit();
  fireEvent.click(await screen.findByRole("button", { name: "Mine is different" }));
  await waitFor(() => expect(onRaceDone).toHaveBeenCalledWith(pending));
  const form = post.mock.calls[1][1] as FormData;
  expect(post.mock.calls[1][0]).toBe("/me/photographer/races");
  expect(form.get("confirmDifferentFrom")).toBe("race-1");
  expect(form.get("eventType")).toBe("ROAD_RACE");
  expect(form.get("provinceCode")).toBe("cebu");
  expect(raceOutcomeMessage(pending)).toMatch(/^Sent for review\. Uploads open once an admin approves it\./);
  expect(raceOutcomeMessage(pending)).toMatch(/same race as one already listed/);
});

it("candidates answer offers Join and Mine is different", async () => {
  vi.spyOn(api, "post").mockResolvedValue({
    outcome: "candidates", checkReasons: [],
    candidates: [candidate({ strength: "medium" }), candidate({ id: "race-3", name: "Cebu City Marathon", strength: "medium" })],
  });
  await raceReady(); submit();
  expect(await screen.findByRole("button", { name: "Join Cebu Marathon 2027" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Join Cebu City Marathon" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Mine is different" })).toBeInTheDocument();
});

it("a pending candidate says it waits for an admin", async () => {
  vi.spyOn(api, "post").mockResolvedValue({ outcome: "match", candidates: [candidate({ reviewStatus: "pending" })], checkReasons: [] });
  await raceReady(); submit();
  expect(await screen.findByText(/Waiting for an admin: you can join now, and uploads open once an admin approves it/)).toBeInTheDocument();
  const joined = { outcome: "joined", event: { id: "race-1", name: "Cebu Marathon 2027", reviewStatus: "pending" }, candidates: [], checkReasons: [] };
  expect(raceOutcomeMessage(joined as never)).toBe("Waiting for an admin: you can join now, and uploads open once an admin approves it.");
});

it("Join failure keeps the form input", async () => {
  vi.spyOn(api, "post")
    .mockResolvedValueOnce({ outcome: "match", candidates: [candidate()], checkReasons: [] })
    .mockRejectedValueOnce(new ApiError([{ code: "EVENT_NOT_JOINABLE", message: "This event can't be joined." }], 422));
  const { onRaceDone } = await raceReady(); submit();
  fireEvent.click(await screen.findByRole("button", { name: "Join Cebu Marathon 2027" }));
  expect(await screen.findByText("This event can't be joined.")).toBeInTheDocument();
  expect(onRaceDone).not.toHaveBeenCalled();
  expect(screen.getByLabelText("Title")).toHaveValue("Cebu Marathon 2027");
});
