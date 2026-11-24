import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, onTestFinished, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import type { User } from "@/types/user";
import { adminEvent, eventRegions } from "@/test/fixtures/admin";
import { AdminEventFormModal } from "./admin-event-form-modal";

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

function mount(mode: "create" | "edit" = "edit", event = adminEvent, seeded = true) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  if (seeded) client.setQueryData(["regions"], eventRegions);
  const submit = vi.fn().mockResolvedValue(false); const close = vi.fn();
  const props = { open: true, mode, event, onClose: close, onSubmit: submit };
  const view = render(<QueryClientProvider client={client}><AdminEventFormModal {...props} /></QueryClientProvider>);
  return { client, submit, close, view, props, rerender: (extra: Partial<typeof props> & { pending?: boolean }) => view.rerender(<QueryClientProvider client={client}><AdminEventFormModal {...props} {...extra} /></QueryClientProvider>) };
}

function fillCreate() {
  fireEvent.change(screen.getByLabelText(/^Title/), { target: { value: "Race" } });
  fireEvent.change(screen.getByLabelText(/^Location/), { target: { value: "Venue" } });
  fireEvent.change(screen.getByLabelText(/^Date/), { target: { value: "2026-11-01" } });
  fireEvent.change(screen.getByLabelText(/^Price per photo · PHP/), { target: { value: "80" } });
}

async function pick(field: string, option: string) {
  await userEvent.click(screen.getByRole("button", { name: new RegExp(`^${field}:`) }));
  await userEvent.click(screen.getByRole("menuitem", { name: option }));
}

it("requires a complete catalog-valid pair on create and clears province when region changes", async () => {
  const { submit } = mount("create"); fillCreate();
  // AX-20: Create stays pressable and says what is missing instead of disabling.
  await userEvent.click(screen.getByRole("button", { name: "Create event" }));
  expect(screen.getByText("Choose a region before saving.")).toBeInTheDocument();
  await pick("Region", eventRegions[0].name);
  await userEvent.click(screen.getByRole("button", { name: "Create event" }));
  expect(screen.getByText("Choose a province before saving.")).toBeInTheDocument();
  await pick("Province", "Cebu");
  expect(screen.getByRole("button", { name: "Create event" })).toBeEnabled();
  await pick("Region", eventRegions[1].name);
  await userEvent.click(screen.getByRole("button", { name: "Create event" }));
  expect(submit).not.toHaveBeenCalled();
  await pick("Province", "Leyte");
  await userEvent.click(screen.getByRole("button", { name: "Create event" }));
  expect(submit).toHaveBeenCalledWith(expect.objectContaining({ regionCode: "region-viii", provinceCode: "leyte", location: "Venue" }));
});

it("allows an unrelated legacy-null edit but requires province after choosing region", async () => {
  const { submit } = mount();
  expect(screen.getByText(/unclassified/i)).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
  expect(submit).toHaveBeenCalledWith(expect.objectContaining({ regionCode: null, provinceCode: null }));
  await pick("Region", eventRegions[0].name);
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
  expect(submit).toHaveBeenCalledOnce();
  expect(screen.getByText("Choose a province before saving.")).toBeInTheDocument();
});

it("keeps saved codes through loading and error, retries, and identifies unknown codes", async () => {
  let reject!: (reason: Error) => void;
  vi.spyOn(api, "get").mockReturnValueOnce(new Promise((_resolve, no) => { reject = no; })).mockResolvedValue(eventRegions);
  const event = { ...adminEvent, regionCode: "retired", provinceCode: "old" };
  const { submit } = mount("edit", event, false);
  expect(screen.getByRole("button", { name: "Region: retired" })).toBeDisabled();
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
  expect(submit).toHaveBeenLastCalledWith(expect.objectContaining({ regionCode: "retired", provinceCode: "old" }));
  await act(async () => reject(new Error("Offline")));
  expect(screen.queryByText(/unrecognized/i)).not.toBeInTheDocument();
  await userEvent.click(await screen.findByRole("button", { name: "Retry regions" }));
  await screen.findByText(/unrecognized region/i);
  expect(screen.getByRole("button", { name: "Region: retired" })).toBeEnabled();
  await pick("Region", eventRegions[0].name);
  const calls = submit.mock.calls.length;
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
  expect(submit).toHaveBeenCalledTimes(calls);
  await pick("Province", "Cebu");
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
  expect(submit).toHaveBeenLastCalledWith(expect.objectContaining({ regionCode: "region-vii", provinceCode: "cebu" }));
});

it("preserves the dirty draft across refetch, resets on a fresh session, and owns Escape before the modal", async () => {
  const event = { ...adminEvent, regionCode: "region-vii", provinceCode: "cebu" };
  const { rerender, close, submit } = mount("edit", event);
  fireEvent.change(screen.getByLabelText(/^Title/), { target: { value: "Draft" } });
  await pick("Region", eventRegions[1].name); await pick("Province", "Leyte");
  rerender({ event: { ...event, name: "Remote", reviewRevision: 5 } });
  expect(screen.getByLabelText(/^Title/)).toHaveValue("Draft");
  expect(screen.getByRole("button", { name: "Province: Leyte" })).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: /^Province:/ }));
  expect(screen.getByRole("menu")).toHaveClass("z-[70]");
  await userEvent.keyboard("{Escape}"); expect(close).not.toHaveBeenCalled();
  expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  await userEvent.keyboard("{Escape}"); expect(close).toHaveBeenCalledOnce();
  await userEvent.click(screen.getByRole("button", { name: /^Region:/ }));
  rerender({ pending: true });
  expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: /^Region:/ })).toBeDisabled();
  expect(submit).not.toHaveBeenCalled();
  rerender({ open: false }); rerender({ open: true });
  await waitFor(() => expect(screen.getByLabelText(/^Title/)).toHaveValue("Cebu Race"));
  expect(within(screen.getByRole("dialog")).getByRole("button", { name: "Province: Cebu" })).toBeInTheDocument();
});

// AA-R16a: each photographer's price in a race, and the admin reset.
it("reset needs a confirm and refreshes the list after the API answers", async () => {
  const row = { photographerId: "p-1", name: "Paksit Photos", handle: "paksit", effectivePhp: 180, custom: true, priceChangedAt: "2026-11-10T02:00:00Z" };
  const list = { floorPhp: 125, ceilingPhp: 250, prices: [row], history: [] };
  vi.spyOn(api, "get").mockImplementation(async (path) =>
    path === "/admin/events/event-1/photographer-prices" ? list : { selfServeEnabled: false, types: [] });
  let answer!: (value: unknown) => void;
  const post = vi.spyOn(api, "post").mockReturnValue(new Promise((resolve) => { answer = resolve; }));
  useAuthStore.setState({ isAuthenticated: true, user: { id: "admin-1", role: "ADMIN" } as User });
  onTestFinished(() => useAuthStore.setState({ isAuthenticated: false, user: null }));
  mount("edit", { ...adminEvent, eventType: "ROAD_RACE" });
  expect(await screen.findByText("Paksit Photos")).toBeInTheDocument();
  expect(screen.getByText("₱180.00")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Reset Paksit Photos" }));
  expect(post).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("button", { name: "Confirm reset" }));
  expect(post).toHaveBeenCalledWith("/admin/events/event-1/photographer-prices/p-1/reset");
  expect(screen.getByText("₱180.00")).toBeInTheDocument();
  await act(async () => answer({ ...list, prices: [{ ...row, effectivePhp: 125, custom: false }] }));
  expect(await screen.findByText("₱125.00")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Reset Paksit Photos" })).not.toBeInTheDocument();
});

it("shows no price list on an untyped event", async () => {
  const get = vi.spyOn(api, "get").mockResolvedValue({ selfServeEnabled: false, types: [] });
  mount("edit");
  await screen.findByLabelText(/^Title/);
  expect(screen.queryByText("Photographer prices")).not.toBeInTheDocument();
  expect(get.mock.calls.some(([path]) => String(path).includes("photographer-prices"))).toBe(false);
});

// AA-R17b: a typed admin create checks for duplicates first.
it("duplicate warning needs a second confirm", async () => {
  const candidate = { id: "race-0", slug: "cebu-marathon", name: "Cebu Marathon", date: "2026-11-01", location: "SRP", regionCode: null, provinceCode: null, organizerName: "", eventType: "ROAD_RACE", bannerUrl: null, reviewStatus: "approved", strength: "strong" };
  const raceTypes = { selfServeEnabled: false, types: [
    { code: "ROAD_RACE", label: "Road race", hint: "5K to 42K", shared: true },
    { code: "OTHER", label: "Other", hint: "Not a race", shared: false },
  ] };
  const get = vi.spyOn(api, "get").mockImplementation(async (path) => path === "/race-types" ? raceTypes : [candidate]);
  const checks = () => get.mock.calls.filter(([path]) => String(path).startsWith("/admin/events/duplicate-candidates"));
  const { submit } = mount("create", adminEvent);
  await userEvent.click(await screen.findByRole("radio", { name: /^Road race/ }));
  expect(screen.queryByRole("radio", { name: /^Other/ })).not.toBeInTheDocument();
  fillCreate(); await pick("Region", eventRegions[0].name); await pick("Province", "Cebu");
  await userEvent.click(screen.getByRole("button", { name: "Create event" }));
  expect(checks()).toEqual([["/admin/events/duplicate-candidates?name=Race&date=2026-11-01&provinceCode=cebu"]]);
  expect(submit).not.toHaveBeenCalled();
  expect(await screen.findByText(/Cebu Marathon/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Create anyway" }));
  expect(submit).toHaveBeenCalledWith(expect.objectContaining({ eventType: "ROAD_RACE", name: "Race" }));
  expect(checks()).toHaveLength(1);
});

// AX-20 (EV4, X3): the form says what is missing, but only once it matters.
it("shows no error before the admin leaves a field or tries to submit", async () => {
  mount("create");
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(screen.queryByText(/Enter a price above zero/)).not.toBeInTheDocument();
  expect(screen.getByLabelText(/^Title/)).toBeRequired();
  expect(screen.getByLabelText(/^Date/)).toBeRequired();
  fireEvent.blur(screen.getByLabelText(/^Title/));
  expect(screen.getByText("Add a title.")).toBeInTheDocument();
  expect(screen.getByLabelText(/^Title/)).toHaveAttribute("aria-invalid", "true");
  expect(screen.queryByText("Add the place.")).not.toBeInTheDocument();
  fireEvent.blur(screen.getByLabelText(/^Price per photo · PHP/));
  expect(screen.getByText(/Enter a price above zero/)).toBeInTheDocument();
});

it("names every missing required field after a submit attempt", async () => {
  const { submit } = mount("edit");
  for (const label of [/^Title/, /^Location/, /^Date/]) fireEvent.change(screen.getByLabelText(label), { target: { value: "" } });
  expect(screen.queryByText("Add a title.")).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
  expect(screen.getByText("Add a title.")).toBeInTheDocument();
  expect(screen.getByText("Add the place.")).toBeInTheDocument();
  expect(screen.getByText("Pick the date.")).toBeInTheDocument();
  expect(screen.getByLabelText(/^Date/)).toHaveAttribute("aria-describedby", "event-date-error");
  expect(submit).not.toHaveBeenCalled();
});

it("cover controls are 44 px house buttons (EV4)", () => {
  mount("create");
  expect(screen.getByText("Upload cover").closest("label")).toHaveClass("rounded-full", "min-h-[44px]", "focus-within:outline-2");
});

it("a press that leaves a field shows its error after the press, not during it", async () => {
  mount("create");
  fireEvent.pointerDown(document.body);
  fireEvent.blur(screen.getByLabelText(/^Title/));
  // Shown mid-press, the new line would push the pressed button and lose the click.
  expect(screen.queryByText("Add a title.")).not.toBeInTheDocument();
  fireEvent.pointerUp(document.body);
  expect(await screen.findByText("Add a title.")).toBeInTheDocument();
});

it("says what a price or a date change does, in edit mode only (FT-15, B-30)", () => {
  mount("edit");
  expect(screen.getByText("Changing the price re-prices every photo already in this event. Orders already paid keep the price they were charged.")).toBeInTheDocument();
  expect(screen.getByText("The date decides which tab the event sits under and when uploads close.")).toBeInTheDocument();
  cleanup(); mount("create");
  expect(screen.queryByText(/re-prices every photo/)).not.toBeInTheDocument();
  expect(screen.queryByText("The date decides which tab the event sits under and when uploads close.")).not.toBeInTheDocument();
});

it("leaves the price line to the race price block on a shared race (FT-15)", () => {
  vi.spyOn(api, "get").mockReturnValue(new Promise(() => {}));
  mount("edit", { ...adminEvent, eventType: "ROAD_RACE" });
  expect(screen.queryByText(/re-prices every photo/)).not.toBeInTheDocument();
  expect(screen.getByText("The date decides which tab the event sits under and when uploads close.")).toBeInTheDocument();
});

it("an empty create shows the site's own error under every required field, with no browser bubble (FT-22, R-04)", async () => {
  const { submit } = mount("create");
  const form = document.body.querySelector("form");
  expect(form).toHaveAttribute("novalidate");
  expect(screen.getByLabelText(/^Price per photo · PHP/)).toBeRequired();
  expect(screen.getByText(/^Price per photo · PHP/, { selector: "label" })).toHaveTextContent("Price per photo · PHP (required)");
  expect(screen.getByText(/^Region/, { selector: "label" })).toHaveTextContent("Region (required)");
  expect(screen.getByText(/^Province/, { selector: "label" })).toHaveTextContent("Province (required)");
  await userEvent.click(screen.getByRole("button", { name: "Create event" }));
  const errors = screen.getAllByRole("alert").map((n) => n.textContent);
  expect(errors).toEqual(expect.arrayContaining([
    "Add a title.", "Add the place.", "Choose a region before saving.", "Pick the date.", "Enter a price above zero (e.g. 80 or 80.50).",
  ]));
  // One error style: the price error is a field error, tied to its input.
  expect(screen.getByLabelText(/^Price per photo · PHP/)).toHaveAttribute("aria-describedby", "event-price-per-photo-error");
  expect(submit).not.toHaveBeenCalled();
});

it("the create intro says the date decides the tab (FT-22, W22.2)", () => {
  mount("create");
  expect(screen.getByText("Title, location, and date populate the event card across the app. The cover banner is optional — events without a cover fall back to the dark-text banner. The date decides which tab the event appears under.")).toBeInTheDocument();
});

it("an empty submit moves focus to the first field with a problem (FT-22 review)", async () => {
  mount("create");
  fireEvent.change(screen.getByLabelText(/^Title/), { target: { value: "Race" } });
  fireEvent.change(screen.getByLabelText(/^Location/), { target: { value: "Venue" } });
  await userEvent.click(screen.getByRole("button", { name: "Create event" }));
  await waitFor(() => expect(screen.getByRole("button", { name: /^Region:/ })).toHaveFocus());
});

it("refuses a price finer than a centavo, which the server would round to ₱0.00 (FT-22 review)", async () => {
  const { submit } = mount("create"); fillCreate();
  await pick("Region", eventRegions[0].name); await pick("Province", "Cebu");
  fireEvent.change(screen.getByLabelText(/^Price per photo · PHP/), { target: { value: "0.001" } });
  await userEvent.click(screen.getByRole("button", { name: "Create event" }));
  expect(screen.getByText("Enter a price above zero (e.g. 80 or 80.50).")).toBeInTheDocument();
  expect(submit).not.toHaveBeenCalled();
});

it("an unclassified event's edit form does not mark Region and Province required (FT-22 review)", () => {
  mount("edit");
  expect(screen.getByText(/^Region/, { selector: "label" })).toHaveTextContent(/^Region$/);
  expect(screen.getByText(/^Title/, { selector: "label" })).toHaveTextContent("Title (required)");
});

it("the edit form says nobody is told and how to undo, in edit mode only (FU-4, U4.5)", () => {
  const line = "Nobody is told about an edit. To undo one, edit the event again.";
  mount("edit");
  expect(screen.getByText(line)).toBeInTheDocument();
  cleanup(); vi.spyOn(api, "get").mockReturnValue(new Promise(() => {}));
  mount("edit", { ...adminEvent, eventType: "ROAD_RACE" });
  expect(screen.getByText(line)).toBeInTheDocument();
  cleanup(); mount("create");
  expect(screen.queryByText(line)).not.toBeInTheDocument();
});
