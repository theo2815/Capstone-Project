import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
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
  fireEvent.change(screen.getByLabelText("Title"), { target: { value: "Race" } });
  fireEvent.change(screen.getByLabelText("Location"), { target: { value: "Venue" } });
  fireEvent.change(screen.getByLabelText("Date"), { target: { value: "2026-11-01" } });
  fireEvent.change(screen.getByLabelText("Price per photo · PHP"), { target: { value: "80" } });
}

async function pick(field: string, option: string) {
  await userEvent.click(screen.getByRole("button", { name: new RegExp(`^${field}:`) }));
  await userEvent.click(screen.getByRole("menuitem", { name: option }));
}

it("requires a complete catalog-valid pair on create and clears province when region changes", async () => {
  const { submit } = mount("create"); fillCreate();
  expect(screen.getByRole("button", { name: "Create event" })).toBeDisabled();
  await pick("Region", eventRegions[0].name);
  expect(screen.getByRole("button", { name: "Create event" })).toBeDisabled();
  await pick("Province", "Cebu");
  expect(screen.getByRole("button", { name: "Create event" })).toBeEnabled();
  await pick("Region", eventRegions[1].name);
  expect(screen.getByRole("button", { name: "Create event" })).toBeDisabled();
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
  expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
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
  expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
  await pick("Province", "Cebu");
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
  expect(submit).toHaveBeenLastCalledWith(expect.objectContaining({ regionCode: "region-vii", provinceCode: "cebu" }));
});

it("preserves the dirty draft across refetch, resets on a fresh session, and owns Escape before the modal", async () => {
  const event = { ...adminEvent, regionCode: "region-vii", provinceCode: "cebu" };
  const { rerender, close, submit } = mount("edit", event);
  fireEvent.change(screen.getByLabelText("Title"), { target: { value: "Draft" } });
  await pick("Region", eventRegions[1].name); await pick("Province", "Leyte");
  rerender({ event: { ...event, name: "Remote", reviewRevision: 5 } });
  expect(screen.getByLabelText("Title")).toHaveValue("Draft");
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
  await waitFor(() => expect(screen.getByLabelText("Title")).toHaveValue("Cebu Race"));
  expect(within(screen.getByRole("dialog")).getByRole("button", { name: "Province: Cebu" })).toBeInTheDocument();
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
