import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, expect, it } from "vitest";
import { eventRegions } from "@/test/fixtures/admin";
import { RegionProvinceFields, type RegionProvinceValue } from "./region-province-fields";

afterEach(cleanup);

it("keeps two controlled copies independent with unique labels and keyboard-accessible selections", async () => {
  function Copy({ name }: { name: string }) {
    const [value, onChange] = useState<RegionProvinceValue>({ regionCode: "region-vii", provinceCode: "cebu" });
    return <section aria-label={name}><RegionProvinceFields value={value} onChange={onChange} regions={eventRegions} /></section>;
  }
  render(<><Copy name="Settings" /><Copy name="Event" /></>);
  const settings = within(screen.getByRole("region", { name: "Settings" }));
  const event = within(screen.getByRole("region", { name: "Event" }));
  expect(settings.getByLabelText("Region").id).not.toBe(event.getByLabelText("Region").id);
  const user = userEvent.setup();
  event.getByLabelText("Region").focus(); await user.keyboard("{Enter}");
  const selected = screen.getByRole("menuitem", { name: eventRegions[0].name });
  expect(selected).toHaveAttribute("aria-current", "true");
  expect(selected).toHaveFocus();
  await user.keyboard("{ArrowDown}{Enter}");
  expect(event.getByLabelText("Region")).toHaveFocus();
  expect(event.getByLabelText("Province")).toHaveAccessibleName("Province: Pick a province");
  expect(settings.getByLabelText("Province")).toHaveAccessibleName("Province: Cebu");
});

it("marks both labels (required) only when asked (FT-22, R-04)", () => {
  const value = { regionCode: null, provinceCode: null };
  const { rerender } = render(<RegionProvinceFields value={value} onChange={() => {}} regions={eventRegions} />);
  expect(screen.queryByText(/\(required\)/)).not.toBeInTheDocument();
  rerender(<RegionProvinceFields value={value} onChange={() => {}} regions={eventRegions} required />);
  expect(screen.getByText(/^Region/, { selector: "label" })).toHaveTextContent("Region (required)");
  expect(screen.getByText(/^Province/, { selector: "label" })).toHaveTextContent("Province (required)");
});
