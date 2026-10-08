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
  screen.getByRole("menuitem", { name: eventRegions[1].name }).focus(); await user.keyboard("{Enter}");
  expect(event.getByLabelText("Province")).toHaveAccessibleName("Province: Pick a province");
  expect(settings.getByLabelText("Province")).toHaveAccessibleName("Province: Cebu");
});
