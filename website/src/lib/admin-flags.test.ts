import { expect, it } from "vitest";
import { flagReasonLabel } from "./admin-flags";

// AX-64: the backend owns the labels (AX-59); the website shows what the row carries.
it("shows the label the flag row carries", () => {
  expect(flagReasonLabel({ reason: "runner_removal_held", reasonLabel: "Runner deletion held" })).toBe("Runner deletion held");
});

it("falls back to the code when the row has no label (an older backend, or an unknown code)", () => {
  expect(flagReasonLabel({ reason: "runner_removal_held" })).toBe("runner_removal_held");
  expect(flagReasonLabel({ reason: "something_new", reasonLabel: null })).toBe("something_new");
  expect(flagReasonLabel({ reason: "something_new", reasonLabel: "" })).toBe("something_new");
});
