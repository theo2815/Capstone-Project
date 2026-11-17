import { expect, it } from "vitest";
import { flagReasonLabel } from "./admin-flags";

it("runner_removal_held has a label", () => {
  expect(flagReasonLabel("runner_removal_held")).toBe("Runner deletion held");
});
