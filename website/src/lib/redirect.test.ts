import { describe, expect, it } from "vitest";
import { ROUTES } from "./constants";
import { roleHome } from "./redirect";

describe("roleHome", () => {
  it("sends an admin to the Inbox", () => {
    expect(roleHome("ADMIN")).toBe(ROUTES.ADMIN_INBOX);
  });

  it("leaves photographers and runners where they were", () => {
    expect(roleHome("PHOTOGRAPHER")).toBe(ROUTES.DASHBOARD);
    expect(roleHome("RUNNER")).toBe(ROUTES.EVENTS);
  });
});
