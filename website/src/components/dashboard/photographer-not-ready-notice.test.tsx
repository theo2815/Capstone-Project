import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { PhotographerNotReadyNotice } from "./photographer-not-ready-notice";

it("a held profile tells the photographer an admin is reviewing it", () => {
  render(<PhotographerNotReadyNotice gate={{ kind: "pending", awaitingAdmin: true }} />);
  expect(screen.getByText("An admin is reviewing your profile.")).toBeInTheDocument();
  expect(screen.queryByText("Your studio goes live when your identity check passes.")).toBeNull();
});

it("a pending profile without the hold keeps the identity text", () => {
  render(<PhotographerNotReadyNotice gate={{ kind: "pending" }} />);
  expect(screen.getByText("Your studio goes live when your identity check passes.")).toBeInTheDocument();
  expect(screen.queryByText("An admin is reviewing your profile.")).toBeNull();
});
