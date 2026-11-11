import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api";
import type { RacePrice } from "@/lib/api-photographer";
import { RacePriceBand } from "./race-price-modal";

const photographerApi = vi.hoisted(() => ({ putRacePrice: vi.fn() }));
vi.mock("@/lib/api-photographer", () => photographerApi);

afterEach(() => vi.clearAllMocks());

const price: RacePrice = { floorPhp: 125, ceilingPhp: 250, myPricePhp: null, effectivePhp: 125, nextChangeAt: null };

function mount(racePrice: RacePrice | null) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><RacePriceBand eventId="race-1" racePrice={racePrice} /></QueryClientProvider>);
}

it("hidden when racePrice is null", () => {
  const { container } = mount(null);
  expect(container).toBeEmptyDOMElement();
});

it("disabled until nextChangeAt", async () => {
  mount({ ...price, myPricePhp: 180, effectivePhp: 180, nextChangeAt: new Date(Date.now() + 3_600_000).toISOString() });
  await userEvent.click(screen.getByRole("button", { name: "Change my price" }));
  expect(screen.getByLabelText("My price · PHP")).toBeDisabled();
  expect(screen.getByRole("button", { name: "Save price" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Use the race price" })).toBeDisabled();
  expect(screen.getByText(/Next change after/)).toBeInTheDocument();
});

it("saves a price inside the range and goes back to the race price with null", async () => {
  photographerApi.putRacePrice.mockResolvedValue({ ...price, myPricePhp: 150, effectivePhp: 150 });
  mount({ ...price, myPricePhp: 180, effectivePhp: 180 });
  await userEvent.click(screen.getByRole("button", { name: "Change my price" }));
  const input = screen.getByLabelText("My price · PHP");
  expect(input).toHaveValue(180);
  expect(screen.getByRole("button", { name: "Save price" })).toBeDisabled();
  await userEvent.clear(input); await userEvent.type(input, "300");
  expect(screen.getByRole("button", { name: "Save price" })).toBeDisabled();
  await userEvent.clear(input); await userEvent.type(input, "150");
  await userEvent.click(screen.getByRole("button", { name: "Save price" }));
  expect(photographerApi.putRacePrice).toHaveBeenLastCalledWith("race-1", 150);
  await userEvent.click(screen.getByRole("button", { name: "Change my price" }));
  await userEvent.click(screen.getByRole("button", { name: "Use the race price" }));
  expect(photographerApi.putRacePrice).toHaveBeenLastCalledWith("race-1", null);
});

it("409 keeps input and shows message", async () => {
  photographerApi.putRacePrice.mockRejectedValue(new ApiError([{ code: "RACE_PRICE_COOLDOWN", message: "You can change your price again after 2026-11-12T10:00Z." }], 409));
  mount(price);
  await userEvent.click(screen.getByRole("button", { name: "Change my price" }));
  const input = screen.getByLabelText("My price · PHP");
  await userEvent.clear(input); await userEvent.type(input, "150");
  await userEvent.click(screen.getByRole("button", { name: "Save price" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("You can change your price again after 2026-11-12T10:00Z.");
  expect(input).toHaveValue(150);
  expect(screen.getByRole("dialog")).toBeInTheDocument();
});
