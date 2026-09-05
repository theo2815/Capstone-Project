import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PhotoMosaicTile } from "./photo-mosaic-tile";
import type { EventDetail } from "@/types/event";
import type { MockPhoto } from "@/types/photo";

describe("PhotoMosaicTile coupon badge", () => {
  it("renders only the coupon eligibility returned with that photo", () => {
    const event: EventDetail = {
      id: "event-1",
      slug: "event-1",
      name: "Event One",
      date: "2026-09-04",
      location: "Cebu",
      photoCount: 1,
      participantCount: 1,
      status: "ACTIVE",
      description: "",
      organizerName: "",
      categories: [],
      pricePerPhoto: 150,
    };
    const photo: MockPhoto = {
      id: "photo-1",
      bib: "101",
      km: null,
      time: "",
      tone: 0,
      span: "default",
      price: 150,
      couponCode: "RACE20",
      couponPercentOff: 20,
      couponPrice: 127.5,
    };
    const { rerender } = render(
      <PhotoMosaicTile event={event} photo={photo} index={0} onOpen={vi.fn()} />,
    );

    // Auto-apply (2026-09-05): the chip announces the offer; nothing to copy.
    expect(screen.getByText("−20%")).toBeInTheDocument();
    expect(screen.getByText(/applied at checkout/)).toBeInTheDocument();
    expect(screen.queryByText(/RACE20/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Copy coupon/ })).not.toBeInTheDocument();

    rerender(
      <PhotoMosaicTile
        event={{ ...event, id: "event-2" }}
        photo={{
          ...photo,
          id: "photo-2",
          couponCode: null,
          couponPercentOff: null,
          couponPrice: null,
        }}
        index={0}
        onOpen={vi.fn()}
      />,
    );

    expect(screen.queryByText(/applied at checkout/)).not.toBeInTheDocument();
  });
});
