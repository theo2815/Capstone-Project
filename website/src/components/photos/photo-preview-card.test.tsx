import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PhotoPreviewCard, type PhotoPreviewItem } from "./photo-preview-card";

describe("PhotoPreviewCard coupon offer", () => {
  it("shows only the coupon fields supplied by the photo API", async () => {
    const photo: PhotoPreviewItem = {
      id: "photo-1",
      bib: "101",
      time: "",
      tone: 0,
      price: 150,
      photographerName: "Aira",
      couponCode: "RACE20",
      couponPercentOff: 20,
      couponPrice: 127.5,
    };
    const props = {
      eventName: "Current Event",
      index: 1,
      total: 1,
      inCart: false,
      onClose: vi.fn(),
      onToggleCart: vi.fn(),
      onBuyNow: vi.fn(),
    };
    const { rerender } = render(<PhotoPreviewCard {...props} photo={photo} />);

    expect(await screen.findAllByText("RACE20")).toHaveLength(2);
    expect(screen.getAllByText(/20% off/)).toHaveLength(2);

    rerender(
      <PhotoPreviewCard
        {...props}
        photo={{ ...photo, id: "photo-2", couponCode: null, couponPercentOff: null, couponPrice: null }}
      />,
    );

    await waitFor(() => expect(screen.queryAllByText("RACE20")).toHaveLength(0));
  });
});
