import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EventCouponModal } from "@/components/dashboard/event-coupon-modal";

const couponApi = vi.hoisted(() => ({
  fetchEventCoupon: vi.fn(),
  putEventCoupon: vi.fn(),
  deleteEventCoupon: vi.fn(),
}));

vi.mock("@/lib/api-coupons", () => couponApi);
vi.mock("@/hooks/use-photographer-data", () => ({
  usePlatformFees: () => ({ couponMaxPercent: 50 }),
}));

describe("EventCouponModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    couponApi.fetchEventCoupon.mockResolvedValue(null);
    couponApi.putEventCoupon.mockResolvedValue(undefined);
  });

  it("preselects the current owned paid event and creates without photo ids", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <EventCouponModal
        isOpen
        onClose={onClose}
        initialEventId="event-2"
        lockEvent
        events={[
          { id: "foreign", name: "Foreign", ownedByMe: false, pricingMode: "paid" },
          { id: "free", name: "Free", ownedByMe: true, pricingMode: "free" },
          { id: "event-1", name: "First", ownedByMe: true, pricingMode: "paid" },
          { id: "event-2", name: "Current", ownedByMe: true, pricingMode: "paid" },
        ]}
      />,
    );

    const eventSelect = await screen.findByLabelText("Event");
    expect(eventSelect).toHaveValue("event-2");
    expect(eventSelect).toBeDisabled();
    expect(screen.queryByRole("option", { name: "Foreign" })).not.toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "Free" })).not.toBeInTheDocument();
    await waitFor(() =>
      expect(couponApi.fetchEventCoupon).toHaveBeenCalledWith("event-2"),
    );

    await user.type(screen.getByLabelText("Coupon code"), "race20");
    await user.click(screen.getByRole("button", { name: "Create coupon" }));

    await waitFor(() =>
      expect(couponApi.putEventCoupon).toHaveBeenCalledWith("event-2", {
        code: "RACE20",
        percentOff: 10,
        active: true,
        expiresAt: null,
        usageLimit: null,
      }),
    );
    expect(onClose).toHaveBeenCalledOnce();
  });
});
