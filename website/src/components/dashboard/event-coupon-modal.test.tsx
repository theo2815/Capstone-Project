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

  it("preselects the current paid event and creates without photo ids", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <EventCouponModal
        isOpen
        onClose={onClose}
        initialEventId="event-2"
        lockEvent
        events={[
          { id: "admin", name: "Admin race", pricingMode: "paid" },
          { id: "free", name: "Free", pricingMode: "free" },
          { id: "event-1", name: "First", pricingMode: "paid" },
          { id: "event-2", name: "Current", pricingMode: "paid" },
        ]}
      />,
    );

    const eventSelect = await screen.findByLabelText("Event");
    expect(eventSelect).toHaveValue("event-2");
    expect(eventSelect).toBeDisabled();
    // Covered admin events qualify; free ones never do.
    expect(screen.getByRole("option", { name: "Admin race" })).toBeInTheDocument();
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

  it("offers an admin-created event the photographer covered", async () => {
    render(
      <EventCouponModal
        isOpen
        onClose={vi.fn()}
        events={[
          { id: "admin", name: "Admin race", pricingMode: "paid" },
        ]}
      />,
    );

    expect(await screen.findByLabelText("Event")).toHaveValue("admin");
    expect(screen.queryByText(/nothing to discount/i)).not.toBeInTheDocument();
    await waitFor(() =>
      expect(couponApi.fetchEventCoupon).toHaveBeenCalledWith("admin"),
    );
  });

  it("explains the empty state without asking for a new event", () => {
    render(
      <EventCouponModal
        isOpen
        onClose={vi.fn()}
        events={[{ id: "free", name: "Free", pricingMode: "free" }]}
      />,
    );

    expect(screen.getByText(/nothing to discount/i)).toBeInTheDocument();
    expect(screen.queryByText(/create a paid event/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Event")).not.toBeInTheDocument();
    expect(couponApi.fetchEventCoupon).not.toHaveBeenCalled();
  });
});
