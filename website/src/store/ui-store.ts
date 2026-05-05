import { create } from "zustand";

interface UiState {
  cartOpen: boolean;
  checkoutOpen: boolean;
  expressCheckoutPending: boolean;
  openCart: () => void;
  closeCart: () => void;
  openCheckout: () => void;
  closeCheckout: () => void;
  startExpressCheckout: () => void;
  clearExpressCheckout: () => void;
}

export const useUiStore = create<UiState>((set) => ({
  cartOpen: false,
  checkoutOpen: false,
  expressCheckoutPending: false,
  openCart: () => set({ cartOpen: true, checkoutOpen: false }),
  closeCart: () => set({ cartOpen: false }),
  openCheckout: () => set({ checkoutOpen: true, cartOpen: false }),
  closeCheckout: () => set({ checkoutOpen: false }),
  startExpressCheckout: () => set({ expressCheckoutPending: true }),
  clearExpressCheckout: () => set({ expressCheckoutPending: false }),
}));
