import { create } from "zustand";
import type { ReactNode } from "react";

export interface ConfirmConfig {
  title: string;
  message: string | ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean;
  onConfirm?: () => void | Promise<void>;
}

export interface ConfirmationRequest {
  id: string;
  config: ConfirmConfig;
  resolve: (value: boolean) => void;
}

interface ConfirmationState {
  active: ConfirmationRequest | null;
  loading: boolean;
  error: string | null;
  open: (req: ConfirmationRequest) => void;
  close: (resolveWith: boolean) => void;
  setLoading: (v: boolean) => void;
  setError: (msg: string | null) => void;
}

function makeId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `confirm-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export const useConfirmationStore = create<ConfirmationState>((set, get) => ({
  active: null,
  loading: false,
  error: null,
  open: (req) => {
    const prev = get().active;
    if (prev) {
      // A confirmation is already on screen. Reject the older one as cancelled
      // so its caller can unblock; the newer one takes the singleton slot.
      prev.resolve(false);
    }
    set({ active: req, loading: false, error: null });
  },
  close: (resolveWith) => {
    const current = get().active;
    if (!current) return;
    current.resolve(resolveWith);
    set({ active: null, loading: false, error: null });
  },
  setLoading: (v) => set({ loading: v }),
  setError: (msg) => set({ error: msg }),
}));

export { makeId as makeConfirmationId };
