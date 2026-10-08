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
  claim: (id: string) => boolean;
  close: (id: string, resolveWith: boolean) => void;
  fail: (id: string, msg: string) => void;
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
    if (get().loading) {
      req.resolve(false);
      return;
    }
    set({ active: req, loading: false, error: null });
    prev?.resolve(false);
  },
  claim: (id) => {
    if (get().active?.id !== id || get().loading) return false;
    set({ loading: true, error: null });
    return true;
  },
  close: (id, resolveWith) => {
    const current = get().active;
    if (current?.id !== id || (get().loading && !resolveWith)) return;
    set({ active: null, loading: false, error: null });
    current.resolve(resolveWith);
  },
  fail: (id, msg) => {
    if (get().active?.id === id) set({ loading: false, error: msg });
  },
}));

export { makeId as makeConfirmationId };
