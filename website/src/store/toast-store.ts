import { create } from "zustand";

export type ToastKind = "success" | "error" | "info";

export interface ToastAction {
  label: string;
  onClick: () => void;
}

export interface ToastLink {
  label: string;
  href: string;
}

export interface Toast {
  id: string;
  kind: ToastKind;
  message: string;
  action?: ToastAction;
  link?: ToastLink;
  duration: number;
}

const DEFAULT_DURATION_MS = 4000;

interface ToastState {
  toasts: Toast[];
  showToast: (
    input: {
      kind: ToastKind;
      message: string;
      action?: ToastAction;
      link?: ToastLink;
      duration?: number;
    },
  ) => string;
  dismissToast: (id: string) => void;
  clear: () => void;
}

function makeId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `toast-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  showToast: ({ kind, message, action, link, duration }) => {
    const id = makeId();
    const toast: Toast = {
      id,
      kind,
      message,
      action,
      link,
      duration: duration ?? DEFAULT_DURATION_MS,
    };
    set((state) => ({ toasts: [...state.toasts, toast] }));
    return id;
  },
  dismissToast: (id) =>
    set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
  clear: () => set({ toasts: [] }),
}));
