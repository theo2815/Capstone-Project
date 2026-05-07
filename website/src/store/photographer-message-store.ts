import { create } from "zustand";
import type {
  PhotographerMessage,
  PhotographerMessageKind,
} from "@/lib/photographer-messages";

// Mock-only store of photographer inbox messages. Submissions are append-only;
// overrides hold sparse `readAt` patches keyed by message id. Real backend
// ships in Phase F (`POST /me/photographer/messages/{id}/read`).
//
// This store is written-to from admin-side actions:
//   • admin-payout-store.hold / bulkHold      → kind = "payout_held"
//   • admin-payout-report-store.acknowledge   → kind = "report_acknowledged"
//   • admin-payout-report-store.resolve       → kind = "report_resolved"
// Cross-store calls use `usePhotographerMessageStore.getState().addMessage()`
// — keeping the inbox a single source of truth for unread-count + display.

export interface AddMessagePayload {
  photographerId: string;
  kind: PhotographerMessageKind;
  title: string;
  body: string;
  payoutCycleId?: string;
  reportId?: string;
}

interface PhotographerMessageState {
  submissions: PhotographerMessage[];
  overrides: Record<string, Partial<PhotographerMessage>>;
  addMessage: (payload: AddMessagePayload) => PhotographerMessage;
  markRead: (id: string) => void;
  markAllRead: (photographerId: string) => void;
  removeMessage: (id: string) => void;
  clear: () => void;
}

function makeMessageId(): string {
  const slug = Math.random().toString(36).slice(2, 10);
  return `msg-${slug}`;
}

function nowIso(): string {
  return new Date().toISOString();
}

export const usePhotographerMessageStore = create<PhotographerMessageState>(
  (set, get) => ({
    submissions: [],
    overrides: {},
    addMessage: (payload) => {
      const message: PhotographerMessage = {
        id: makeMessageId(),
        photographerId: payload.photographerId,
        kind: payload.kind,
        title: payload.title,
        body: payload.body,
        payoutCycleId: payload.payoutCycleId,
        reportId: payload.reportId,
        createdAt: nowIso(),
        readAt: null,
      };
      set((s) => ({ submissions: [message, ...s.submissions] }));
      return message;
    },
    markRead: (id) => {
      set((s) => ({
        overrides: {
          ...s.overrides,
          [id]: { ...s.overrides[id], readAt: nowIso() },
        },
      }));
    },
    markAllRead: (photographerId) => {
      const at = nowIso();
      const all = [...get().submissions];
      // Seed messages need a read flag too — read them off the lib seed via the
      // overrides keyed by their id. We don't have direct access to seed here,
      // so the inbox modal passes effective messages and walks the unread set
      // via repeated markRead calls. Implement that directly to keep logic
      // colocated.
      set((s) => {
        const next: Record<string, Partial<PhotographerMessage>> = {
          ...s.overrides,
        };
        for (const m of all) {
          if (m.photographerId === photographerId && m.readAt === null) {
            next[m.id] = { ...next[m.id], readAt: at };
          }
        }
        return { overrides: next };
      });
    },
    removeMessage: (id) => {
      // Soft-delete via override so seed messages can be removed too without
      // touching the seed array. getEffectiveMessages filters `removed:true`.
      set((s) => ({
        overrides: {
          ...s.overrides,
          [id]: { ...s.overrides[id], removed: true },
        },
      }));
    },
    clear: () => set({ submissions: [], overrides: {} }),
  }),
);

// Helper for the inbox modal to mark every unread effective message read in
// one set call. Pass the messages already filtered to the current
// photographer — the helper flips only those whose readAt is null.
export function markAllEffectiveRead(
  effectiveMessages: ReadonlyArray<PhotographerMessage>,
): void {
  const at = new Date().toISOString();
  usePhotographerMessageStore.setState((s) => {
    const next: Record<string, Partial<PhotographerMessage>> = { ...s.overrides };
    for (const m of effectiveMessages) {
      if (m.readAt === null) {
        next[m.id] = { ...next[m.id], readAt: at };
      }
    }
    return { overrides: next };
  });
}
