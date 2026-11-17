"use client";

import { BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { Modal } from "@/components/ui/modal";
import type { RaceCandidate } from "@/lib/api-photographer";
import { RACE_PENDING_JOIN_LINE } from "@/lib/photographer-events";
import { cn, formatDate } from "@/lib/utils";

// The server's "this may already exist" answer to a race create (AA-R3):
// `match` (one strong candidate) or `candidates` (medium ones). The
// photographer joins one, or says theirs is different, which always lands as
// a review case for an admin. The client never picks a scenario itself.
interface RaceDuplicateDialogProps {
  outcome: "match" | "candidates" | null;
  candidates: RaceCandidate[];
  busy: boolean;
  error: string | null;
  onJoin: (candidate: RaceCandidate) => void;
  onDifferent: () => void;
  onClose: () => void;
}

export function RaceDuplicateDialog({
  outcome,
  candidates,
  busy,
  error,
  onJoin,
  onDifferent,
  onClose,
}: RaceDuplicateDialogProps) {
  return (
    <Modal
      isOpen={outcome !== null}
      onClose={() => { if (!busy) onClose(); }}
      title={outcome === "match" ? "This race is already listed." : "Is it one of these races?"}
    >
      <p className="font-sans text-sm text-ink-soft max-w-prose">
        Join the race so your photos land beside everyone else&apos;s. If yours
        is a different race, say so and an admin checks it.
      </p>
      <ul className="mt-6 space-y-4">
        {candidates.map((c) => (
          <li key={c.id} className="rounded-2xl border border-line bg-surface p-5">
            <p className="font-display text-lg font-bold text-ink">{c.name}</p>
            <p className="font-sans text-sm text-slate mt-1 tnum">
              {formatDate(c.date)} · {c.location}
            </p>
            {c.reviewStatus === "pending" && (
              <p className="font-sans text-sm text-ink-soft mt-2">{RACE_PENDING_JOIN_LINE}</p>
            )}
            <button
              type="button"
              onClick={() => onJoin(c)}
              disabled={busy}
              // One fresh button per viewport: primary only for a single match.
              className={cn(candidates.length === 1 ? BTN_PRIMARY : BTN_SECONDARY, BTN_SIZE.sm, "mt-4")}
            >
              Join {c.name}
            </button>
          </li>
        ))}
      </ul>
      {error && (
        <p role="alert" className="font-sans text-sm text-error mt-4">{error}</p>
      )}
      <div className="mt-6 flex flex-wrap gap-3">
        <button
          type="button"
          onClick={onDifferent}
          disabled={busy}
          className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
        >
          Mine is different
        </button>
      </div>
    </Modal>
  );
}
