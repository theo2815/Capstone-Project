import { Kicker } from "@/components/ui/kicker";

export function FaceEmptyResult({ onClear }: { onClear: () => void }) {
  return (
    <section className="px-6 md:px-10 py-16 md:py-24 bg-bone min-h-[40vh] flex items-center">
      <div className="max-w-2xl mx-auto w-full text-center">
        <Kicker as="p" className="mb-3">
          No matches yet
        </Kicker>
        <p className="font-display font-extrabold text-3xl md:text-4xl text-ink tracking-tight">
          We didn&apos;t find your face.
        </p>
        <p className="font-sans text-base md:text-lg text-ink-soft mt-4">
          Try adding another selfie angle, or browse the wall while photos roll
          in.
        </p>
        <button
          type="button"
          onClick={onClear}
          className="mt-7 inline-flex items-center gap-2 bg-fresh hover:bg-fresh-deep text-surface px-6 py-3 rounded-full font-display font-bold text-[15px] transition-colors"
        >
          Browse the wall →
        </button>
      </div>
    </section>
  );
}
