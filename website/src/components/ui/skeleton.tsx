import { cn } from "@/lib/utils";

// Quiet Studio loading placeholders. The base Skeleton is a pulse-animated
// bone-deep block; the variants below cover the two shapes that come up
// across the photographer dashboard often enough to deserve their own
// helpers (text rows under a kicker, image tiles in a grid).
//
// Color rule: skeletons live on `bg-bone` surfaces, so they use `bg-bone-deep`
// to match the same low-contrast track Sparkline uses. Never `bg-gray-*` —
// those are Tailwind defaults outside the Quiet Studio palette.
//
// Wiring rule: skeletons mount behind useMockLatency (see lib/mock-latency.ts).
// With MOCK_LATENCY_MS = 0 (the dev default) they never appear; bump the
// constant to stress-test them or to demo the loading affordance.

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      aria-hidden="true"
      className={cn("animate-pulse rounded-md bg-bone-deep", className)}
    />
  );
}

interface TextSkeletonProps {
  /** Number of lines to render. Last line is auto-shortened to look natural. */
  lines?: number;
  className?: string;
}

// Stack of skeleton text rows. Mimics a paragraph or a stat caption block.
// The last row is shortened to ~70% so the placeholder doesn't read as a
// solid rectangle.
export function TextSkeleton({ lines = 3, className }: TextSkeletonProps) {
  return (
    <div className={cn("space-y-2", className)} aria-hidden="true">
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          className={cn("h-3", i === lines - 1 ? "w-[70%]" : "w-full")}
        />
      ))}
    </div>
  );
}

interface TileSkeletonProps {
  /** Tailwind aspect-ratio class fragment, e.g. "aspect-[3/2]" or "aspect-square". */
  aspectRatio?: string;
  className?: string;
}

// Block placeholder for tile/card grids. Default 3:2 matches EventTile's
// hero ratio so a covered-events grid can swap in skeletons without
// reflowing the layout.
export function TileSkeleton({
  aspectRatio = "aspect-[3/2]",
  className,
}: TileSkeletonProps) {
  return <Skeleton className={cn("w-full rounded-2xl", aspectRatio, className)} />;
}
