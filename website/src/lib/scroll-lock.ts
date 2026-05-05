import { useEffect } from "react";

let lockCount = 0;
let savedOverflow = "";

function lock(): void {
  if (typeof document === "undefined") return;
  if (lockCount === 0) {
    savedOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
  }
  lockCount += 1;
}

function unlock(): void {
  if (typeof document === "undefined") return;
  if (lockCount === 0) return;
  lockCount -= 1;
  if (lockCount === 0) {
    document.body.style.overflow = savedOverflow;
    savedOverflow = "";
  }
}

export function useScrollLock(active: boolean): void {
  useEffect(() => {
    if (!active) return;
    lock();
    return () => unlock();
  }, [active]);
}
