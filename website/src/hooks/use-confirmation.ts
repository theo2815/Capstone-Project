"use client";

import {
  useConfirmationStore,
  makeConfirmationId,
  type ConfirmConfig,
} from "@/store/confirmation-store";

export function useConfirmation() {
  const open = useConfirmationStore((s) => s.open);

  function confirm(config: ConfirmConfig): Promise<boolean> {
    return new Promise<boolean>((resolve) => {
      open({ id: makeConfirmationId(), config, resolve });
    });
  }

  return { confirm };
}
