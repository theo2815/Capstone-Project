"use client";

import { useEffect } from "react";
import dynamic from "next/dynamic";
import { Modal } from "@/components/ui/modal";

/** The STS keys in the liveness-session response (C1). */
export interface LivenessCredentials {
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken: string;
  expiration: string;
}

export type LiveSelfieErrorKind = "camera" | "timeout" | "connection" | "load" | "other";

export interface LiveSelfieCheckProps {
  sessionId: string;
  region: string;
  credentials: LivenessCredentials;
  onComplete: () => void;
  onError: (kind: LiveSelfieErrorKind) => void;
  onCancel: () => void;
}

// If the detector chunk cannot download, report "load" so the caller can
// offer the admin check instead.
function LoadFailed({ onError }: LiveSelfieCheckProps) {
  useEffect(() => onError("load"), [onError]);
  return null;
}

const LiveSelfieDetector = dynamic(
  () => import("./live-selfie-detector").catch(() => LoadFailed),
  { ssr: false },
);

/**
 * The AWS Face Liveness check in a modal. Callers own starting the session
 * and submitting its result; this only runs the camera step.
 */
export default function LiveSelfieCheck(props: LiveSelfieCheckProps) {
  return (
    <Modal isOpen onClose={props.onCancel} title="Live check">
      <LiveSelfieDetector {...props} />
    </Modal>
  );
}
