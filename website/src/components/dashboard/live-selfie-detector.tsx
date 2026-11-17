"use client";

// Loaded only through next/dynamic from live-selfie-check.tsx, so this chunk
// and its stylesheet download only when the live check opens.
import "@aws-amplify/ui-react/styles.css";
import {
  FaceLivenessDetectorCore,
  type AwsCredentialProvider,
  type ErrorState,
} from "@aws-amplify/ui-react-liveness";
import type { LiveSelfieCheckProps, LiveSelfieErrorKind } from "./live-selfie-check";

// Self-hosted copies (AA-0.3a): no third-party origin in the CSP. The WASM
// folder name carries the installed tfjs-backend-wasm version; a test fails
// when the package moves and these files were not re-copied.
export const LIVENESS_BINARY_PATH = "/liveness/tfjs-backend-wasm-4.11.0/";
const LIVENESS_FACE_MODEL_URL = "/liveness/blazeface-1.0.2/model.json";

function errorKind(state: ErrorState): LiveSelfieErrorKind {
  switch (state) {
    case "CAMERA_ACCESS_ERROR":
    case "CAMERA_FRAMERATE_ERROR":
    case "DEFAULT_CAMERA_NOT_FOUND_ERROR":
      return "camera";
    case "TIMEOUT":
    case "FRESHNESS_TIMEOUT":
      return "timeout";
    case "CONNECTION_TIMEOUT":
    case "SERVER_ERROR": // also an expired session (AA-0.3b): the caller starts a new one
      return "connection";
    default:
      return "other";
  }
}

export default function LiveSelfieDetector({
  sessionId,
  region,
  credentials,
  onComplete,
  onError,
  onCancel,
}: LiveSelfieCheckProps) {
  // Short-lived STS keys from the backend (AA-0.2); no Amplify.configure.
  const credentialProvider: AwsCredentialProvider = async () => ({
    accessKeyId: credentials.accessKeyId,
    secretAccessKey: credentials.secretAccessKey,
    sessionToken: credentials.sessionToken,
    expiration: new Date(credentials.expiration),
  });

  return (
    <FaceLivenessDetectorCore
      sessionId={sessionId}
      region={region}
      onAnalysisComplete={async () => onComplete()}
      onError={(e) => onError(errorKind(e.state))}
      onUserCancel={onCancel}
      config={{ credentialProvider, binaryPath: LIVENESS_BINARY_PATH, faceModelUrl: LIVENESS_FACE_MODEL_URL }}
    />
  );
}
