"use client";

// AA-0.3a throwaway spike: never merged. Loaded only through next/dynamic.
import "@aws-amplify/ui-react/styles.css";
import {
  FaceLivenessDetectorCore,
  type AwsCredentialProvider,
} from "@aws-amplify/ui-react-liveness";

export interface SpikeCredentials {
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken: string;
  expiration: string;
}

export default function Detector({
  sessionId,
  region,
  credentials,
  selfHost,
}: {
  sessionId: string;
  region: string;
  credentials: SpikeCredentials;
  selfHost: boolean;
}) {
  // The AA-0.2 shape: STS keys from the backend, no Amplify.configure.
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
      onAnalysisComplete={async () => console.log("[spike-liveness] onAnalysisComplete")}
      onError={(e) => console.log("[spike-liveness] onError", e.state, e.error?.message)}
      config={
        selfHost
          ? {
              credentialProvider,
              binaryPath: "/spike-liveness/wasm/",
              faceModelUrl: "/spike-liveness/model/model.json",
            }
          : { credentialProvider }
      }
    />
  );
}
