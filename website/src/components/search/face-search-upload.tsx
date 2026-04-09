"use client";

import { useState, useRef, useCallback } from "react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Camera, Upload, X } from "lucide-react";

interface FaceSearchUploadProps {
  onSearch: (image: File) => void;
  isSearching?: boolean;
}

export function FaceSearchUpload({
  onSearch,
  isSearching,
}: FaceSearchUploadProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [useWebcam, setUseWebcam] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setUseWebcam(false);
  }, []);

  async function startWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setUseWebcam(true);
      setPreview(null);
      setFile(null);
    } catch {
      // User denied camera
    }
  }

  function capturePhoto() {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext("2d")?.drawImage(videoRef.current, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const captured = new File([blob], "selfie.jpg", { type: "image/jpeg" });
      handleFile(captured);
      stopWebcam();
    }, "image/jpeg");
  }

  function stopWebcam() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setUseWebcam(false);
  }

  function clear() {
    setPreview(null);
    setFile(null);
    stopWebcam();
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900">Search by Face</h3>

      {!preview && !useWebcam && (
        <div className="flex gap-3">
          <label className="cursor-pointer">
            <span className="inline-flex items-center justify-center rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50">
              <Upload className="mr-2 h-4 w-4" /> Upload Photo
            </span>
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
          </label>
          <Button variant="outline" onClick={startWebcam}>
            <Camera className="mr-2 h-4 w-4" /> Use Webcam
          </Button>
        </div>
      )}

      {useWebcam && (
        <div className="space-y-3">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full max-w-sm rounded-lg"
          />
          <div className="flex gap-3">
            <Button onClick={capturePhoto}>Capture</Button>
            <Button variant="outline" onClick={stopWebcam}>
              Cancel
            </Button>
          </div>
        </div>
      )}

      {preview && (
        <div className="space-y-3">
          <div className="relative inline-block">
            <Image
              src={preview}
              alt="Selected face"
              width={200}
              height={200}
              className="rounded-lg object-cover"
            />
            <button
              onClick={clear}
              className="absolute -right-2 -top-2 rounded-full bg-white p-1 shadow"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <div>
            <Button
              onClick={() => file && onSearch(file)}
              isLoading={isSearching}
            >
              Search
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
