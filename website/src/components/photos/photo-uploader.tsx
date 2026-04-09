"use client";

import { useState, useCallback, type DragEvent, type ChangeEvent } from "react";
import { Button } from "@/components/ui/button";
import { ACCEPTED_IMAGE_TYPES, MAX_UPLOAD_SIZE } from "@/lib/constants";
import { Upload, X, ImageIcon } from "lucide-react";

interface PhotoUploaderProps {
  onUpload: (files: File[]) => void;
  isUploading?: boolean;
  maxFiles?: number;
}

export function PhotoUploader({
  onUpload,
  isUploading,
  maxFiles = 50,
}: PhotoUploaderProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");

  const validateAndAdd = useCallback(
    (newFiles: FileList | File[]) => {
      setError("");
      const valid: File[] = [];

      for (const file of Array.from(newFiles)) {
        if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) {
          setError(`${file.name}: unsupported format. Use JPEG, PNG, or WebP.`);
          continue;
        }
        if (file.size > MAX_UPLOAD_SIZE) {
          setError(`${file.name}: exceeds 10 MB limit.`);
          continue;
        }
        valid.push(file);
      }

      setFiles((prev) => {
        const combined = [...prev, ...valid];
        if (combined.length > maxFiles) {
          setError(`Maximum ${maxFiles} files per upload.`);
          return combined.slice(0, maxFiles);
        }
        return combined;
      });
    },
    [maxFiles],
  );

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    setDragOver(false);
    validateAndAdd(e.dataTransfer.files);
  }

  function handleChange(e: ChangeEvent<HTMLInputElement>) {
    if (e.target.files) validateAndAdd(e.target.files);
  }

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors ${
          dragOver
            ? "border-blue-500 bg-blue-50"
            : "border-gray-300 bg-gray-50"
        }`}
      >
        <Upload className="mb-2 h-8 w-8 text-gray-400" />
        <p className="text-sm text-gray-600">
          Drag & drop photos here, or{" "}
          <label className="cursor-pointer font-medium text-blue-600 hover:underline">
            browse
            <input
              type="file"
              multiple
              accept={ACCEPTED_IMAGE_TYPES.join(",")}
              onChange={handleChange}
              className="hidden"
            />
          </label>
        </p>
        <p className="mt-1 text-xs text-gray-400">
          JPEG, PNG, WebP up to 10 MB. Max {maxFiles} files.
        </p>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      {files.length > 0 && (
        <>
          <div className="max-h-60 space-y-2 overflow-y-auto">
            {files.map((file, i) => (
              <div
                key={`${file.name}-${i}`}
                className="flex items-center justify-between rounded-lg border border-gray-200 px-3 py-2"
              >
                <div className="flex items-center gap-2 truncate">
                  <ImageIcon className="h-4 w-4 shrink-0 text-gray-400" />
                  <span className="truncate text-sm text-gray-700">
                    {file.name}
                  </span>
                  <span className="shrink-0 text-xs text-gray-400">
                    {(file.size / 1024 / 1024).toFixed(1)} MB
                  </span>
                </div>
                <button onClick={() => removeFile(i)}>
                  <X className="h-4 w-4 text-gray-400 hover:text-red-500" />
                </button>
              </div>
            ))}
          </div>

          <Button
            onClick={() => onUpload(files)}
            isLoading={isUploading}
            disabled={files.length === 0}
          >
            Upload {files.length} {files.length === 1 ? "photo" : "photos"}
          </Button>
        </>
      )}
    </div>
  );
}
