"use client";

import { useCallback, useEffect, useRef, type RefObject, type ReactNode } from "react";
import { X } from "lucide-react";
import { Modal } from "@/components/ui/modal";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { usePhotoMetadata } from "@/hooks/use-photo-metadata";
import { ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { PhotoMetadataAccess, PhotoMetadataFields } from "@/types/photo-metadata";

interface PhotoMetadataDialogProps {
  photoId: string;
  access: PhotoMetadataAccess;
  isOpen: boolean;
  onClose: () => void;
  onDenied?: () => void;
  triggerRef: RefObject<HTMLElement | null>;
  fallbackFocus?: () => void;
}

const fallback = "Not recorded";
const display = (value: string | number | null | undefined) => value == null || value === "" ? fallback : String(value);

function cameraModel(fields: PhotoMetadataFields | null | undefined) {
  const make = fields?.cameraMake?.trim();
  const model = fields?.cameraModel?.trim();
  if (!make) return model;
  if (!model) return make;
  return model.toLowerCase() === make.toLowerCase() || model.toLowerCase().startsWith(`${make.toLowerCase()} `)
    ? model : `${make} ${model}`;
}

function row(label: string, value: string | number | null | undefined, numeric = false) {
  return <div className="grid gap-1 border-b border-line py-3 last:border-b-0" key={label}>
    <dt className="font-sans text-sm font-semibold text-slate">{label}</dt>
    <dd className={cn("min-w-0 break-words font-sans text-base text-ink", numeric && "font-mono tnum")}>{display(value)}</dd>
  </div>;
}

function section(title: string, content: ReactNode) {
  return <section className="border-t border-line pt-5">
    <h3 className="font-display text-base font-bold text-ink">{title}</h3>
    <dl>{content}</dl>
  </section>;
}

export function PhotoMetadataDialog({ photoId, access, isOpen, onClose, onDenied, triggerRef, fallbackFocus }: PhotoMetadataDialogProps) {
  const query = usePhotoMetadata(photoId, access, isOpen);
  const contentRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const focusClose = useCallback((node: HTMLButtonElement | null) => {
    closeButtonRef.current = node;
    if (node && isOpen) node.focus();
  }, [isOpen]);
  const refetch = () => {
    closeButtonRef.current?.focus();
    void query.refetch();
  };

  useEffect(() => {
    if (!isOpen) return;
    const trigger = triggerRef.current;
    const containTab = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const controls = contentRef.current?.querySelectorAll<HTMLElement>("button:not(:disabled), a[href], input:not(:disabled)");
      if (!controls?.length) return;
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && (document.activeElement === first || !contentRef.current?.contains(document.activeElement))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (document.activeElement === last || !contentRef.current?.contains(document.activeElement))) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", containTab, true);
    return () => {
      document.removeEventListener("keydown", containTab, true);
      if (trigger?.isConnected) trigger.focus();
      else fallbackFocus?.();
    };
  }, [isOpen, triggerRef, fallbackFocus]);

  const data = query.data;
  const missing = access === "public" && query.error instanceof ApiError && query.error.status === 404;
  useEffect(() => {
    if (isOpen && missing) onDenied?.();
  }, [isOpen, missing, onDenied]);
  const loading = query.isPending || query.isFetching;
  const status = !loading && !query.isError ? data?.status : undefined;
  const fields = status === "ready" ? data?.fields : null;
  const fileFacts = (status === "ready" || status === "empty" || status === "unavailable") && data;

  return <Modal isOpen={isOpen} onClose={onClose} title="Photo metadata" hideClose
    className="max-w-md p-5 md:p-7 [&>h2]:font-display [&>h2]:text-xl [&>h2]:font-bold [&>h2]:normal-case [&>h2]:tracking-normal [&>h2]:text-ink">
    <div ref={contentRef} className="space-y-5">
      <div className="absolute right-2 top-2">
        <button ref={focusClose} type="button" onClick={onClose} aria-label="Close metadata"
          className="flex size-11 items-center justify-center rounded-full text-slate hover:bg-bone-deep hover:text-ink">
          <X className="size-4" aria-hidden="true" />
        </button>
      </div>
      {loading ? <div role="status" aria-label="Loading metadata" className="space-y-3">
        <p className="font-sans text-base text-slate">Loading metadata…</p>
        <div aria-hidden="true" className="h-16 animate-pulse rounded-lg bg-bone-deep" />
        <div aria-hidden="true" className="h-16 animate-pulse rounded-lg bg-bone-deep" />
      </div> : missing ? <p className="font-sans text-base text-slate">Metadata is no longer available</p>
        : query.isError || status === "failed" ? <div className="space-y-4">
          <p className="font-sans text-base text-slate">Metadata could not be loaded.</p>
          <button type="button" onClick={refetch} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>Retry</button>
        </div> : <>
          {status === "pending" && <div className="space-y-4">
            <p className="font-sans text-base text-slate">Metadata is being prepared.</p>
            <button type="button" onClick={refetch} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>Refresh</button>
          </div>}
          {status === "empty" && <p className="font-sans text-base text-slate">No camera metadata was found.</p>}
          {status === "unavailable" && <p className="font-sans text-base text-slate">Metadata is unavailable for this photo.</p>}
          {status === "ready" && <>
            {section("Camera", <>{row("Model", cameraModel(fields))}{row("Lens", fields?.lens)}</>)}
            {section("Exposure", <>
              {row("Aperture", fields?.aperture == null ? null : `f/${fields.aperture}`, true)}
              {row("Exposure", fields?.exposureTime, true)}
              {row("ISO", fields?.iso, true)}
              {row("Flash", fields?.flash)}
              {row("Exposure mode", fields?.exposureMode)}
              {row("White balance", fields?.whiteBalance)}
            </>)}
            {section("Attribution and capture", <>
              {row("Author", fields?.author)}
              {row("Copyright", fields?.copyright)}
              {row("Date", fields?.takenAtLocal ? `${fields.takenAtLocal.replace("T", " ")}${fields.takenAtOffset ? ` ${fields.takenAtOffset}` : ""}` : null, true)}
            </>)}
          </>}
          {fileFacts && section("File", <>
            {row("Name", data?.originalFilename)}
            <div className="grid gap-1 py-3">
              <dt className="font-sans text-sm font-semibold text-slate">Original size</dt>
              <dd className="min-w-0 break-words font-mono text-base tnum text-ink">
                {data?.fileSizeBytes == null ? fallback : <span>
                  {(data.fileSizeBytes / 1_048_576).toFixed(1)} MiB
                  <span className="sr-only"> ({data.fileSizeBytes.toLocaleString("en-US")} bytes)</span>
                </span>}
              </dd>
            </div>
          </>)}
        </>}
    </div>
  </Modal>;
}
