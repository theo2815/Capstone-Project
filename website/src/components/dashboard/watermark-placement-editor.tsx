"use client";

import {
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
  type PointerEvent,
} from "react";
import { Kicker } from "@/components/ui/kicker";
import { cn } from "@/lib/utils";
import {
  OPACITY_MAX,
  OPACITY_MIN,
  SCALE_MAX,
  SCALE_MIN,
  WATERMARK_PREVIEW_SIZE,
  positionFromBox,
  watermarkBox,
  type Orientation,
} from "@/lib/watermark-placement";
import type { WatermarkPlacement } from "@/store/photographer-settings-store";

// Drag-to-position watermark editor. The photographer drags their logo on a
// landscape and a portrait preview (each orientation keeps its own position),
// and tunes size + opacity with sliders. The preview runs the exact same
// free-space formula the backend bakes with (lib/watermark-placement +
// WatermarkService.watermarkBox), scaled from a representative photo.

interface Props {
  dataUrl: string;
  editing: boolean;
  placement: WatermarkPlacement;
  onChange: (next: WatermarkPlacement) => void;
}

const ORIENTATIONS: { key: Orientation; label: string; wrap: string }[] = [
  { key: "landscape", label: "Landscape", wrap: "w-full" },
  { key: "portrait", label: "Portrait", wrap: "w-2/3 max-w-[280px]" },
];

export function WatermarkPlacementEditor({ dataUrl, editing, placement, onChange }: Props) {
  const [orientation, setOrientation] = useState<Orientation>("landscape");
  const [logoAspect, setLogoAspect] = useState(3); // natural w/h, set on image load
  const [box, setBox] = useState({ w: 0, h: 0 });
  const boxRef = useRef<HTMLDivElement>(null);
  const grab = useRef<{ dx: number; dy: number } | null>(null);

  const pos = orientation === "landscape" ? placement.landscape : placement.portrait;
  const photo = WATERMARK_PREVIEW_SIZE[orientation];

  // Scale rendered-photo coordinates to the responsive preview. Applying the
  // compositor's minimums to this small DOM box would exaggerate the logo.
  useLayoutEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const measure = () => setBox({ w: el.clientWidth, h: el.clientHeight });
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [orientation]);

  const previewScale = box.w / photo.width;
  const logo = watermarkBox(photo.width, photo.height, logoAspect, placement.scale, pos.x, pos.y);

  const setPos = useCallback(
    (x: number, y: number) => {
      onChange(
        orientation === "landscape"
          ? { ...placement, landscape: { x, y } }
          : { ...placement, portrait: { x, y } },
      );
    },
    [onChange, orientation, placement],
  );

  const onPointerDown = (e: PointerEvent) => {
    if (!editing) return;
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    const el = boxRef.current!;
    const r = el.getBoundingClientRect();
    // Offset within the logo where it was grabbed, so it doesn't jump under the
    // cursor on the first move.
    grab.current = {
      dx: e.clientX - (r.left + el.clientLeft + logo.left * previewScale),
      dy: e.clientY - (r.top + el.clientTop + logo.top * previewScale),
    };
  };

  const onPointerMove = (e: PointerEvent) => {
    if (!editing || !grab.current || previewScale <= 0) return;
    const el = boxRef.current!;
    const r = el.getBoundingClientRect();
    const left = (e.clientX - r.left - el.clientLeft - grab.current.dx) / previewScale;
    const top = (e.clientY - r.top - el.clientTop - grab.current.dy) / previewScale;
    const next = positionFromBox(photo.width, photo.height, logoAspect, placement.scale, left, top, pos.x, pos.y);
    setPos(next.x, next.y);
  };

  const endDrag = (e: PointerEvent) => {
    grab.current = null;
    (e.currentTarget as HTMLElement).releasePointerCapture?.(e.pointerId);
  };

  return (
    <div className="space-y-5">
      <div className="flex gap-2" role="tablist" aria-label="Preview orientation">
        {ORIENTATIONS.map((o) => {
          const active = o.key === orientation;
          return (
            <button
              key={o.key}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => setOrientation(o.key)}
              className={cn(
                "rounded-full border px-4 py-2 font-sans text-sm font-medium transition-colors",
                active
                  ? "border-ink bg-fresh-tint/50 text-ink"
                  : "border-line text-slate hover:border-slate hover:text-ink",
              )}
            >
              {o.label}
            </button>
          );
        })}
      </div>

      <div className="flex justify-center">
        <div
          ref={boxRef}
          className={cn(
            "relative overflow-hidden rounded-2xl border border-line",
            ORIENTATIONS.find((o) => o.key === orientation)!.wrap,
          )}
          style={{
            aspectRatio: `${photo.width} / ${photo.height}`,
            background:
              "linear-gradient(135deg, var(--ink-soft) 0%, var(--slate) 45%, var(--bone-deep) 100%)",
          }}
        >
          {box.w > 0 && (
            /* eslint-disable-next-line @next/next/no-img-element -- data-URL preview mock; the backend composites the real watermark at upload time. */
            <img
              src={dataUrl}
              alt="Watermark position preview"
              draggable={false}
              onLoad={(e) => {
                const el = e.currentTarget;
                if (el.naturalWidth && el.naturalHeight) {
                  setLogoAspect(el.naturalWidth / el.naturalHeight);
                }
              }}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={endDrag}
              onPointerCancel={endDrag}
              className={cn("select-none", editing && "active:cursor-grabbing")}
              style={{
                position: "absolute",
                left: logo.left * previewScale,
                top: logo.top * previewScale,
                width: logo.width * previewScale,
                height: logo.height * previewScale,
                opacity: placement.opacity,
                touchAction: "none",
                cursor: editing ? "grab" : "default",
              }}
            />
          )}
          {editing && (
            <span className="pointer-events-none absolute left-2.5 top-2.5 rounded-full bg-ink/55 px-3 py-1 font-mono text-[12px] uppercase tracking-[0.14em] text-surface backdrop-blur-sm">
              Drag to position
            </span>
          )}
        </div>
      </div>

      <div className="space-y-4">
        <p className="text-sm text-slate">Sample photo. Tall logos shrink to fit inside the frame.</p>
        <SliderRow
          label="Horizontal position"
          value={pos.x}
          min={0}
          max={1}
          step={0.01}
          display={`${Math.round(pos.x * 100)}%`}
          valueText={`${orientation}, ${Math.round(pos.x * 100)}% from left`}
          disabled={!editing}
          onChange={(x) => setPos(x, pos.y)}
        />
        <SliderRow
          label="Vertical position"
          value={pos.y}
          min={0}
          max={1}
          step={0.01}
          display={`${Math.round(pos.y * 100)}%`}
          valueText={`${orientation}, ${Math.round(pos.y * 100)}% from top`}
          disabled={!editing}
          onChange={(y) => setPos(pos.x, y)}
        />
        <SliderRow
          label="Size"
          value={placement.scale}
          min={SCALE_MIN}
          max={SCALE_MAX}
          step={0.01}
          display={`${Math.round(placement.scale * 100)}% of width`}
          disabled={!editing}
          onChange={(scale) => onChange({ ...placement, scale })}
        />
        <SliderRow
          label="Opacity"
          value={placement.opacity}
          min={OPACITY_MIN}
          max={OPACITY_MAX}
          step={0.01}
          display={`${Math.round(placement.opacity * 100)}%`}
          disabled={!editing}
          onChange={(opacity) => onChange({ ...placement, opacity })}
        />
      </div>
    </div>
  );
}

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  display,
  valueText = display,
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  valueText?: string;
  disabled: boolean;
  onChange: (v: number) => void;
}) {
  return (
    <label className={cn("block", disabled && "opacity-50")}>
      <span className="mb-2 flex items-baseline justify-between gap-3">
        <Kicker as="span">{label}</Kicker>
        <span className="font-mono text-sm text-ink tnum">{display}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full disabled:cursor-not-allowed"
        style={{ accentColor: "var(--ink)" }}
        aria-label={label}
        aria-valuetext={valueText}
      />
    </label>
  );
}
