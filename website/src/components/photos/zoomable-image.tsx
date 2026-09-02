"use client";

import { PROTECTED_IMG_CLASS, PROTECTED_IMG_PROPS } from "@/lib/protected-image";
import { useEffect, useRef, useState, type PointerEvent, type WheelEvent } from "react";
import { cn } from "@/lib/utils";

// Pinch / ctrl-wheel / double-click zoom over the lightbox image, drag to pan
// while zoomed. Pointer Events cover mouse, trackpad and touch in one path;
// `touch-action: none` keeps the browser from page-zooming instead.
//
// What is zoomed is exactly what was served — the watermarked preview for a
// browser, the clean original only for an owner. Zoom never requests a
// different asset. Mount with `key={photo.id}` so state resets per photo.

const MAX_SCALE = 4;
const DOUBLE_CLICK_SCALE = 2.5;

interface ZoomableImageProps {
  src: string;
  alt: string;
  onLoad: () => void;
  onError: () => void;
  loaded: boolean;
}

export function ZoomableImage({ src, alt, onLoad, onError, loaded }: ZoomableImageProps) {
  const box = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const [animate, setAnimate] = useState(false);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const last = useRef<{ dist: number; cx: number; cy: number } | null>(null);

  const clamp = (x: number, y: number, s: number) => {
    const el = box.current;
    if (!el) return { x, y };
    const maxX = (el.clientWidth * (s - 1)) / 2;
    const maxY = (el.clientHeight * (s - 1)) / 2;
    return {
      x: Math.min(maxX, Math.max(-maxX, x)),
      y: Math.min(maxY, Math.max(-maxY, y)),
    };
  };

  // Scale to `next`, keeping the client point (px, py) visually fixed.
  const zoomAt = (next: number, px: number, py: number) => {
    const el = box.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const dx = px - (r.left + r.width / 2);
    const dy = py - (r.top + r.height / 2);
    const s = Math.min(MAX_SCALE, Math.max(1, next));
    const k = s / scale;
    const c = s === 1 ? { x: 0, y: 0 } : clamp(dx - (dx - tx) * k, dy - (dy - ty) * k, s);
    setScale(s);
    setTx(c.x);
    setTy(c.y);
  };

  const onDoubleClick = (e: React.MouseEvent) => {
    setAnimate(true);
    zoomAt(scale > 1.05 ? 1 : DOUBLE_CLICK_SCALE, e.clientX, e.clientY);
  };

  const onWheel = (e: WheelEvent) => {
    // Trackpad pinch arrives as wheel + ctrlKey; plain wheel scrolls nothing
    // here (the modal locks scroll), so treat both as zoom.
    e.preventDefault();
    setAnimate(false);
    zoomAt(scale * Math.exp(-e.deltaY / 300), e.clientX, e.clientY);
  };

  const onPointerDown = (e: PointerEvent) => {
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    last.current = null;
    setAnimate(false);
  };

  const onPointerMove = (e: PointerEvent) => {
    const p = pointers.current;
    if (!p.has(e.pointerId)) return;
    const prev = p.get(e.pointerId)!;
    p.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (p.size >= 2) {
      const [a, b] = [...p.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      const cx = (a.x + b.x) / 2;
      const cy = (a.y + b.y) / 2;
      if (last.current) {
        zoomAt(scale * (dist / last.current.dist), cx, cy);
        const c = clamp(tx + cx - last.current.cx, ty + cy - last.current.cy, scale);
        setTx(c.x);
        setTy(c.y);
      }
      last.current = { dist, cx, cy };
    } else if (scale > 1) {
      const c = clamp(tx + e.clientX - prev.x, ty + e.clientY - prev.y, scale);
      setTx(c.x);
      setTy(c.y);
    }
  };

  const onPointerUp = (e: PointerEvent) => {
    pointers.current.delete(e.pointerId);
    last.current = null;
    if (scale < 1.02) {
      setScale(1);
      setTx(0);
      setTy(0);
    }
  };

  // React attaches wheel listeners passively; preventDefault needs a native one.
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const stop = (e: globalThis.WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", stop, { passive: false });
    return () => el.removeEventListener("wheel", stop);
  }, []);

  const zoomed = scale > 1;

  return (
    <div
      ref={box}
      className={cn(
        "absolute inset-0 overflow-hidden touch-none select-none",
        zoomed ? "cursor-grab active:cursor-grabbing" : "cursor-zoom-in",
      )}
      onDoubleClick={onDoubleClick}
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={alt}
        onLoad={onLoad}
        onError={onError}
        style={{
          transform: `translate(${tx}px, ${ty}px) scale(${scale})`,
          transition: animate ? "transform 0.22s ease-out" : undefined,
        }}
        className={cn(
          "absolute inset-0 w-full h-full object-contain",
          PROTECTED_IMG_CLASS,
          loaded ? "opacity-100" : "opacity-0",
          !animate && "transition-opacity duration-500",
        )}
        {...PROTECTED_IMG_PROPS}
      />
      {!zoomed && loaded && (
        <p
          aria-hidden="true"
          className="absolute bottom-3 left-3 font-mono uppercase tracking-[0.18em] text-[12px] text-bone/60 pointer-events-none"
        >
          Pinch or double-click to zoom
        </p>
      )}
    </div>
  );
}
