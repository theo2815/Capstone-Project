import type { CSSProperties, ReactNode } from "react";

const mono = "var(--font-mono), ui-monospace, monospace";
const sans = "var(--font-sans), system-ui, sans-serif";

export function AppBar({
  title,
  back = true,
  trailing,
}: {
  title: string;
  back?: boolean;
  trailing?: ReactNode;
}) {
  return (
    <div
      style={{
        height: 48,
        borderBottom: "1.5px solid #000",
        display: "flex",
        alignItems: "center",
        padding: "0 14px",
        gap: 12,
        boxSizing: "border-box",
        background: "#fff",
      }}
    >
      {back && (
        <span
          aria-hidden
          style={{
            width: 9,
            height: 9,
            borderLeft: "1.5px solid #000",
            borderBottom: "1.5px solid #000",
            transform: "rotate(45deg)",
          }}
        />
      )}
      <span
        style={{
          fontFamily: mono,
          fontSize: 12,
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          color: "#000",
          flex: 1,
        }}
      >
        {title}
      </span>
      {trailing}
    </div>
  );
}

export function Kicker({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <span
      style={{
        fontFamily: mono,
        fontSize: 9,
        letterSpacing: "0.22em",
        textTransform: "uppercase",
        color: "#666",
        ...style,
      }}
    >
      {children}
    </span>
  );
}

export function Body({
  children,
  size = 12,
  weight = 400,
  color = "#000",
  style,
}: {
  children: ReactNode;
  size?: number;
  weight?: number;
  color?: string;
  style?: CSSProperties;
}) {
  return (
    <span
      style={{
        fontFamily: sans,
        fontSize: size,
        fontWeight: weight,
        color,
        ...style,
      }}
    >
      {children}
    </span>
  );
}

export function MonoNum({
  children,
  size = 14,
  color = "#000",
  style,
}: {
  children: ReactNode;
  size?: number;
  color?: string;
  style?: CSSProperties;
}) {
  return (
    <span
      style={{
        fontFamily: mono,
        fontSize: size,
        fontVariantNumeric: "tabular-nums",
        color,
        ...style,
      }}
    >
      {children}
    </span>
  );
}

export function Hatch({
  width = "100%",
  height = "100%",
  density = 5,
  style,
  label,
}: {
  width?: number | string;
  height?: number | string;
  density?: number;
  style?: CSSProperties;
  label?: string;
}) {
  return (
    <div
      style={{
        position: "relative",
        width,
        height,
        border: "1px solid #000",
        boxSizing: "border-box",
        backgroundImage: `repeating-linear-gradient(45deg, transparent 0 ${density - 0.6}px, #000 ${density - 0.6}px ${density}px)`,
        backgroundColor: "#fff",
        ...style,
      }}
    >
      {/* lighten the hatch so labels stay readable */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "#fff",
          opacity: 0.78,
        }}
      />
      {label && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontFamily: mono,
            fontSize: 9,
            letterSpacing: "0.22em",
            textTransform: "uppercase",
            color: "#000",
          }}
        >
          {label}
        </div>
      )}
    </div>
  );
}

export function Chip({
  children,
  filled = false,
  style,
}: {
  children: ReactNode;
  filled?: boolean;
  style?: CSSProperties;
}) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        padding: "3px 8px",
        border: "1px solid #000",
        borderRadius: 999,
        background: filled ? "#000" : "#fff",
        color: filled ? "#fff" : "#000",
        fontFamily: mono,
        fontSize: 9,
        letterSpacing: "0.18em",
        textTransform: "uppercase",
        fontVariantNumeric: "tabular-nums",
        ...style,
      }}
    >
      {children}
    </span>
  );
}

export function Button({
  children,
  variant = "primary",
  fullWidth = true,
  style,
}: {
  children: ReactNode;
  variant?: "primary" | "secondary" | "ghost";
  fullWidth?: boolean;
  style?: CSSProperties;
}) {
  const base: CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    height: 44,
    padding: "0 16px",
    width: fullWidth ? "100%" : undefined,
    fontFamily: mono,
    fontSize: 12,
    letterSpacing: "0.18em",
    textTransform: "uppercase",
    border: "1.5px solid #000",
    boxSizing: "border-box",
    cursor: "default",
  };
  const variants: Record<string, CSSProperties> = {
    primary: { background: "#000", color: "#fff" },
    secondary: { background: "#fff", color: "#000" },
    ghost: { background: "#fff", color: "#000", border: "1px dashed #000" },
  };
  return <div style={{ ...base, ...variants[variant], ...style }}>{children}</div>;
}

export function Divider({ style }: { style?: CSSProperties }) {
  return (
    <div
      style={{
        height: 1,
        background: "#000",
        opacity: 0.6,
        ...style,
      }}
    />
  );
}

export function Frame({
  children,
  padding = 16,
  style,
}: {
  children: ReactNode;
  padding?: number | string;
  style?: CSSProperties;
}) {
  return (
    <div
      style={{
        border: "1px solid #000",
        padding,
        boxSizing: "border-box",
        background: "#fff",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function Caret({
  direction = "right",
  size = 8,
}: {
  direction?: "left" | "right" | "up" | "down";
  size?: number;
}) {
  const rot = {
    left: 135,
    right: -45,
    up: 45,
    down: -135,
  }[direction];
  return (
    <span
      aria-hidden
      style={{
        display: "inline-block",
        width: size,
        height: size,
        borderRight: "1.5px solid #000",
        borderBottom: "1.5px solid #000",
        transform: `rotate(${rot}deg)`,
      }}
    />
  );
}
