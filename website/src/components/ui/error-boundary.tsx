"use client";

import { Component, type ErrorInfo, type ReactNode } from "react";
import { unstable_rethrow } from "next/navigation";
import { Kicker } from "@/components/ui/kicker";

// React still requires a class component for componentDidCatch — function
// components have no equivalent hook. This boundary catches throws from any
// render/lifecycle inside its subtree and shows a Quiet Studio fallback that
// keeps the photographer in the rail (so they can navigate away without a
// hard reload).
//
// Reset model: bumping `resetKey` inside the boundary clears `error` and
// re-renders children. The boundary itself doesn't know what failed, so it
// trusts the user to want to retry; pages that need to reset external state
// (debounced saves, in-flight uploads) can pass `onReset` to wire that up.
//
// Mounting rule: per-page wrap (inside DashboardShell) — a layout-level wrap
// would blank the rail, which is the dashboard's identity. See website
// Slice 4 plan in the vault for the rationale.

interface ErrorBoundaryProps {
  children: ReactNode;
  /** Optional callback fired when the user clicks Try again. Use to clear
   *  external state that contributed to the error. */
  onReset?: () => void;
  /** Optional override for the fallback heading. Defaults to a generic copy
   *  appropriate for any photographer surface. */
  fallbackTitle?: string;
  /** Optional override for the fallback body copy. */
  fallbackBody?: string;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    // notFound() and redirect() work by throwing. A React boundary catches
    // every throw, so without this the dashboard layout's boundary swallowed
    // them and rendered "Something broke on this panel." instead of the 404
    // — every notFound() under /dashboard/* was silently broken.
    unstable_rethrow(error);
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    unstable_rethrow(error);
    // TODO(backend): pipe to Sentry / equivalent once observability lands.
    // Until then, surface to the dev console so the trace isn't swallowed.
    if (process.env.NODE_ENV !== "production") {
      console.error("[ErrorBoundary]", error, info);
    }
  }

  handleReset = (): void => {
    this.props.onReset?.();
    this.setState({ error: null });
  };

  render(): ReactNode {
    if (!this.state.error) return this.props.children;

    const title = this.props.fallbackTitle ?? "Something broke on this panel.";
    const body =
      this.props.fallbackBody ??
      "Reload the panel to try again. If it keeps happening, the surrounding pages are still safe to use.";

    return (
      <div
        role="alert"
        className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center"
      >
        <Kicker as="p" className="mb-3">
          Something broke
        </Kicker>
        <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
          {title}
        </p>
        <p className="font-sans text-base text-ink-soft mt-3 max-w-md mx-auto">
          {body}
        </p>
        <button
          type="button"
          onClick={this.handleReset}
          className="mt-6 font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
        >
          Try again
        </button>
      </div>
    );
  }
}
