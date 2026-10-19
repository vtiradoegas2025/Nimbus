import { Component, type ErrorInfo, type ReactNode } from "react";
import "./ErrorBoundary.css";

interface ErrorBoundaryProps {
  children: ReactNode;
  /** Short label for what failed, shown in the fallback (e.g. "This plot"). */
  label?: string;
  /** Clears the error and re-renders children when any value changes; allows auto-recovery on navigation. */
  resetKeys?: ReadonlyArray<unknown>;
}

interface ErrorBoundaryState {
  error: Error | null;
}

/** Isolates render/lifecycle failures to a single panel fallback. Must be a class component (no Hook equivalent). */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Surface in the console for debugging; the UI shows the message itself.
    console.error("[ErrorBoundary] caught render error:", error, info.componentStack);
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps): void {
    if (this.state.error && resetKeysChanged(prevProps.resetKeys, this.props.resetKeys)) {
      this.setState({ error: null });
    }
  }

  private handleRetry = (): void => {
    this.setState({ error: null });
  };

  render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;

    const title = this.props.label ? `${this.props.label} failed` : "Something went wrong";
    return (
      <div className="error-boundary" role="alert">
        <div className="error-boundary__card">
          <h3 className="error-boundary__title">{title}</h3>
          <p className="error-boundary__message">{error.message || String(error)}</p>
          <p className="error-boundary__hint">
            Other views are still available. Try again or switch views.
            For plots, you can also choose another field or frame.
          </p>
          <button
            type="button"
            className="btn btn--secondary"
            onClick={this.handleRetry}
          >
            Try again
          </button>
        </div>
      </div>
    );
  }
}

/** Shallow per-element comparison of two resetKeys arrays. A length or
 *  identity change in any slot (or one array being absent) counts as changed. */
function resetKeysChanged(
  prev: ReadonlyArray<unknown> | undefined,
  next: ReadonlyArray<unknown> | undefined,
): boolean {
  if (prev === next) return false;
  if (!prev || !next || prev.length !== next.length) return true;
  for (let i = 0; i < prev.length; i++) {
    if (!Object.is(prev[i], next[i])) return true;
  }
  return false;
}
