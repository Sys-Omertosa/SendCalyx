import { RotateCcw } from "lucide-react";

function Mark({ className = "" }) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" className={className}>
      <rect width="32" height="32" rx="9" fill="var(--color-teal)" />
      <path
        d="M16 6c5 3.2 7.6 6.5 7.6 10.1 0 4.3-3.4 7.4-7.6 7.4s-7.6-3.1-7.6-7.4C8.4 12.5 11 9.2 16 6Z"
        fill="var(--color-mint)"
      />
      <circle cx="16" cy="17" r="3.1" fill="var(--color-pink)" />
    </svg>
  );
}

export function ResearchBadge({ className = "" }) {
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border border-pink-mid bg-pink/50 px-3 py-1.5 text-[0.7rem] font-semibold tracking-wide text-pink-deep ${className}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-pink-deep" aria-hidden="true" />
      Research prototype
    </span>
  );
}

/**
 * Compact application bar shown once an analysis exists. The landing state
 * carries its own, much larger, brand treatment.
 */
export default function Header({ onReset }) {
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-paper/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-[86rem] items-center gap-4 px-5 py-3 sm:px-8">
        <Mark className="h-8 w-8 shrink-0" />

        <div className="min-w-0">
          <p className="wordmark text-lg text-teal-deep sm:text-xl">SendCalyx</p>
          <p className="hidden truncate text-xs text-ink-faint sm:block">
            Explainable ensemble intelligence for kidney CT imaging
          </p>
        </div>

        <div className="ml-auto flex items-center gap-3">
          <ResearchBadge className="hidden sm:inline-flex" />
          {onReset && (
            <button
              type="button"
              onClick={onReset}
              className="inline-flex items-center gap-2 rounded-full bg-teal px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-teal-deep"
            >
              <RotateCcw className="h-4 w-4" aria-hidden="true" />
              New image
            </button>
          )}
        </div>
      </div>
    </header>
  );
}

export { Mark };
