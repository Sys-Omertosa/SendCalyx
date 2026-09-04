import { useEffect, useState } from "react";
import { RotateCcw } from "lucide-react";

const NAV_LINKS = [
  { id: "workflow", label: "Overview" },
  { id: "ensemble", label: "Ensemble" },
  { id: "explainability", label: "Explainability" },
  { id: "signals", label: "Signals" },
];

/**
 * Brand mark. `compact` uses a stroke-thickened variant that stays legible at
 * header and footer sizes, where the original hairline art disappears.
 */
export function Logo({ className = "", disk = true, compact = true }) {
  const mark = (
    <img
      src={compact ? "/sendcalyx-mark.png" : "/sendcalyx-logo.png"}
      alt=""
      aria-hidden="true"
      className={disk ? "h-[76%] w-[76%] object-contain" : "h-full w-full object-contain"}
      draggable="false"
    />
  );

  if (!disk) return <span className={className}>{mark}</span>;

  return (
    <span
      className={`inline-flex items-center justify-center rounded-full bg-white ring-1 ring-line ${className}`}
    >
      {mark}
    </span>
  );
}

/**
 * Sticky application bar. Carries the brand in both states; the landing state
 * adds section navigation, the analysis state swaps it for a reset action.
 */
export default function Header({ mode = "landing", onReset, onAnalyzeClick }) {
  const [lifted, setLifted] = useState(false);

  useEffect(() => {
    const onScroll = () => setLifted(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const isLanding = mode === "landing";

  return (
    <header
      className={`sticky top-0 z-50 transition-colors duration-300 ${
        lifted
          ? "border-b border-line bg-paper/80 backdrop-blur-xl"
          : "border-b border-transparent bg-transparent"
      }`}
    >
      {/* 4.75rem tall; the hero subtracts exactly this to fill the viewport. */}
      <div className="shell flex h-[4.75rem] items-center gap-4">
        <a
          href="#top"
          className="flex shrink-0 items-center gap-2.5"
          aria-label="SendCalyx home"
        >
          <Logo className="h-8 w-8 sm:h-9 sm:w-9" />
          <span className="wordmark text-[1.1rem] text-teal-deep sm:text-[1.35rem]">
            SendCalyx
          </span>
        </a>

        {isLanding && (
          <nav className="ml-auto hidden items-center gap-1 lg:flex" aria-label="Sections">
            {NAV_LINKS.map((link) => (
              <a
                key={link.id}
                href={`#${link.id}`}
                className="rounded-full px-3.5 py-2 text-[0.88rem] font-medium text-ink-soft transition-colors hover:bg-mint-pale/70 hover:text-teal-deep"
              >
                {link.label}
              </a>
            ))}
          </nav>
        )}

        <div className={`flex items-center gap-2 ${isLanding ? "lg:ml-4 ml-auto" : "ml-auto"}`}>
          {isLanding ? (
            <a
              href="#analyze"
              onClick={onAnalyzeClick}
              className="btn-primary shrink-0 px-4 py-2 text-[0.84rem] sm:px-5 sm:text-[0.88rem]"
            >
              Analyze CT
            </a>
          ) : (
            <button
              type="button"
              onClick={onReset}
              className="btn-primary shrink-0 px-4 py-2 text-[0.88rem] sm:px-5"
            >
              <RotateCcw className="h-4 w-4" aria-hidden="true" />
              New analysis
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
