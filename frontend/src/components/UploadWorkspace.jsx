import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, ImageUp, Play, RefreshCw, Trash2 } from "lucide-react";
import { ResearchBadge } from "./Header.jsx";
import { ACCEPTED_TYPES, formatBytes, validateImageFile } from "../utils/api.js";

const CAPABILITIES = [
  "Three-CNN stacked ensemble",
  "Base-model agreement",
  "Predictive entropy",
  "Per-model Grad-CAM",
  "Attribution consensus",
];

// Anchored to the intake surface itself so they read as members of the
// ensemble waiting on an image, not as loose decoration.
const MODEL_PILLS = [
  { name: "InceptionV3", className: "-left-4 top-8" },
  { name: "InceptionResNetV2", className: "-right-4 top-1/2 -translate-y-1/2" },
  { name: "Xception", className: "-left-2 bottom-9" },
];

/**
 * Landing state and image intake. This is the one surface where the gradient
 * identity is used at full strength; once results exist the app steps back and
 * lets the analysis lead.
 */
export default function UploadWorkspace({ file, preview, onSelect, onClear, onAnalyze, isAnalyzing }) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);
  const dragDepth = useRef(0);

  useEffect(() => {
    if (file) setError("");
  }, [file]);

  const accept = useCallback(
    (candidate) => {
      const { valid, error: reason } = validateImageFile(candidate);
      if (!valid) {
        setError(reason);
        return;
      }
      setError("");
      onSelect(candidate);
    },
    [onSelect],
  );

  const onDrop = (event) => {
    event.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    const dropped = event.dataTransfer?.files?.[0];
    if (dropped) accept(dropped);
  };

  const onDragEnter = (event) => {
    event.preventDefault();
    dragDepth.current += 1;
    setDragging(true);
  };

  const onDragLeave = (event) => {
    event.preventDefault();
    dragDepth.current -= 1;
    if (dragDepth.current <= 0) setDragging(false);
  };

  return (
    <div className="mx-auto flex w-full max-w-[86rem] flex-col justify-center px-4 py-6 sm:px-8 sm:py-10 lg:min-h-screen">
      <div className="panel overflow-hidden shadow-[0_28px_70px_-40px_rgba(1,82,79,0.45)]">
        <div className="grid gap-12 p-6 sm:p-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:items-center lg:gap-20 lg:p-14">
          {/* Brand and proposition */}
          <div className="flex flex-col">
            <ResearchBadge className="self-start" />

            <h1 className="wordmark mt-5 text-[clamp(2.8rem,6.6vw,4.4rem)] text-teal-deep">
              SendCalyx
            </h1>

            <p className="headline mt-5 max-w-[22ch] text-[clamp(1.3rem,2.4vw,1.8rem)] leading-[1.15] text-ink">
              Explainable ensemble intelligence for kidney CT imaging
            </p>

            <p className="mt-5 max-w-[52ch] text-[0.98rem] leading-relaxed text-ink-soft">
              Inspect predictions from a multi-CNN kidney CT ensemble, quantify model
              consensus, and explore where visual attributions converge or disagree.
            </p>

            <ul className="mt-8 flex flex-wrap gap-2">
              {CAPABILITIES.map((item) => (
                <li
                  key={item}
                  className="rounded-full border border-line bg-mint-pale/60 px-3.5 py-1.5 text-[0.78rem] font-medium text-teal-deep"
                >
                  {item}
                </li>
              ))}
            </ul>

            <p className="mt-8 max-w-[52ch] text-[0.8rem] leading-relaxed text-ink-faint">
              PNG, JPEG, or WebP up to 10 MB. Images are analysed in memory and are not
              stored. SendCalyx is a research and educational prototype and is not
              intended for clinical diagnosis or medical decision-making.
            </p>
          </div>

          {/* Intake */}
          <div className="relative">
            {!preview && (
              <div className="pointer-events-none absolute inset-0 z-20 hidden lg:block">
                {MODEL_PILLS.map((pill) => (
                  <span
                    key={pill.name}
                    className={`absolute rounded-full border border-line bg-surface px-3.5 py-1.5 text-[0.76rem] font-semibold text-teal-deep shadow-[0_10px_24px_-14px_rgba(1,82,79,0.6)] ${pill.className}`}
                  >
                    {pill.name}
                  </span>
                ))}
              </div>
            )}

            {preview ? (
              <div className="rounded-panel border border-line bg-mint-pale/40 p-4">
                <div className="overflow-hidden rounded-card bg-teal-dark">
                  <img
                    src={preview}
                    alt="Selected CT slice, ready to analyse"
                    className="mx-auto max-h-[24rem] w-full object-contain"
                  />
                </div>

                <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-2">
                  <p className="min-w-0 flex-1 truncate text-sm font-medium text-ink">
                    {file?.name}
                  </p>
                  <p className="figure text-xs text-ink-faint">{formatBytes(file?.size)}</p>
                </div>

                <div className="mt-4 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={onAnalyze}
                    disabled={isAnalyzing}
                    className="inline-flex flex-1 items-center justify-center gap-2 rounded-full bg-teal px-5 py-3 text-sm font-semibold text-white transition-colors hover:bg-teal-deep disabled:cursor-not-allowed disabled:bg-ink-faint"
                  >
                    {isAnalyzing ? (
                      <>
                        <RefreshCw className="h-4 w-4 animate-spin" aria-hidden="true" />
                        Running ensemble…
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4" aria-hidden="true" />
                        Analyse image
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={() => inputRef.current?.click()}
                    disabled={isAnalyzing}
                    className="inline-flex items-center gap-2 rounded-full border border-line bg-surface px-4 py-3 text-sm font-semibold text-teal-deep transition-colors hover:border-mint disabled:opacity-50"
                  >
                    Replace
                  </button>

                  <button
                    type="button"
                    onClick={onClear}
                    disabled={isAnalyzing}
                    className="inline-flex items-center gap-2 rounded-full border border-line bg-surface px-4 py-3 text-sm font-semibold text-ink-soft transition-colors hover:border-pink-mid hover:text-pink-deep disabled:opacity-50"
                  >
                    <Trash2 className="h-4 w-4" aria-hidden="true" />
                    Remove
                  </button>
                </div>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                onDrop={onDrop}
                onDragOver={(event) => event.preventDefault()}
                onDragEnter={onDragEnter}
                onDragLeave={onDragLeave}
                className={`relative z-10 flex w-full flex-col items-center justify-center rounded-panel px-6 py-16 text-center transition-shadow ${
                  dragging
                    ? "shadow-[0_0_0_3px_var(--color-teal)]"
                    : "shadow-[0_0_0_1px_var(--color-line)]"
                }`}
                style={{
                  background:
                    "linear-gradient(152deg, #f9d2e4 0%, #9bdcc9 24%, #3aa79c 54%, #01524f 100%)",
                }}
              >
                <span className="pointer-events-none absolute inset-3 rounded-[1.15rem] border-2 border-dashed border-white/55" />

                {/* Keeps the white copy legible where the gradient is at its
                    lightest, without flattening the gradient itself. */}
                <span
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-x-8 inset-y-16 rounded-full"
                  style={{
                    background:
                      "radial-gradient(closest-side, rgba(1,49,47,0.58), rgba(1,49,47,0) 100%)",
                  }}
                />

                <span className="relative flex h-14 w-14 items-center justify-center rounded-2xl bg-white/95 text-teal-deep shadow-[0_12px_30px_-16px_rgba(1,49,47,0.9)]">
                  <ImageUp className="h-6 w-6" aria-hidden="true" />
                </span>

                <span className="relative mt-5 block text-lg font-semibold text-white">
                  {dragging ? "Drop to load the slice" : "Drop a kidney CT slice here"}
                </span>
                <span className="relative mt-1.5 block text-sm text-white/95">
                  or click to browse your files
                </span>

                <span className="relative mt-7 inline-flex items-center gap-2 rounded-full bg-teal-dark px-5 py-2.5 text-sm font-semibold text-white">
                  Choose image
                </span>
              </button>
            )}

            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED_TYPES.join(",")}
              className="sr-only"
              onChange={(event) => {
                const selected = event.target.files?.[0];
                if (selected) accept(selected);
                event.target.value = "";
              }}
            />

            {error && (
              <p
                role="alert"
                className="mt-4 flex items-start gap-2 rounded-card border border-pink-mid bg-pink/40 px-4 py-3 text-sm text-pink-deep"
              >
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                {error}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
