import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, ImageUp, Play, RefreshCw, Trash2 } from "lucide-react";
import { ACCEPTED_TYPES, formatBytes, validateImageFile } from "../utils/api.js";

/**
 * Image intake: drag and drop, file picker, preview, and the analyze action.
 * Purely functional. The surrounding page owns all brand and product copy.
 */
export default function UploadWorkspace({
  file,
  preview,
  onSelect,
  onClear,
  onAnalyze,
  isAnalyzing,
}) {
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
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)] lg:items-center lg:gap-14">
      <div>
        {preview ? (
          <div className="rounded-panel border border-line bg-surface p-4">
            <div className="overflow-hidden rounded-card bg-teal-dark">
              <img
                src={preview}
                alt="Selected CT slice, ready to analyze"
                className="mx-auto max-h-[26rem] w-full object-contain"
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
                    Running ensemble
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" aria-hidden="true" />
                    Analyze image
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
            className={`relative flex w-full flex-col items-center justify-center rounded-panel px-6 py-16 text-center transition-shadow ${
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

            {/* Keeps the white copy legible where the gradient is lightest. */}
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

      <div className="lg:pl-2">
        <h3 className="headline text-[1.05rem] text-ink">What you get back</h3>
        <ul className="mt-4 space-y-3">
          {[
            "Ensemble prediction with class probabilities",
            "Per-model votes, confidence, and probability divergence",
            "Grad-CAM attribution for each backbone",
            "Consensus and disagreement attribution maps",
          ].map((item) => (
            <li key={item} className="flex gap-3 text-[0.92rem] leading-relaxed text-ink-soft">
              <span
                aria-hidden="true"
                className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-mint"
              />
              {item}
            </li>
          ))}
        </ul>

        <p className="mt-6 border-t border-line-soft pt-5 text-[0.82rem] leading-relaxed text-ink-faint">
          PNG, JPEG, or WebP up to 10 MB. Images are analyzed in memory and are not stored.
        </p>
      </div>
    </div>
  );
}
