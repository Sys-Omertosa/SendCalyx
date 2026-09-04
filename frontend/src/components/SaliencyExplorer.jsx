import { useMemo, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { EyeOff, Layers, ScanEye } from "lucide-react";
import WaterSurface from "./WaterSurface.jsx";

const VIEW_COPY = {
  original: "The uploaded slice, resized to the 299 × 299 model input.",
  consensus: "Mean Grad-CAM attribution across available base models.",
  disagreement: "Spatial variance in Grad-CAM attribution across available base models.",
};

/**
 * Selectable attribution views. Grad-CAM imagery keeps its own scientific
 * colormaps (jet for attribution strength, inferno for variance) rather than
 * being recoloured to match the interface.
 */
export default function SaliencyExplorer({
  preview,
  individualModels,
  xaiConsensus,
  consensus,
}) {
  const reduceMotion = useReducedMotion();

  const views = useMemo(() => {
    const list = [
      {
        id: "original",
        label: "Original",
        group: "source",
        src: preview ?? null,
        caption: VIEW_COPY.original,
      },
    ];

    Object.entries(individualModels ?? {}).forEach(([id, model]) => {
      list.push({
        id,
        label: model.display_name ?? id,
        group: "model",
        src: model.gradcam_overlay ? `data:image/png;base64,${model.gradcam_overlay}` : null,
        caption: `Grad-CAM attribution for ${model.display_name ?? id}, taken at its last spatial feature map for the class it selected.`,
        dissents:
          Boolean(consensus?.majority_class) && model.prediction !== consensus.majority_class,
      });
    });

    const included = xaiConsensus?.models_included ?? 0;
    list.push({
      id: "consensus",
      label: "Consensus",
      group: "cross",
      src: xaiConsensus?.mean_gradcam
        ? `data:image/png;base64,${xaiConsensus.mean_gradcam}`
        : null,
      caption: `${VIEW_COPY.consensus} Aggregated over ${included} model${included === 1 ? "" : "s"}. Warmer regions receive consistently stronger attribution.`,
    });
    list.push({
      id: "disagreement",
      label: "Disagreement",
      group: "cross",
      src: xaiConsensus?.variance_gradcam
        ? `data:image/png;base64,${xaiConsensus.variance_gradcam}`
        : null,
      caption: `${VIEW_COPY.disagreement} Aggregated over ${included} model${included === 1 ? "" : "s"}. Brighter regions are where the models attribute most differently.`,
    });

    return list;
  }, [preview, individualModels, xaiConsensus, consensus]);

  const [selectedId, setSelectedId] = useState("consensus");
  const selected = views.find((view) => view.id === selectedId) ?? views[0];

  return (
    <section className="panel overflow-hidden" aria-labelledby="saliency-heading">
      <div className="flex flex-wrap items-start justify-between gap-3 p-6 pb-0 sm:p-7 sm:pb-0">
        <div>
          <h2 id="saliency-heading" className="headline text-lg text-teal-deep">
            Explainability
          </h2>
          <p className="mt-1 text-[0.82rem] text-ink-faint">
            Where each network placed attribution, and where those maps agree.
          </p>
        </div>

        {xaiConsensus?.available ? (
          <span className="inline-flex items-center gap-2 rounded-full border border-mint bg-mint-pale/70 px-3.5 py-1.5 text-[0.78rem] font-semibold text-teal-deep">
            <Layers className="h-3.5 w-3.5" aria-hidden="true" />
            {xaiConsensus.models_included} maps aggregated
          </span>
        ) : (
          <span className="inline-flex items-center gap-2 rounded-full border border-line bg-paper px-3.5 py-1.5 text-[0.78rem] font-semibold text-ink-faint">
            <EyeOff className="h-3.5 w-3.5" aria-hidden="true" />
            Cross-model view unavailable
          </span>
        )}
      </div>

      {/* View selector */}
      <div
        role="tablist"
        aria-label="Attribution views"
        className="flex flex-wrap gap-2 p-6 pb-5 sm:px-7"
      >
        {views.map((view) => {
          const isSelected = view.id === selected?.id;
          const isMissing = !view.src;
          return (
            <button
              key={view.id}
              type="button"
              role="tab"
              aria-selected={isSelected}
              onClick={() => setSelectedId(view.id)}
              className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-[0.82rem] font-semibold transition-colors ${
                isSelected
                  ? "border-teal bg-teal text-white"
                  : "border-line bg-surface text-ink-soft hover:border-mint hover:text-teal-deep"
              }`}
            >
              {view.group === "cross" && (
                <ScanEye className="h-3.5 w-3.5" aria-hidden="true" />
              )}
              {view.label}
              {view.dissents && !isSelected && (
                <span className="h-1.5 w-1.5 rounded-full bg-pink-deep" aria-hidden="true" />
              )}
              {isMissing && (
                <span className={isSelected ? "text-white/80" : "text-ink-faint"}>
                  (none)
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Selected view */}
      <div className="relative overflow-hidden border-t border-line-soft p-6 sm:p-7">
        {/* Same gradient family as the hero, so the viewer surround belongs to
            the same visual world. */}
        <span
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 z-0"
          style={{
            background:
              "radial-gradient(46rem 30rem at 88% 4%, rgba(249,210,228,0.85), transparent 60%)," +
              "radial-gradient(40rem 30rem at 10% 96%, rgba(103,201,181,0.55), transparent 62%)," +
              "radial-gradient(56rem 34rem at 50% 120%, rgba(0,126,121,0.22), transparent 68%)",
          }}
        />

        {/* Water fills the viewer surround only. */}
        <WaterSurface className="water-ambient z-0" />

        {/* The attribution map sits above both decorative layers on an opaque
            card, so neither the gradient nor the water can tint the Grad-CAM.
            The explicit z-index keeps that true regardless of DOM order: these
            maps must stay exactly as the backend rendered them. */}
        <div className="relative z-10 mx-auto max-w-2xl">
          <div className="overflow-hidden rounded-card border border-line bg-teal-dark">
            {selected?.src ? (
              /* Cross-fade between views so attribution differences are easier
                 to compare than an instant swap. */
              <AnimatePresence mode="wait" initial={false}>
                <motion.img
                  key={selected.id}
                  src={selected.src}
                  alt={`${selected.label} view: ${selected.caption}`}
                  className="mx-auto block w-full object-contain"
                  initial={reduceMotion ? false : { opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={reduceMotion ? undefined : { opacity: 0 }}
                  transition={{ duration: 0.28, ease: "easeOut" }}
                />
              </AnimatePresence>
            ) : (
              <div className="flex aspect-square w-full flex-col items-center justify-center gap-3 px-6 text-center">
                <EyeOff className="h-7 w-7 text-mint" aria-hidden="true" />
                <p className="text-sm font-semibold text-white">Attribution unavailable</p>
                <p className="max-w-sm text-[0.8rem] leading-relaxed text-mint-mid">
                  {selected?.group === "cross"
                    ? (xaiConsensus?.message ??
                      "Cross-model attribution needs at least two valid Grad-CAM maps.")
                    : "This model did not return a usable attribution map for this image."}
                </p>
              </div>
            )}
          </div>

          <p className="mt-4 text-[0.85rem] leading-relaxed text-ink-soft">
            {selected?.caption}
          </p>

          {selected?.dissents && (
            <p className="mt-2 text-[0.8rem] font-medium text-pink-deep">
              This model selected a different class from the base-model majority.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
