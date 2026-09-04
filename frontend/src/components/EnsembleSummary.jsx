import { motion, useReducedMotion } from "framer-motion";
import { formatClassName, formatPercent } from "../utils/api.js";

function ProbabilityRow({ label, value, emphasis, delay = 0 }) {
  const reduceMotion = useReducedMotion();
  const pct = Math.max(0, Math.min(1, value ?? 0)) * 100;
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm font-medium text-ink">{label}</span>
        <span className="figure text-sm text-ink-soft">{formatPercent(value)}</span>
      </div>
      <div className="mt-1.5 h-2.5 overflow-hidden rounded-full bg-line-soft">
        {/* Bars grow to their final width once, which makes the split between
            the two classes easier to read at a glance. */}
        <motion.div
          className="h-full rounded-full"
          initial={reduceMotion ? false : { width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] }}
          style={{
            background: emphasis
              ? "linear-gradient(90deg, var(--color-mint) 0%, var(--color-teal) 100%)"
              : "var(--color-mint-mid)",
          }}
        />
      </div>
    </div>
  );
}

/**
 * The headline conclusion: what the meta-learner predicted and how the base
 * models voted, kept together so the main finding is not split across cards.
 */
export default function EnsembleSummary({ ensemble, consensus }) {
  const isStone = ensemble?.prediction === "Kidney_stone";
  const probabilities = ensemble?.probabilities ?? {};
  const disagrees = Boolean(consensus?.disagreement_flag);

  return (
    <section className="panel p-6 sm:p-7" aria-labelledby="ensemble-summary-heading">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <h2 id="ensemble-summary-heading" className="headline text-lg text-teal-deep">
          Ensemble result
        </h2>
        <span
          className="inline-flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-[0.78rem] font-semibold"
          style={{
            borderColor: disagrees ? "var(--color-pink-mid)" : "var(--color-mint)",
            background: disagrees ? "rgba(249,210,228,0.5)" : "rgba(220,241,234,0.8)",
            color: disagrees ? "var(--color-pink-deep)" : "var(--color-teal-deep)",
          }}
        >
          {disagrees
            ? `Split vote ${consensus.majority_votes}/${consensus.num_models}`
            : `Unanimous ${consensus?.majority_votes ?? 0}/${consensus?.num_models ?? 0}`}
        </span>
      </div>

      <div
        className="mt-5 rounded-card border p-5"
        style={{
          borderColor: isStone ? "var(--color-pink-mid)" : "var(--color-mint)",
          background: isStone
            ? "linear-gradient(135deg, rgba(249,210,228,0.55), rgba(255,255,255,0.85))"
            : "linear-gradient(135deg, rgba(220,241,234,0.85), rgba(255,255,255,0.9))",
        }}
      >
        <p className="text-[0.8rem] text-ink-soft">Ensemble prediction</p>
        <p className="display mt-1 text-[clamp(1.8rem,3.8vw,2.4rem)] text-teal-dark">
          {formatClassName(ensemble?.prediction)}
        </p>
        <p className="figure mt-2 text-sm text-ink-soft">
          Model confidence {formatPercent(ensemble?.confidence)}
        </p>
      </div>

      <div className="mt-6 space-y-4">
        <ProbabilityRow
          label="Kidney stone"
          value={probabilities.Kidney_stone}
          emphasis={isStone}
          delay={0.12}
        />
        <ProbabilityRow
          label="Normal"
          value={probabilities.Normal}
          emphasis={!isStone}
          delay={0.2}
        />
      </div>
    </section>
  );
}
