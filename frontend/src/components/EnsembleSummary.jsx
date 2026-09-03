import { Timer } from "lucide-react";
import { formatClassName, formatNumber, formatPercent } from "../utils/api.js";

function Readout({ label, value, hint }) {
  return (
    <div className="rounded-card border border-line-soft bg-mint-pale/45 px-4 py-3">
      <p className="text-[0.78rem] text-ink-faint">{label}</p>
      <p className="figure mt-1 text-lg font-semibold text-teal-deep">{value}</p>
      {hint && <p className="mt-0.5 text-[0.72rem] leading-snug text-ink-faint">{hint}</p>}
    </div>
  );
}

function ProbabilityRow({ label, value, tone }) {
  const pct = Math.max(0, Math.min(1, value ?? 0)) * 100;
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm font-medium text-ink">{label}</span>
        <span className="figure text-sm text-ink-soft">{formatPercent(value)}</span>
      </div>
      <div className="mt-1.5 h-2.5 overflow-hidden rounded-full bg-line-soft">
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            background:
              tone === "primary"
                ? "linear-gradient(90deg, var(--color-mint) 0%, var(--color-teal) 100%)"
                : "var(--color-mint-mid)",
          }}
        />
      </div>
    </div>
  );
}

/**
 * The stacked meta-learner's own output: what it predicted and how decisive
 * that output was.
 */
export default function EnsembleSummary({ ensemble, consensus, processingTime }) {
  const prediction = formatClassName(ensemble?.prediction);
  const isStone = ensemble?.prediction === "Kidney_stone";
  const probabilities = ensemble?.probabilities ?? {};
  const entropy = consensus?.ensemble_entropy;

  return (
    <section className="panel p-6 sm:p-7" aria-labelledby="ensemble-summary-heading">
      <h2 id="ensemble-summary-heading" className="headline text-lg text-teal-deep">
        Ensemble summary
      </h2>
      <p className="mt-1 text-[0.82rem] text-ink-faint">
        Output of the stacked meta-learner over three base CNNs.
      </p>

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
        <p className="headline mt-1 text-[clamp(1.7rem,3.6vw,2.3rem)] leading-tight text-teal-dark">
          {prediction}
        </p>
        <p className="figure mt-2 text-sm text-ink-soft">
          Model confidence {formatPercent(ensemble?.confidence)}
        </p>
      </div>

      <div className="mt-6 space-y-4">
        <ProbabilityRow
          label="Kidney stone"
          value={probabilities.Kidney_stone}
          tone={isStone ? "primary" : "muted"}
        />
        <ProbabilityRow
          label="Normal"
          value={probabilities.Normal}
          tone={isStone ? "muted" : "primary"}
        />
      </div>

      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        <Readout
          label="Prediction margin"
          value={formatPercent(consensus?.ensemble_margin)}
          hint="Gap between the two class probabilities"
        />
        <Readout
          label="Predictive entropy"
          value={formatNumber(entropy, 3)}
          hint="0 one-sided · 1 even split"
        />
        <Readout
          label="Processing time"
          value={
            typeof processingTime === "number" ? `${processingTime.toFixed(2)}s` : "—"
          }
          hint="Inference plus attribution"
        />
      </div>

      <p className="mt-4 flex items-start gap-2 text-[0.75rem] leading-snug text-ink-faint">
        <Timer className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        Margin and entropy describe how decisive this model output was. They are not
        calibrated uncertainty and not a probability that the prediction is correct.
      </p>
    </section>
  );
}
