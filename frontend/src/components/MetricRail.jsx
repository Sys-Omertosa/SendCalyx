import { formatNumber, formatPercent } from "../utils/api.js";

/**
 * Compact strip of the numeric diagnostics, kept in one row so they read as a
 * set of related measurements rather than five competing cards.
 */
export default function MetricRail({ consensus, processingTime }) {
  const divergence = consensus?.probability_divergence;

  const metrics = [
    {
      label: "Prediction margin",
      value: formatPercent(consensus?.ensemble_margin),
      hint: "Gap between the two class probabilities",
    },
    {
      label: "Predictive entropy",
      value: formatNumber(consensus?.ensemble_entropy, 3),
      hint: "0 one-sided, 1 evenly split",
    },
    {
      label: "Probability divergence",
      value: divergence?.available ? formatNumber(divergence.mean, 3) : "N/A",
      hint: "Mean pairwise Jensen-Shannon divergence across base-model class distributions",
    },
    {
      label: "Confidence spread",
      value: formatPercent(consensus?.confidence_spread),
      hint: "Highest minus lowest selected-class confidence",
    },
    {
      label: "Processing time",
      value: typeof processingTime === "number" ? `${processingTime.toFixed(2)}s` : "N/A",
      hint: "Inference plus attribution",
    },
  ];

  return (
    <section className="panel overflow-hidden" aria-label="Analysis diagnostics">
      <dl className="grid gap-px bg-line sm:grid-cols-2 lg:grid-cols-5">
        {metrics.map((metric) => (
          <div key={metric.label} className="bg-surface px-5 py-4">
            <dt className="text-[0.78rem] text-ink-faint">{metric.label}</dt>
            <dd className="figure mt-1 text-[1.15rem] font-semibold text-teal-deep">
              {metric.value}
            </dd>
            <dd className="mt-1 text-[0.72rem] leading-snug text-ink-faint">{metric.hint}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
