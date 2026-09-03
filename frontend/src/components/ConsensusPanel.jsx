import { Check, Split } from "lucide-react";
import { formatClassName, formatPercent } from "../utils/api.js";

/**
 * How the three base models voted, independent of what the meta-learner
 * concluded. Disagreement is a finding to read, not an error to fix.
 */
export default function ConsensusPanel({ consensus, individualModels }) {
  if (!consensus) return null;

  const models = Object.entries(individualModels ?? {});
  const disagrees = Boolean(consensus.disagreement_flag);
  const StatusIcon = disagrees ? Split : Check;

  return (
    <section className="panel p-6 sm:p-7" aria-labelledby="consensus-heading">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id="consensus-heading" className="headline text-lg text-teal-deep">
            Base-model votes
          </h2>
          <p className="mt-1 text-[0.82rem] text-ink-faint">
            Each CNN classifies the slice on its own before stacking.
          </p>
        </div>

        <span
          className="inline-flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-[0.78rem] font-semibold"
          style={{
            borderColor: disagrees ? "var(--color-pink-mid)" : "var(--color-mint)",
            background: disagrees ? "rgba(249,210,228,0.5)" : "rgba(220,241,234,0.8)",
            color: disagrees ? "var(--color-pink-deep)" : "var(--color-teal-deep)",
          }}
        >
          <StatusIcon className="h-3.5 w-3.5" aria-hidden="true" />
          {disagrees ? "Model disagreement" : "Unanimous"}
        </span>
      </div>

      <ul className="mt-5 divide-y divide-line-soft border-y border-line-soft">
        {models.map(([id, model]) => {
          const agreesWithMajority = model.prediction === consensus.majority_class;
          return (
            <li key={id} className="flex items-center gap-3 py-3">
              <span
                aria-hidden="true"
                className="h-6 w-1 shrink-0 rounded-full"
                style={{
                  background: agreesWithMajority
                    ? "var(--color-teal)"
                    : "var(--color-pink-mid)",
                }}
              />
              <span className="min-w-0 flex-1 truncate text-sm font-medium text-ink">
                {model.display_name ?? id}
              </span>
              <span className="text-sm text-ink-soft">
                {formatClassName(model.prediction)}
              </span>
              <span className="figure w-16 text-right text-sm font-semibold text-teal-deep">
                {formatPercent(model.confidence, 0)}
              </span>
            </li>
          );
        })}
      </ul>

      <dl className="mt-5 grid gap-3 sm:grid-cols-3">
        <div className="rounded-card border border-line-soft bg-mint-pale/45 px-4 py-3">
          <dt className="text-[0.78rem] text-ink-faint">Agreement</dt>
          <dd className="figure mt-1 text-lg font-semibold text-teal-deep">
            {consensus.majority_votes} / {consensus.num_models}
          </dd>
          <dd className="mt-0.5 text-[0.72rem] text-ink-faint">
            Ratio {consensus.agreement_ratio?.toFixed(2) ?? "—"}
          </dd>
        </div>

        <div className="rounded-card border border-line-soft bg-mint-pale/45 px-4 py-3">
          <dt className="text-[0.78rem] text-ink-faint">Confidence spread</dt>
          <dd className="figure mt-1 text-lg font-semibold text-teal-deep">
            {formatPercent(consensus.confidence_spread)}
          </dd>
          <dd className="mt-0.5 text-[0.72rem] leading-snug text-ink-faint">
            Highest minus lowest selected-class confidence
          </dd>
        </div>

        <div className="rounded-card border border-line-soft bg-mint-pale/45 px-4 py-3">
          <dt className="text-[0.78rem] text-ink-faint">Majority class</dt>
          <dd className="figure mt-1 text-lg font-semibold text-teal-deep">
            {formatClassName(consensus.majority_class)}
          </dd>
          <dd className="mt-0.5 text-[0.72rem] text-ink-faint">
            {consensus.ensemble_matches_majority
              ? "Ensemble agrees with the majority"
              : "Ensemble differs from the majority"}
          </dd>
        </div>
      </dl>

      {disagrees && (
        <p className="mt-4 rounded-card border border-pink-mid bg-pink/35 px-4 py-3 text-[0.8rem] leading-relaxed text-pink-deep">
          The base models did not all select the same class. That is a model-behaviour
          signal worth inspecting in the attribution maps below, not a failed analysis.
        </p>
      )}
    </section>
  );
}
