import { formatClassName, formatPercent } from "../utils/api.js";

function Bar({ value, tone }) {
  const pct = Math.max(0, Math.min(1, value ?? 0)) * 100;
  return (
    <div className="flex items-center gap-2.5">
      <div className="h-2 w-full min-w-[3.5rem] overflow-hidden rounded-full bg-line-soft">
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            background:
              tone === "stone"
                ? "linear-gradient(90deg, var(--color-mint) 0%, var(--color-teal) 100%)"
                : "var(--color-mint-mid)",
          }}
        />
      </div>
      <span className="figure w-14 shrink-0 text-right text-[0.8rem] text-ink-soft">
        {formatPercent(value)}
      </span>
    </div>
  );
}

/**
 * Side-by-side class probabilities for every base model, so a divergent member
 * is visible at a glance.
 */
export default function ModelComparison({ individualModels, consensus }) {
  const models = Object.entries(individualModels ?? {});
  if (models.length === 0) return null;

  return (
    <section className="panel p-6 sm:p-7" aria-labelledby="model-comparison-heading">
      <h2 id="model-comparison-heading" className="headline text-lg text-teal-deep">
        Model comparison
      </h2>
      <p className="mt-1 text-[0.82rem] text-ink-faint">
        Class probabilities from each base CNN, before the meta-learner combines them.
      </p>

      <div className="mt-5 overflow-x-auto">
        <table className="w-full min-w-[42rem] border-collapse text-left">
          <thead>
            <tr className="border-b border-line text-[0.78rem] font-semibold text-ink-faint">
              <th scope="col" className="py-2.5 pr-4 font-semibold">
                Model
              </th>
              <th scope="col" className="py-2.5 pr-4 font-semibold">
                Prediction
              </th>
              <th scope="col" className="w-[26%] py-2.5 pr-4 font-semibold">
                Kidney stone
              </th>
              <th scope="col" className="w-[26%] py-2.5 pr-4 font-semibold">
                Normal
              </th>
              <th scope="col" className="py-2.5 text-right font-semibold">
                Confidence
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line-soft">
            {models.map(([id, model]) => {
              const dissents =
                consensus?.majority_class && model.prediction !== consensus.majority_class;
              return (
                <tr key={id} className={dissents ? "bg-pink/25" : undefined}>
                  <th scope="row" className="py-3.5 pr-4 text-sm font-semibold text-ink">
                    {model.display_name ?? id}
                    {dissents && (
                      <span className="mt-0.5 block text-[0.72rem] font-medium text-pink-deep">
                        Differs from majority
                      </span>
                    )}
                  </th>
                  <td className="py-3.5 pr-4 text-sm text-ink-soft">
                    {formatClassName(model.prediction)}
                  </td>
                  <td className="py-3.5 pr-4">
                    <Bar value={model.probabilities?.Kidney_stone} tone="stone" />
                  </td>
                  <td className="py-3.5 pr-4">
                    <Bar value={model.probabilities?.Normal} tone="normal" />
                  </td>
                  <td className="figure py-3.5 text-right text-sm font-semibold text-teal-deep">
                    {formatPercent(model.confidence)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="mt-4 text-[0.75rem] leading-snug text-ink-faint">
        Confidence is the probability a model assigned to the class it selected. It is not
        a measure of diagnostic certainty.
      </p>
    </section>
  );
}
