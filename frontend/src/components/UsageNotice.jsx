const LIMITATIONS = [
  {
    title: "Probabilities are not medical certainty",
    body: "Model confidence, prediction margin, predictive entropy, and probability divergence describe how this ensemble behaved on this image. They are uncalibrated.",
  },
  {
    title: "Grad-CAM shows attribution, not cause",
    body: "The heatmaps mark regions a network was sensitive to when producing its output. They are not evidence of pathology and do not localize a finding.",
  },
  {
    title: "Disagreement is a signal to inspect",
    body: "When base models disagree, their predictions differ on this image. Inspect the individual probabilities and attribution maps before interpreting the ensemble output.",
  },
];

export default function UsageNotice() {
  return (
    <section className="panel p-6 sm:p-7" aria-labelledby="usage-notice-heading">
      <h2 id="usage-notice-heading" className="headline text-lg text-teal-deep">
        Reading these results
      </h2>

      <div className="mt-5 grid gap-5 sm:grid-cols-3">
        {LIMITATIONS.map((item) => (
          <div key={item.title} className="border-t-2 border-mint pt-3">
            <h3 className="text-[0.88rem] font-semibold text-ink">{item.title}</h3>
            <p className="mt-1.5 text-[0.8rem] leading-relaxed text-ink-soft">{item.body}</p>
          </div>
        ))}
      </div>

      <p className="mt-6 border-t border-line-soft pt-4 text-[0.82rem] leading-relaxed text-ink-faint">
        For research and educational use only. Not intended for clinical diagnosis or
        medical decision-making.
      </p>
    </section>
  );
}
