const LIMITATIONS = [
  {
    title: "Probabilities are not medical certainty",
    body: "Model confidence, prediction margin, and predictive entropy describe how this ensemble behaved on this image. They are uncalibrated and say nothing about whether a prediction is medically correct.",
  },
  {
    title: "Grad-CAM shows attribution, not cause",
    body: "The heatmaps mark regions the network was sensitive to when producing its output. They are not evidence of pathology and do not localise a stone.",
  },
  {
    title: "Disagreement is a model-behaviour signal",
    body: "When base models split, the ensemble sits on a region of input space where its members generalise differently. That is informative about the models, not about the patient.",
  },
];

export default function ResearchNotice() {
  return (
    <section className="panel p-6 sm:p-7" aria-labelledby="research-notice-heading">
      <h2 id="research-notice-heading" className="headline text-lg text-teal-deep">
        Research limitations
      </h2>

      <p className="mt-3 rounded-card border border-pink-mid bg-pink/40 px-4 py-3 text-[0.88rem] font-medium leading-relaxed text-pink-deep">
        SendCalyx is a research and educational prototype and is not intended for clinical
        diagnosis or medical decision-making.
      </p>

      <div className="mt-5 grid gap-5 sm:grid-cols-3">
        {LIMITATIONS.map((item) => (
          <div key={item.title} className="border-t-2 border-mint pt-3">
            <h3 className="text-[0.88rem] font-semibold text-ink">{item.title}</h3>
            <p className="mt-1.5 text-[0.8rem] leading-relaxed text-ink-soft">{item.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
