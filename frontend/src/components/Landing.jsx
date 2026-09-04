import { motion, useReducedMotion } from "framer-motion";
import { ArrowDown, Info, Layers, Scan, Sigma, Upload } from "lucide-react";
import EnsembleDiagram from "./EnsembleDiagram.jsx";
import InputGuide from "./InputGuide.jsx";
import WaterSurface from "./WaterSurface.jsx";

const rise = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] } },
};

/**
 * Scroll-in reveal.
 *
 * The reveal only starts once the element has been observed in the viewport.
 * `reveal` in the stylesheet re-asserts full opacity after a short delay, so a
 * section that is never observed (print, a stalled script, an offscreen
 * capture) still ends up visible rather than stranded at opacity 0.
 */
function Reveal({ children, className = "", delay = 0 }) {
  const reduceMotion = useReducedMotion();
  if (reduceMotion) return <div className={className}>{children}</div>;
  return (
    <motion.div
      className={`reveal ${className}`}
      variants={rise}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, amount: 0.15, margin: "0px 0px -8% 0px" }}
      transition={{ delay }}
    >
      {children}
    </motion.div>
  );
}

const STEPS = [
  {
    icon: Upload,
    title: "Load a slice",
    body: "A single axial or coronal kidney CT image, resized to the 299 x 299 model input.",
  },
  {
    icon: Layers,
    title: "Run three CNNs",
    body: "InceptionV3, InceptionResNetV2, and Xception each classify the slice independently.",
  },
  {
    icon: Sigma,
    title: "Stack probabilities",
    body: "Six class probabilities feed a trained meta-learner that produces the ensemble output.",
  },
  {
    icon: Scan,
    title: "Inspect the result",
    body: "Compare per-model votes, divergence, and Grad-CAM attribution side by side.",
  },
];

const SIGNALS = [
  {
    name: "Base-model agreement",
    body: "How many backbones selected the same class, and which one dissented.",
  },
  {
    name: "Prediction margin",
    body: "The gap between the two ensemble class probabilities.",
  },
  {
    name: "Predictive entropy",
    body: "Normalized binary entropy of the ensemble output, from one-sided to evenly split.",
  },
  {
    name: "Probability divergence",
    body: "Mean pairwise Jensen-Shannon divergence across base-model class distributions.",
  },
  {
    name: "Attribution consensus",
    body: "Pixel-wise mean of the available Grad-CAM maps.",
  },
  {
    name: "Attribution disagreement",
    body: "Pixel-wise variance across those same maps.",
  },
];

export default function Landing({ children }) {
  const reduceMotion = useReducedMotion();

  return (
    <main id="top">
      {/* ── Hero ─────────────────────────────────────────────────────── */}
      {/* Header height is 4.75rem; the hero takes the rest of the first
          viewport so the two read as one composition. */}
      <section className="relative flex flex-col overflow-hidden lg:min-h-[calc(100svh-4.75rem)]">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 -z-10"
          style={{
            background:
              "radial-gradient(46rem 30rem at 88% 4%, rgba(249,210,228,0.85), transparent 60%)," +
              "radial-gradient(40rem 30rem at 10% 96%, rgba(103,201,181,0.55), transparent 62%)," +
              "radial-gradient(56rem 34rem at 50% 120%, rgba(0,126,121,0.22), transparent 68%)",
          }}
        />

        {/* Faint concentric contours, drifting very slowly. Reads as imaging
            atmosphere rather than decoration; static under reduced motion. */}
        <div aria-hidden="true" className="pointer-events-none absolute inset-0 -z-10">
          <svg
            className="hero-contours h-full w-full"
            viewBox="0 0 1200 800"
            preserveAspectRatio="xMidYMid slice"
          >
            <g fill="none" stroke="var(--color-teal)" strokeWidth="1">
              {[0, 1, 2, 3, 4, 5, 6].map((ring) => (
                <ellipse
                  key={ring}
                  cx="880"
                  cy="330"
                  rx={120 + ring * 68}
                  ry={92 + ring * 52}
                  opacity={0.07 - ring * 0.008}
                />
              ))}
            </g>
            <g fill="none" stroke="var(--color-mint)" strokeWidth="1">
              {[0, 1, 2, 3, 4].map((ring) => (
                <ellipse
                  key={ring}
                  cx="180"
                  cy="640"
                  rx={140 + ring * 84}
                  ry={104 + ring * 62}
                  opacity={0.075 - ring * 0.012}
                />
              ))}
            </g>
          </svg>
          <span className="hero-sweep" />
        </div>

        {/* Water across the hero, above the gradient and contours, below the
            content. Faint so the headline stays the focus. */}
        <WaterSurface className="water-ambient -z-10" />

        <div className="shell grid flex-1 items-center gap-12 pb-10 pt-10 lg:grid-cols-[minmax(0,1.02fr)_minmax(0,1.06fr)] lg:gap-14 lg:pb-0 lg:pt-0">
          <motion.div
            initial={reduceMotion ? false : "hidden"}
            animate="show"
            variants={{ show: { transition: { staggerChildren: 0.08, delayChildren: 0.05 } } }}
          >
            {/* The brand mark lives in the header only. Repeating it here read
                as duplication in the first viewport. */}
            <motion.h1
              variants={rise}
              className="display max-w-[15ch] text-[clamp(2.3rem,5.2vw,3.7rem)] text-ink"
            >
              Inspect the ensemble, not just the answer.
            </motion.h1>

            <motion.p
              variants={rise}
              className="mt-6 max-w-[54ch] text-[1.05rem] leading-relaxed text-ink-soft"
            >
              Analyze kidney CT imagery across three CNN backbones, compare their
              predictions, and inspect where their visual attributions align or diverge.
            </motion.p>

            <motion.div variants={rise} className="mt-9 flex flex-wrap items-center gap-3">
              <a href="#analyze" className="btn-primary">
                Analyze a CT image
              </a>
              <a href="#workflow" className="btn-secondary">
                Explore the workflow
                <ArrowDown className="h-4 w-4" aria-hidden="true" />
              </a>
            </motion.div>
          </motion.div>

          <motion.div
            initial={reduceMotion ? false : { opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
            className="rounded-panel border border-line bg-surface/75 p-6 backdrop-blur-sm sm:p-8"
          >
            <EnsembleDiagram />
            <p className="mt-5 border-t border-line-soft pt-4 text-[0.8rem] leading-relaxed text-ink-faint">
              Every analysis exposes each stage of this pipeline, not only its final
              output.
            </p>
          </motion.div>
        </div>

        <div className="shell flex shrink-0 justify-center pb-7 lg:pb-8">
          <a
            href="#workflow"
            className="group inline-flex flex-col items-center gap-2 rounded-lg px-3 py-1 text-[0.78rem] font-medium text-ink-faint transition-colors hover:text-teal-deep"
          >
            Scroll to explore
            <span
              aria-hidden="true"
              className="scroll-cue flex h-6 w-6 items-center justify-center rounded-full border border-line bg-surface/70 text-teal"
            >
              <ArrowDown className="h-3.5 w-3.5" />
            </span>
          </a>
        </div>
      </section>

      {/* ── A. Workflow: horizontal process ──────────────────────────── */}
      <section id="workflow" className="border-t border-line-soft bg-surface/55 py-20 sm:py-24">
        <div className="shell">
          <Reveal>
            <p className="section-tag">Workflow</p>
            <h2 className="display mt-4 max-w-[18ch] text-[clamp(1.7rem,3.4vw,2.5rem)] text-ink">
              From CT slice to inspectable ensemble output
            </h2>
          </Reveal>

          <ol className="mt-12 grid gap-px overflow-hidden rounded-panel border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
            {STEPS.map((step, index) => (
              <li key={step.title} className="bg-surface p-6">
                <Reveal delay={index * 0.06}>
                  <div className="flex items-center gap-3">
                    <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-mint-pale text-teal-deep">
                      <step.icon className="h-4.5 w-4.5" aria-hidden="true" />
                    </span>
                    <span className="figure text-[0.78rem] text-ink-faint">
                      Step {index + 1}
                    </span>
                  </div>
                  <h3 className="headline mt-4 text-[1.02rem] text-ink">{step.title}</h3>
                  <p className="mt-2 text-[0.88rem] leading-relaxed text-ink-soft">
                    {step.body}
                  </p>
                </Reveal>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* ── B. Ensemble: asymmetric visual plus text ─────────────────── */}
      <section id="ensemble" className="py-20 sm:py-24">
        <div className="shell grid min-w-0 items-center gap-12 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] lg:gap-20">
          <Reveal>
            <p className="section-tag">Ensemble</p>
            <h2 className="display mt-4 text-[clamp(1.7rem,3.4vw,2.5rem)] text-ink">
              Three models. One inspectable decision.
            </h2>
            <p className="mt-5 max-w-[48ch] text-[0.98rem] leading-relaxed text-ink-soft">
              Each backbone is a frozen feature extractor with its own classifier head.
              Their class probabilities, six values in total, feed a probability-level
              stacked meta-learner rather than a simple average.
            </p>
            <p className="mt-4 max-w-[48ch] text-[0.98rem] leading-relaxed text-ink-soft">
              Most classifiers collapse that structure into a single number. SendCalyx
              keeps every member visible, so a confident ensemble built on a split vote
              looks different from a confident ensemble built on unanimity.
            </p>
          </Reveal>

          <Reveal delay={0.08}>
            <div className="overflow-hidden rounded-panel border border-line bg-surface">
              <table className="w-full text-left">
                <caption className="sr-only">Base model feature dimensions</caption>
                <thead>
                  <tr className="border-b border-line text-[0.78rem] font-semibold text-ink-faint">
                    <th scope="col" className="px-5 py-3 font-semibold">
                      Backbone
                    </th>
                    <th scope="col" className="px-5 py-3 font-semibold">
                      Feature dimension
                    </th>
                    <th scope="col" className="px-5 py-3 font-semibold">
                      Classifier head
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line-soft">
                  {[
                    ["InceptionV3", "2048"],
                    ["InceptionResNetV2", "1536"],
                    ["Xception", "2048"],
                  ].map(([name, dim]) => (
                    <tr key={name}>
                      <th scope="row" className="px-5 py-4 text-[0.92rem] font-semibold text-ink">
                        {name}
                      </th>
                      <td className="figure px-5 py-4 text-[0.88rem] text-ink-soft">{dim}</td>
                      <td className="figure px-5 py-4 text-[0.82rem] text-ink-soft">
                        256 &rarr; 128 &rarr; 2
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="border-t border-line bg-mint-pale/40 px-5 py-4">
                <p className="text-[0.82rem] text-ink-soft">
                  Meta-learner input{" "}
                  <span className="figure text-teal-deep">6 &rarr; 512 &rarr; 128 &rarr; 2</span>
                </p>
              </div>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ── C. Explainability: staged attribution concept ────────────── */}
      <section
        id="explainability"
        className="border-y border-line-soft bg-surface/55 py-20 sm:py-24"
      >
        <div className="shell">
          <Reveal className="max-w-[52ch]">
            <p className="section-tag">Explainability</p>
            <h2 className="display mt-4 text-[clamp(1.7rem,3.4vw,2.5rem)] text-ink">
              See where the models agree
            </h2>
            <p className="mt-5 text-[0.98rem] leading-relaxed text-ink-soft">
              Grad-CAM runs per backbone at its last spatial feature map. Those maps are
              then combined two ways, so overlap and divergence are separable rather than
              averaged into one picture.
            </p>
          </Reveal>

          <div className="mt-12 grid gap-5 md:grid-cols-3">
            {[
              {
                title: "Individual attribution",
                body: "One map per backbone, for the class that model selected.",
                bars: [0.35, 0.8, 0.5],
                tint: "var(--color-mint)",
              },
              {
                title: "Attribution consensus",
                body: "Pixel-wise mean across the available maps.",
                bars: [0.5, 0.92, 0.62],
                tint: "var(--color-teal)",
              },
              {
                title: "Attribution disagreement",
                body: "Pixel-wise variance across those same maps.",
                bars: [0.72, 0.28, 0.85],
                tint: "var(--color-pink-mid)",
              },
            ].map((card, index) => (
              <Reveal key={card.title} delay={index * 0.07}>
                <div className="h-full rounded-panel border border-line bg-surface p-6">
                  <div
                    className="flex h-28 items-end gap-2 rounded-card p-4"
                    style={{ background: "var(--color-mint-pale)" }}
                    aria-hidden="true"
                  >
                    {card.bars.map((value, barIndex) => (
                      <span
                        key={barIndex}
                        className="flex-1 rounded-t-md"
                        style={{ height: `${value * 100}%`, background: card.tint }}
                      />
                    ))}
                  </div>
                  <h3 className="headline mt-5 text-[1.02rem] text-ink">{card.title}</h3>
                  <p className="mt-2 text-[0.88rem] leading-relaxed text-ink-soft">
                    {card.body}
                  </p>
                </div>
              </Reveal>
            ))}
          </div>

          <Reveal delay={0.1}>
            <p className="mt-8 max-w-[62ch] text-[0.85rem] leading-relaxed text-ink-faint">
              Attribution maps mark regions a network was sensitive to when producing its
              output. They are not causal explanations and do not localize a finding.
            </p>
          </Reveal>
        </div>
      </section>

      {/* ── D. Signals: editorial definition list ────────────────────── */}
      <section id="signals" className="py-20 sm:py-24">
        <div className="shell grid gap-12 lg:grid-cols-[minmax(0,0.72fr)_minmax(0,1.28fr)] lg:gap-16">
          <Reveal>
            <p className="section-tag">Signals</p>
            <h2 className="display mt-4 text-[clamp(1.7rem,3.4vw,2.5rem)] text-ink">
              What every analysis reports
            </h2>
            <p className="mt-5 max-w-[40ch] text-[0.98rem] leading-relaxed text-ink-soft">
              Deterministic diagnostics computed from the same forward pass. They describe
              model behavior on one image, not calibrated confidence.
            </p>
          </Reveal>

          <Reveal delay={0.08}>
            <dl className="divide-y divide-line-soft border-t border-line-soft">
              {SIGNALS.map((signal) => (
                <div
                  key={signal.name}
                  className="grid gap-1.5 py-5 sm:grid-cols-[minmax(0,14rem)_minmax(0,1fr)] sm:gap-6"
                >
                  <dt className="text-[0.95rem] font-semibold text-ink">{signal.name}</dt>
                  <dd className="text-[0.9rem] leading-relaxed text-ink-soft">
                    {signal.body}
                  </dd>
                </div>
              ))}
            </dl>
          </Reveal>
        </div>
      </section>

      {/* ── E. Input guide ──────────────────────────────────────────── */}
      <InputGuide />

      {/* ── F. Analyze: the working intake ──────────────────────────── */}
      <section id="analyze" className="border-t border-line-soft py-20 sm:py-24">
        <div className="shell">
          <Reveal>
            <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
              <div className="max-w-[46ch]">
                <p className="section-tag">Analyze</p>
                <h2 className="display mt-4 text-[clamp(1.7rem,3.4vw,2.5rem)] text-ink">
                  Run an image through the ensemble
                </h2>
              </div>

              <a
                href="#input-guide"
                className="inline-flex items-center gap-2.5 rounded-full border border-line bg-surface/80 px-4 py-2.5 text-[0.85rem] text-ink-soft transition-colors hover:border-mint hover:text-teal-deep"
              >
                <Info className="h-4 w-4 shrink-0 text-teal" aria-hidden="true" />
                For guidance on suitable images, see the input guide above
              </a>
            </div>
          </Reveal>

          <Reveal delay={0.06}>
            <div className="mt-10">{children}</div>
          </Reveal>
        </div>
      </section>
    </main>
  );
}
