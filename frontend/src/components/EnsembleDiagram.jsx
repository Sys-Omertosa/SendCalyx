import { motion, useReducedMotion } from "framer-motion";

const MODELS = [
  { id: "inception_v3", label: "InceptionV3", features: "2048 features", y: 58 },
  { id: "inception_resnet_v2", label: "InceptionResNetV2", features: "1536 features", y: 140 },
  { id: "xception", label: "Xception", features: "2048 features", y: 222 },
];

// Each backbone routes into the stacking node at the same point. The paths stop
// just short of the node so every connector, including the middle one, stays
// visible against the box edge rather than merging into it.
const JOIN_X = 300;
const JOIN_Y = 140;
const START_X = 196;
const GAP = 6;

function path(fromY) {
  const endX = JOIN_X - GAP;
  const midX = (START_X + endX) / 2;
  return `M ${START_X} ${fromY} C ${midX} ${fromY}, ${midX} ${JOIN_Y}, ${endX} ${JOIN_Y}`;
}

/**
 * Structural diagram of the pipeline: three backbones converge on the stacked
 * meta-learner, which splits into the two things SendCalyx surfaces. Conceptual
 * only, and carries no prediction values.
 */
export default function EnsembleDiagram() {
  const reduceMotion = useReducedMotion();

  return (
    <div className="relative w-full">
      <svg
        viewBox="0 0 660 280"
        className="w-full"
        role="img"
        aria-label="Three CNN backbones, InceptionV3, InceptionResNetV2 and Xception, feed class probabilities into a stacked meta-learner, which produces a consensus reading and attribution maps."
      >
        <defs>
          {/* gradientUnits="userSpaceOnUse" is required: with the default
              objectBoundingBox, a perfectly horizontal path has a zero-height
              box and the gradient has no area to paint, so the stroke renders
              invisible. That silently hid the middle connector. */}
          <linearGradient
            id="flow"
            gradientUnits="userSpaceOnUse"
            x1={START_X}
            y1="0"
            x2={JOIN_X}
            y2="0"
          >
            <stop offset="0%" stopColor="#67c9b5" />
            <stop offset="100%" stopColor="#007e79" />
          </linearGradient>
          <linearGradient
            id="flowOut"
            gradientUnits="userSpaceOnUse"
            x1="424"
            y1="0"
            x2="480"
            y2="0"
          >
            <stop offset="0%" stopColor="#007e79" />
            <stop offset="100%" stopColor="#f0b4d0" />
          </linearGradient>
        </defs>

        {/* Backbone rows */}
        {MODELS.map((model, index) => (
          <g key={model.id}>
            <rect
              x="8"
              y={model.y - 24}
              width="188"
              height="48"
              rx="12"
              fill="#ffffff"
              stroke="#d8e7e3"
            />
            <text x="24" y={model.y - 3} fontSize="13.5" fontWeight="600" fill="#06302f">
              {model.label}
            </text>
            <text x="24" y={model.y + 14} fontSize="11.5" fill="#6f8b89">
              {model.features}
            </text>

            <motion.path
              d={path(model.y)}
              fill="none"
              stroke="url(#flow)"
              strokeWidth="1.75"
              strokeLinecap="round"
              initial={reduceMotion ? false : { pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 1 }}
              transition={{ duration: 0.9, delay: 0.35 + index * 0.12, ease: "easeOut" }}
            />

            {/* One faint signal travelling toward the meta-learner, staggered so
                the three lines never pulse together. */}
            {!reduceMotion && (
              <path
                className="signal-pulse"
                d={path(model.y)}
                fill="none"
                stroke="var(--color-mint)"
                strokeWidth="2.75"
                strokeLinecap="round"
                pathLength="1"
                style={{ animationDelay: `${2.2 + index * 1.6}s` }}
              />
            )}
          </g>
        ))}

        {/* Junction where the three probability vectors meet. Makes the middle
            connector legible, since its path is straight while the outer two
            curve into the same point. */}
        <motion.circle
          cx={JOIN_X - GAP}
          cy={JOIN_Y}
          r="4"
          fill="var(--color-teal)"
          initial={reduceMotion ? false : { opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.35, delay: 0.95 }}
          style={{ transformOrigin: `${JOIN_X - GAP}px ${JOIN_Y}px` }}
        />

        {/* Stacked meta-learner */}
        <motion.g
          initial={reduceMotion ? false : { opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.85 }}
          style={{ transformOrigin: `${JOIN_X + 62}px ${JOIN_Y}px` }}
        >
          <rect
            x={JOIN_X}
            y={JOIN_Y - 30}
            width="124"
            height="60"
            rx="14"
            fill="#01524f"
          />
          <text
            x={JOIN_X + 62}
            y={JOIN_Y - 6}
            fontSize="12.5"
            fontWeight="600"
            fill="#ffffff"
            textAnchor="middle"
          >
            Stacked
          </text>
          <text
            x={JOIN_X + 62}
            y={JOIN_Y + 11}
            fontSize="12.5"
            fontWeight="600"
            fill="#9bdcc9"
            textAnchor="middle"
          >
            meta-learner
          </text>
        </motion.g>

        {/* Two outputs */}
        {[
          { y: 92, label: "Consensus", sub: "agreement, margin, entropy" },
          { y: 188, label: "Attribution", sub: "overlap and divergence" },
        ].map((out, index) => (
          <g key={out.label}>
            <motion.path
              d={`M 424 ${JOIN_Y} C 452 ${JOIN_Y}, 452 ${out.y}, 480 ${out.y}`}
              fill="none"
              stroke="url(#flowOut)"
              strokeWidth="1.75"
              strokeLinecap="round"
              initial={reduceMotion ? false : { pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 1 }}
              transition={{ duration: 0.7, delay: 1.15 + index * 0.1, ease: "easeOut" }}
            />
            <motion.g
              initial={reduceMotion ? false : { opacity: 0, x: 8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.45, delay: 1.35 + index * 0.1 }}
            >
              <circle cx="484" cy={out.y} r="4" fill="#007e79" />
              <text x="496" y={out.y - 2} fontSize="12.5" fontWeight="600" fill="#06302f">
                {out.label}
              </text>
              <text x="496" y={out.y + 13} fontSize="11" fill="#6f8b89">
                {out.sub}
              </text>
            </motion.g>
          </g>
        ))}
      </svg>
    </div>
  );
}
