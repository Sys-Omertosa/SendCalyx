import { useId } from "react";
import { useReducedMotion } from "framer-motion";

/*
  Top-down water surface.

  Read as looking straight down at the sea: broad swells drifting from the upper
  left toward the lower right, with finer chop crossing them at a different angle
  and speed, and slow scattered glints for depth.

  Purely decorative. Renders above the host's own background and below its
  content, and is inert to the pointer so nothing underneath loses
  interactivity. Translucent whites and teals carry the palette, so whatever
  sits beneath still drives the hue.
*/

export default function WaterSurface({ className = "" }) {
  const reduceMotion = useReducedMotion();
  const animated = reduceMotion ? "" : "is-animated";

  // Several instances can be mounted at once, so the filter id must be unique
  // per instance. A shared id would make every surface reference whichever
  // definition happened to be in the document first.
  const filterId = `water-warp-${useId().replace(/:/g, "")}`;

  return (
    <span
      aria-hidden="true"
      className={`water-root pointer-events-none absolute inset-0 overflow-hidden ${className}`}
    >
      <span className="water-surface" style={{ "--water-filter": `url(#${filterId})` }}>
        {/* Turbulence warps the otherwise straight crests into organic swells,
            so the surface reads as water rather than diagonal stripes. */}
        <svg className="water-defs" aria-hidden="true" focusable="false">
          <defs>
            <filter id={filterId} x="-20%" y="-20%" width="140%" height="140%">
              <feTurbulence
                type="fractalNoise"
                baseFrequency="0.011 0.021"
                numOctaves="3"
                seed="7"
                result="noise"
              />
              <feDisplacementMap
                in="SourceGraphic"
                in2="noise"
                scale="46"
                xChannelSelector="R"
                yChannelSelector="G"
              />
            </filter>
          </defs>
        </svg>

        {/* Each layer is a wide repeating gradient rotated so the crests travel
            diagonally. Stacking them at different scales and speeds keeps the
            motion from reading as one sliding pattern. */}
        <span className={`water-layer water-swell ${animated}`} />
        <span className={`water-layer water-chop ${animated}`} />
        <span className={`water-layer water-glint ${animated}`} />
      </span>
    </span>
  );
}
