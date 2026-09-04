import { Check, Info, X } from "lucide-react";

/*
  Illustrations are purpose-drawn schematics of image *formats*, not patient
  examples. No dataset imagery is used anywhere in this component.
*/

const FRAME = "#0d3b3a";
const TISSUE = "#8fa3a2";

function ScanFrame({ children, label }) {
  return (
    <svg viewBox="0 0 120 120" className="h-full w-full" role="img" aria-label={label}>
      <rect width="120" height="120" rx="6" fill={FRAME} />
      {children}
    </svg>
  );
}

/** A single focused grayscale slice: the shape the ensemble expects.
 *  Kidneys are drawn as bean forms with a concave hilum so the schematic is
 *  recognisable rather than reading as a pair of plain ellipses. */
function GoodInput() {
  return (
    <ScanFrame label="A single kidney CT slice filling the frame">
      {/* Coronal abdomen: torso silhouette, continuous vertical spine, and a
          pair of small kidneys flanking it. Kept low and modest in scale so the
          composition reads as anatomy rather than a face. */}
      <path
        d="M22 18c0-3 3-5 8-5h60c5 0 8 2 8 5v76c0 6-4 10-11 12-9 3-19 4-27 4s-18-1-27-4c-7-2-11-6-11-12Z"
        fill={TISSUE}
        opacity="0.3"
      />
      {/* continuous spinal column */}
      <rect x="56" y="20" width="8" height="76" rx="4" fill="#e8f1ee" opacity="0.5" />
      {[28, 40, 52, 64, 76, 88].map((y) => (
        <rect key={y} x="54" y={y} width="12" height="7" rx="3" fill="#e8f1ee" opacity="0.8" />
      ))}
      {/* kidneys: small, low, hilum facing the spine */}
      <g fill="#d5e3e2" opacity="0.95">
        <path d="M44 46c6 1 9 8 8 16-1 7-5 12-11 11-4-.5-7-4-6.5-9 .3-3 2.5-4 2.8-6 .3-2-1.3-4-.8-6.5.6-3.5 3.5-6 7.5-5.5Z" />
        <path d="M76 46c-6 1-9 8-8 16 1 7 5 12 11 11 4-.5 7-4 6.5-9-.3-3-2.5-4-2.8-6-.3-2 1.3-4 .8-6.5-.6-3.5-3.5-6-7.5-5.5Z" />
      </g>
      {/* ureters trailing toward the pelvis */}
      <path
        d="M47 71c1 8 2 13 5 19M73 71c-1 8-2 13-5 19"
        stroke="#d5e3e2"
        strokeWidth="1.6"
        strokeLinecap="round"
        fill="none"
        opacity="0.5"
      />
    </ScanFrame>
  );
}

/** Several slices tiled into one upload. */
function MultiPanel() {
  const tiles = [];
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 2; col += 1) {
      tiles.push(
        <g key={`${row}-${col}`}>
          <rect
            x={10 + col * 52}
            y={10 + row * 34}
            width="48"
            height="30"
            rx="3"
            fill="#17514f"
          />
          <ellipse
            cx={34 + col * 52}
            cy={25 + row * 34}
            rx="13"
            ry="9"
            fill={TISSUE}
            opacity="0.55"
          />
        </g>,
      );
    }
  }
  return <ScanFrame label="A montage of several CT slices in one image">{tiles}</ScanFrame>;
}

/** A slice buried under annotation and measurement overlays. */
function Annotated() {
  return (
    <ScanFrame label="A CT slice covered by text and measurement overlays">
      <ellipse cx="60" cy="62" rx="36" ry="42" fill={TISSUE} opacity="0.4" />
      <ellipse cx="44" cy="58" rx="11" ry="14" fill="#cfdedd" opacity="0.7" />
      <ellipse cx="78" cy="58" rx="11" ry="14" fill="#cfdedd" opacity="0.7" />
      {[16, 26, 36].map((y) => (
        <rect key={y} x="12" y={y} width={y === 26 ? 52 : 38} height="5" rx="2.5" fill="#f9d2e4" />
      ))}
      <path d="M30 92 L90 92" stroke="#f9d2e4" strokeWidth="2" strokeDasharray="5 4" />
      <rect x="70" y="100" width="34" height="6" rx="3" fill="#f9d2e4" />
      <path d="M26 70 L52 44" stroke="#f9d2e4" strokeWidth="2" />
    </ScanFrame>
  );
}

/** Anything that is not a CT slice at all. */
function NonScan() {
  return (
    <ScanFrame label="A document or non-CT image">
      <rect x="26" y="18" width="68" height="84" rx="5" fill="#e8f1ee" opacity="0.92" />
      {[30, 42, 54, 66, 78].map((y) => (
        <rect
          key={y}
          x="36"
          y={y}
          width={y === 78 ? 30 : 48}
          height="5"
          rx="2.5"
          fill={TISSUE}
          opacity="0.65"
        />
      ))}
      <circle cx="76" cy="88" r="9" fill="#f0b4d0" opacity="0.85" />
    </ScanFrame>
  );
}

const AVOID = [
  { Art: MultiPanel, title: "Multi-panel reports", body: "Collages or screenshots holding several slices at once." },
  { Art: Annotated, title: "Heavy overlays", body: "Large text, callouts, or measurement lines across the anatomy." },
  { Art: NonScan, title: "Non-CT imagery", body: "Documents, photos of equipment, or unrelated anatomy." },
];

const WORKS_BEST = [
  "One CT slice per image",
  "Axial or coronal kidney CT view",
  "Kidney anatomy clearly visible",
  "Standard grayscale scan appearance",
  "Minimal text, measurements, or annotations",
  "Sufficient source resolution",
];

export default function InputGuide() {
  return (
    <section
      id="input-guide"
      className="border-t border-line-soft bg-surface/55 py-20 sm:py-24"
    >
      <div className="shell">
        <div className="max-w-[54ch]">
          <p className="section-tag">Input guide</p>
          <h2 className="display mt-4 text-[clamp(1.7rem,3.4vw,2.5rem)] text-ink">
            Start with a clean CT slice
          </h2>
          <p className="mt-5 text-[0.98rem] leading-relaxed text-ink-soft">
            SendCalyx expects a single kidney CT image. Clean, focused inputs make the
            ensemble output and attribution maps easier to inspect.
          </p>
        </div>

        <div className="mt-12 grid items-start gap-6 lg:grid-cols-[minmax(0,0.86fr)_minmax(0,1.14fr)] lg:gap-8">
          {/* Works best */}
          <div className="h-full rounded-panel border-2 border-mint bg-mint-pale/35 p-6 sm:p-7">
            <div className="flex items-center gap-2.5">
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-teal text-white">
                <Check className="h-4 w-4" aria-hidden="true" />
              </span>
              <h3 className="headline text-[1.05rem] text-teal-deep">Works best</h3>
            </div>

            <div className="mt-5 mx-auto aspect-square w-full max-w-[16rem] overflow-hidden rounded-card border border-mint/60">
              <GoodInput />
            </div>

            <ul className="mt-5 space-y-2.5">
              {WORKS_BEST.map((item) => (
                <li key={item} className="flex gap-2.5 text-[0.88rem] leading-relaxed text-ink">
                  <Check
                    className="mt-0.5 h-4 w-4 shrink-0 text-teal"
                    aria-hidden="true"
                  />
                  {item}
                </li>
              ))}
            </ul>
          </div>

          {/* Usually avoid */}
          <div className="h-full rounded-panel border border-dashed border-pink-mid bg-surface/70 p-6 sm:p-7">
            <div className="flex items-center gap-2.5">
              <span className="flex h-7 w-7 items-center justify-center rounded-full border-2 border-pink-deep text-pink-deep">
                <X className="h-4 w-4" aria-hidden="true" />
              </span>
              <h3 className="headline text-[1.05rem] text-pink-deep">Usually avoid</h3>
            </div>

            <ul className="mt-5 grid gap-4 sm:grid-cols-3">
              {AVOID.map(({ Art, title, body }) => (
                <li key={title}>
                  <div className="overflow-hidden rounded-card border border-dashed border-pink-mid">
                    <div className="aspect-square w-full">
                      <Art />
                    </div>
                  </div>
                  <p className="mt-3 flex items-start gap-1.5 text-[0.85rem] font-semibold text-ink">
                    <X className="mt-0.5 h-3.5 w-3.5 shrink-0 text-pink-deep" aria-hidden="true" />
                    {title}
                  </p>
                  <p className="mt-1 text-[0.8rem] leading-relaxed text-ink-soft">{body}</p>
                </li>
              ))}
            </ul>

            <ul className="mt-6 grid gap-x-6 gap-y-2 border-t border-line-soft pt-5 sm:grid-cols-2">
              {[
                "Severe cropping that removes kidney anatomy",
                "Photos of medical equipment",
                "Documents or reports",
                "Unrelated anatomy",
              ].map((item) => (
                <li
                  key={item}
                  className="flex gap-2 text-[0.84rem] leading-relaxed text-ink-soft"
                >
                  <X className="mt-0.5 h-3.5 w-3.5 shrink-0 text-pink-deep" aria-hidden="true" />
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Two columns so the copy runs the full width of its measure instead
            of wrapping early against empty space. */}
        <div className="mt-8 grid gap-5 rounded-card border border-line bg-surface/70 p-5 sm:p-6 lg:grid-cols-[minmax(0,1.55fr)_minmax(0,1fr)] lg:gap-8">
          <div className="grid gap-4 sm:grid-cols-[auto_minmax(0,1fr)] sm:gap-4">
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-mint-pale text-teal-deep">
              <Info className="h-4 w-4" aria-hidden="true" />
            </span>
            <div>
              <h3 className="text-[0.95rem] font-semibold text-ink">
                SendCalyx validates image files, not clinical appropriateness
              </h3>
              <p className="mt-1.5 text-[0.86rem] leading-relaxed text-ink-soft">
                Uploads are checked for a supported format, a readable image, and the size
                limit. The ensemble will still return a prediction for an image outside its
                training domain, so the output is only meaningful when the input is a
                kidney CT slice. Use the guidance above before reading any result.
              </p>
            </div>
          </div>

          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 border-line-soft pt-1 lg:border-l lg:pl-8 lg:pt-0">
            <div>
              <dt className="text-[0.76rem] text-ink-faint">Formats</dt>
              <dd className="mt-0.5 text-[0.88rem] font-semibold text-ink">
                PNG, JPEG, WebP
              </dd>
            </div>
            <div>
              <dt className="text-[0.76rem] text-ink-faint">Maximum size</dt>
              <dd className="figure mt-0.5 text-[0.88rem] font-semibold text-ink">10 MB</dd>
            </div>
            <div>
              <dt className="text-[0.76rem] text-ink-faint">Model input</dt>
              <dd className="figure mt-0.5 text-[0.88rem] font-semibold text-ink">
                299 x 299
              </dd>
            </div>
            <div>
              <dt className="text-[0.76rem] text-ink-faint">Storage</dt>
              <dd className="mt-0.5 text-[0.88rem] font-semibold text-ink">
                Not retained
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </section>
  );
}
