import { TriangleAlert } from "lucide-react";
import { formatBytes } from "../utils/api.js";

function Row({ label, value }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2.5">
      <dt className="text-[0.82rem] text-ink-faint">{label}</dt>
      <dd className="figure text-[0.82rem] font-medium text-ink">{value}</dd>
    </div>
  );
}

/** Technical facts about the uploaded file. No quality judgement is implied. */
export default function InputMetadata({ metadata }) {
  if (!metadata) return null;

  const [inputWidth, inputHeight] = metadata.model_input_size ?? [299, 299];

  return (
    <section className="panel p-6" aria-labelledby="input-metadata-heading">
      <h2 id="input-metadata-heading" className="headline text-base text-teal-deep">
        Image metadata
      </h2>

      <dl className="mt-3 divide-y divide-line-soft">
        <Row label="Source dimensions" value={`${metadata.width} × ${metadata.height}`} />
        <Row label="Format" value={metadata.format ?? "—"} />
        <Row label="File size" value={formatBytes(metadata.file_size_bytes)} />
        <Row label="Model input" value={`${inputWidth} × ${inputHeight}`} />
      </dl>

      {metadata.below_recommended_resolution ? (
        <p className="mt-3 flex items-start gap-2 rounded-card border border-pink-mid bg-pink/35 px-3.5 py-2.5 text-[0.76rem] leading-snug text-pink-deep">
          <TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          Source is smaller than {inputWidth} × {inputHeight}, so it was upsampled before
          inference.
        </p>
      ) : (
        <p className="mt-3 text-[0.76rem] leading-snug text-ink-faint">
          Source meets the {inputWidth} × {inputHeight} model input size; it was resized
          before inference.
        </p>
      )}
    </section>
  );
}
