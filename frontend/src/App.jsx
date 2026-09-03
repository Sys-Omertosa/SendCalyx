import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { AlertCircle, ServerCrash } from "lucide-react";

import Header from "./components/Header.jsx";
import UploadWorkspace from "./components/UploadWorkspace.jsx";
import EnsembleSummary from "./components/EnsembleSummary.jsx";
import ConsensusPanel from "./components/ConsensusPanel.jsx";
import ModelComparison from "./components/ModelComparison.jsx";
import SaliencyExplorer from "./components/SaliencyExplorer.jsx";
import InputMetadata from "./components/InputMetadata.jsx";
import ResearchNotice from "./components/ResearchNotice.jsx";
import { checkHealth, predictImage } from "./utils/api.js";

const ANALYSIS_STAGES = [
  "Preprocessing slice to 299 × 299",
  "Running InceptionV3, InceptionResNetV2, Xception",
  "Stacking base probabilities",
  "Generating Grad-CAM attribution",
  "Aggregating cross-model maps",
];

function AnalysisProgress() {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setStage((current) => Math.min(current + 1, ANALYSIS_STAGES.length - 1));
    }, 1100);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="mx-auto w-full max-w-[86rem] px-4 pb-12 sm:px-8">
      <div className="panel p-8 sm:p-12">
        <p className="headline text-lg text-teal-deep">Analysing slice</p>
        <ol className="mt-6 space-y-3">
          {ANALYSIS_STAGES.map((label, index) => {
            const state = index < stage ? "done" : index === stage ? "active" : "waiting";
            return (
              <li key={label} className="flex items-center gap-3">
                <span
                  aria-hidden="true"
                  className={`h-2.5 w-2.5 shrink-0 rounded-full ${
                    state === "done"
                      ? "bg-teal"
                      : state === "active"
                        ? "animate-pulse bg-mint"
                        : "bg-line"
                  }`}
                />
                <span
                  className={`text-sm ${
                    state === "waiting" ? "text-ink-faint" : "text-ink"
                  }`}
                >
                  {label}
                </span>
              </li>
            );
          })}
        </ol>
        <div className="mt-8 h-1.5 overflow-hidden rounded-full bg-line-soft">
          <div
            className="h-full rounded-full transition-[width] duration-700 ease-out"
            style={{
              width: `${((stage + 1) / ANALYSIS_STAGES.length) * 100}%`,
              background:
                "linear-gradient(90deg, var(--color-pink) 0%, var(--color-mint) 50%, var(--color-teal) 100%)",
            }}
          />
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");
  const [backendReachable, setBackendReachable] = useState(true);
  const previewUrl = useRef(null);

  useEffect(() => {
    let cancelled = false;
    checkHealth().then((health) => {
      if (!cancelled) setBackendReachable(Boolean(health?.model_loaded));
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(
    () => () => {
      if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    },
    [],
  );

  const setPreviewFor = useCallback((nextFile) => {
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    previewUrl.current = nextFile ? URL.createObjectURL(nextFile) : null;
    setPreview(previewUrl.current);
  }, []);

  const handleSelect = useCallback(
    (nextFile) => {
      setFile(nextFile);
      setPreviewFor(nextFile);
      setResult(null);
      setError("");
    },
    [setPreviewFor],
  );

  const handleClear = useCallback(() => {
    setFile(null);
    setPreviewFor(null);
    setResult(null);
    setError("");
  }, [setPreviewFor]);

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setError("");
    try {
      const data = await predictImage(file);
      setResult(data);
      setBackendReachable(true);
    } catch (thrown) {
      setError(thrown.message);
      setResult(null);
    } finally {
      setIsAnalyzing(false);
    }
  }, [file]);

  const hasResult = Boolean(result?.success);

  return (
    <div className="min-h-screen">
      {hasResult && <Header onReset={handleClear} />}

      {!backendReachable && !hasResult && (
        <div className="mx-auto w-full max-w-[86rem] px-4 pt-6 sm:px-8">
          <p className="flex items-start gap-2.5 rounded-card border border-pink-mid bg-pink/40 px-4 py-3 text-[0.85rem] text-pink-deep">
            <ServerCrash className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            The SendCalyx API isn&rsquo;t responding yet. Start the backend with{" "}
            <code className="figure rounded bg-white/70 px-1.5 py-0.5 text-[0.78rem]">
              python backend/app.py
            </code>{" "}
            and reload.
          </p>
        </div>
      )}

      {!hasResult && (
        <UploadWorkspace
          file={file}
          preview={preview}
          onSelect={handleSelect}
          onClear={handleClear}
          onAnalyze={handleAnalyze}
          isAnalyzing={isAnalyzing}
        />
      )}

      {error && !isAnalyzing && (
        <div className="mx-auto w-full max-w-[86rem] px-4 pb-10 sm:px-8">
          <p
            role="alert"
            className="flex items-start gap-2.5 rounded-card border border-pink-mid bg-pink/40 px-4 py-3.5 text-[0.88rem] text-pink-deep"
          >
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            {error}
          </p>
        </div>
      )}

      {isAnalyzing && <AnalysisProgress />}

      {hasResult && !isAnalyzing && (
        <motion.main
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, ease: "easeOut" }}
          className="mx-auto w-full max-w-[86rem] space-y-6 px-4 py-8 sm:px-8"
        >
          <div className="grid gap-6 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
            <div className="space-y-6">
              <section className="panel overflow-hidden" aria-label="Uploaded CT slice">
                <div className="bg-teal-dark">
                  <img
                    src={preview}
                    alt="Uploaded kidney CT slice"
                    className="mx-auto block max-h-[26rem] w-full object-contain"
                  />
                </div>
                <p className="truncate px-5 py-3 text-[0.82rem] text-ink-faint">
                  {file?.name}
                </p>
              </section>

              <InputMetadata metadata={result.input_metadata} />
            </div>

            <div className="space-y-6">
              <EnsembleSummary
                ensemble={result.ensemble}
                consensus={result.consensus}
                processingTime={result.processing_time}
              />
              <ConsensusPanel
                consensus={result.consensus}
                individualModels={result.individual_models}
              />
            </div>
          </div>

          <ModelComparison
            individualModels={result.individual_models}
            consensus={result.consensus}
          />

          <SaliencyExplorer
            preview={preview}
            individualModels={result.individual_models}
            xaiConsensus={result.xai_consensus}
            consensus={result.consensus}
          />

          <ResearchNotice />

          <footer className="pb-6 pt-2 text-center text-[0.78rem] text-ink-faint">
            Analysed by {result.num_models} models in this SendCalyx research prototype.
          </footer>
        </motion.main>
      )}
    </div>
  );
}
