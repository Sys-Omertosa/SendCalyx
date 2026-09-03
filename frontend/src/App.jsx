import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { AlertCircle, Loader2 } from "lucide-react";

import Header from "./components/Header.jsx";
import Landing from "./components/Landing.jsx";
import Footer from "./components/Footer.jsx";
import UploadWorkspace from "./components/UploadWorkspace.jsx";
import EnsembleSummary from "./components/EnsembleSummary.jsx";
import ConsensusPanel from "./components/ConsensusPanel.jsx";
import MetricRail from "./components/MetricRail.jsx";
import ModelComparison from "./components/ModelComparison.jsx";
import SaliencyExplorer from "./components/SaliencyExplorer.jsx";
import InputMetadata from "./components/InputMetadata.jsx";
import UsageNotice from "./components/UsageNotice.jsx";
import { checkHealth, predictImage } from "./utils/api.js";

const ANALYSIS_STAGES = [
  "Preprocessing slice to 299 x 299",
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
    <div className="shell py-20">
      <div className="panel mx-auto max-w-xl p-8 sm:p-10">
        <p className="headline text-lg text-teal-deep">Analyzing slice</p>
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
                <span className={`text-sm ${state === "waiting" ? "text-ink-faint" : "text-ink"}`}>
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
  const [serviceState, setServiceState] = useState("starting");
  const previewUrl = useRef(null);

  useEffect(() => {
    let cancelled = false;
    checkHealth().then((health) => {
      if (cancelled) return;
      setServiceState(health?.model_loaded ? "ready" : "unavailable");
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
    window.scrollTo({ top: 0, behavior: "auto" });
  }, [setPreviewFor]);

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setError("");
    window.scrollTo({ top: 0, behavior: "smooth" });
    try {
      const data = await predictImage(file);
      setResult(data);
      setServiceState("ready");
    } catch (thrown) {
      setError(thrown.message);
      setResult(null);
    } finally {
      setIsAnalyzing(false);
    }
  }, [file]);

  const hasResult = Boolean(result?.success);

  const serviceBanner =
    !hasResult && serviceState !== "ready" ? (
      <div className="shell pt-4">
        <p
          className={`flex items-start gap-2.5 rounded-card border px-4 py-3 text-[0.85rem] ${
            serviceState === "starting"
              ? "border-mint bg-mint-pale/70 text-teal-deep"
              : "border-pink-mid bg-pink/40 text-pink-deep"
          }`}
        >
          {serviceState === "starting" ? (
            <>
              <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin" aria-hidden="true" />
              The analysis service is starting. This can take a moment on first load.
            </>
          ) : (
            <>
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
              The analysis service is unavailable. Retry in a moment.
            </>
          )}
        </p>
      </div>
    ) : null;

  const errorBanner =
    error && !isAnalyzing ? (
      <div className="shell pb-4 pt-4">
        <p
          role="alert"
          className="flex items-start gap-2.5 rounded-card border border-pink-mid bg-pink/40 px-4 py-3.5 text-[0.88rem] text-pink-deep"
        >
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          {error}
        </p>
      </div>
    ) : null;

  if (isAnalyzing) {
    return (
      <div className="flex min-h-screen flex-col">
        <Header mode="analysis" onReset={handleClear} />
        <AnalysisProgress />
        <div className="mt-auto">
          <Footer />
        </div>
      </div>
    );
  }

  if (!hasResult) {
    return (
      <div className="flex min-h-screen flex-col">
        <Header mode="landing" />
        {serviceBanner}
        {errorBanner}
        <Landing>
          <UploadWorkspace
            file={file}
            preview={preview}
            onSelect={handleSelect}
            onClear={handleClear}
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
          />
        </Landing>
        <Footer />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col">
      <Header mode="analysis" onReset={handleClear} />

      <motion.main
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: "easeOut" }}
        className="shell space-y-6 py-8"
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
              <p className="truncate px-5 py-3 text-[0.82rem] text-ink-faint">{file?.name}</p>
            </section>

            <InputMetadata metadata={result.input_metadata} />
          </div>

          <div className="space-y-6">
            <EnsembleSummary ensemble={result.ensemble} consensus={result.consensus} />
            <ConsensusPanel
              consensus={result.consensus}
              individualModels={result.individual_models}
            />
          </div>
        </div>

        <MetricRail consensus={result.consensus} processingTime={result.processing_time} />

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

        <UsageNotice />
      </motion.main>

      <Footer />
    </div>
  );
}
