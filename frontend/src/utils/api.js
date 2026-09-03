import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
export const MAX_FILE_BYTES = 10 * 1024 * 1024;

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 180000,
});

/** Class ids come from the model; render them the way a person reads them. */
export function formatClassName(name) {
  if (!name) return "—";
  return name.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

export function formatPercent(value, digits = 1) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(value, digits = 3) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

export function formatBytes(bytes) {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes <= 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/** Checks the file before it leaves the browser, so bad uploads fail fast. */
export function validateImageFile(file) {
  if (!file) return { valid: false, error: "No file selected." };
  if (!ACCEPTED_TYPES.includes(file.type)) {
    return { valid: false, error: "Use a PNG, JPEG, or WebP image." };
  }
  if (file.size > MAX_FILE_BYTES) {
    return { valid: false, error: "That file is over the 10 MB limit." };
  }
  return { valid: true };
}

export async function checkHealth() {
  try {
    const { data } = await client.get("/health", { timeout: 8000 });
    return data;
  } catch {
    return null;
  }
}

export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const { data } = await client.post("/predict", formData);
    return data;
  } catch (error) {
    const detail = error?.response?.data?.detail;
    if (typeof detail === "string") throw new Error(detail);
    if (error?.code === "ECONNABORTED") {
      throw new Error("The analysis timed out. Try a smaller image.");
    }
    if (!error?.response) {
      throw new Error(
        `Can't reach the SendCalyx API at ${API_BASE_URL}. Start the backend and try again.`,
      );
    }
    throw new Error("The analysis failed. Check the backend logs for details.");
  }
}

export { API_BASE_URL };
