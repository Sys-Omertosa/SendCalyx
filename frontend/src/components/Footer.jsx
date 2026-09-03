import { Logo } from "./Header.jsx";

const REPO_URL = "https://github.com/Sys-Omertosa/SendCalyx";

export default function Footer() {
  return (
    <footer className="border-t border-line bg-surface/70">
      <div className="shell flex flex-col gap-8 py-12 sm:flex-row sm:items-start sm:justify-between">
        <div className="max-w-[46ch]">
          <div className="flex items-center gap-2.5">
            <Logo className="h-8 w-8" />
            <span className="wordmark text-[1.2rem] text-teal-deep">SendCalyx</span>
          </div>
          <p className="mt-4 text-[0.85rem] leading-relaxed text-ink-soft">
            For research and educational use only. Not intended for clinical diagnosis or
            medical decision-making.
          </p>
        </div>

        <div className="flex flex-col gap-3 text-[0.85rem] sm:items-end">
          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer noopener"
            className="font-medium text-teal-deep underline-offset-4 hover:underline"
          >
            Source on GitHub
          </a>
          <p className="text-ink-faint">MIT licensed</p>
          <p className="text-ink-faint">&copy; {new Date().getFullYear()} SendCalyx</p>
        </div>
      </div>
    </footer>
  );
}
