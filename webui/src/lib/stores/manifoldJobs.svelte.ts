import { readWorkProgress, type WorkProgress, type WorkSample } from "../manifoldProgress";
import { isFittingCancellation } from "../runtime/fittingCancellation";
import { describeError } from "../runtime/errors";
import { getHostedController } from "../runtime/registry";

export interface ManifoldJob {
  id: number;
  namespace: string;
  name: string;
  phase: "generating" | "fitting" | "extracting" | "installing" | "scoring";
  status: "running" | "waiting" | "cancelling" | "complete" | "cancelled" | "failed";
  startedAt: number;
  finishedAt: number | null;
  concepts: string[];
  fitAfterwards: boolean;
  progress: WorkProgress;
  samples: WorkSample[];
  error: string | null;
  cancellable: boolean;
}

export const manifoldJobs: { current: ManifoldJob | null } = $state({ current: null });
let sequence = 0;
let cancelCurrent: (() => Promise<void>) | null = null;
const startingLabel = { generating: "Preparing examples", fitting: "Preparing the fit", extracting: "Extracting the concept", installing: "Installing artifact", scoring: "Scoring template" };
const finishedLabel = { generating: "Examples ready", fitting: "Ready to use", extracting: "Concept ready", installing: "Artifact installed", scoring: "Scoring complete" };

export function beginManifoldJob(namespace: string, name: string, phase: ManifoldJob["phase"], concepts: string[] = [], fitAfterwards = false, options?: { cancellable: boolean; cancel?: () => Promise<void> }): number {
  const previous = manifoldJobs.current;
  const continuing = phase === "fitting" && previous?.status === "waiting" && previous.namespace === namespace && previous.name === name;
  if (previous && ["running", "waiting", "cancelling"].includes(previous.status) && !continuing) {
    throw new Error("Another creation job is running. Finish or cancel it before starting another.");
  }
  const id = ++sequence;
  const hosted = getHostedController();
  const cancellable = options?.cancellable ?? hosted !== null;
  cancelCurrent = cancellable ? options?.cancel ?? (() => hosted!.cancelFitting()) : null;
  manifoldJobs.current = {
    id, namespace, name, phase, status: "running", startedAt: continuing ? previous.startedAt : Date.now(), finishedAt: null,
    concepts, fitAfterwards: continuing || fitAfterwards, samples: [], error: null, cancellable,
    progress: { stage: phase, label: startingLabel[phase], completed: null, total: null, detail: "" },
  };
  return id;
}

export function updateManifoldJob(id: number, event: { event: string; data: unknown }): void {
  const job = manifoldJobs.current;
  if (!job || job.id !== id || event.event !== "progress" || job.status !== "running") return;
  const progress = readWorkProgress(event.data, job.concepts);
  if (!progress) return;
  if (job.phase === "generating" && progress.stage === "working") progress.label = "Generating examples";
  const reset = job.progress.stage !== progress.stage || job.progress.total !== progress.total
    || (progress.completed !== null && job.progress.completed !== null && progress.completed < job.progress.completed);
  let samples = reset ? [] : job.samples;
  if (progress.completed !== null && samples.at(-1)?.completed !== progress.completed) {
    samples = [...samples.slice(-19), { at: Date.now(), completed: progress.completed }];
  }
  manifoldJobs.current = { ...job, progress, samples };
}

export function finishManifoldJob(id: number): void {
  const job = manifoldJobs.current;
  if (!job || job.id !== id) return;
  const waiting = job.phase === "generating" && job.fitAfterwards;
  manifoldJobs.current = {
    ...job, status: waiting ? "waiting" : "complete", finishedAt: waiting ? null : Date.now(),
    progress: { ...job.progress, label: waiting ? "Examples ready · preparing the fit" : finishedLabel[job.phase], completed: job.progress.total ?? 1, total: job.progress.total ?? 1, detail: "" },
  };
}

export function failManifoldJob(id: number, error: unknown): void {
  const job = manifoldJobs.current;
  if (!job || job.id !== id) return;
  const cancelled = isFittingCancellation(error) || error instanceof DOMException && error.name === "AbortError";
  manifoldJobs.current = { ...job, status: cancelled ? "cancelled" : "failed", finishedAt: Date.now(), error: cancelled ? null : describeError(error) };
}

export async function cancelManifoldJob(): Promise<void> {
  const current = manifoldJobs.current;
  if (!current || current.status !== "running" || !current.cancellable || !cancelCurrent) return;
  manifoldJobs.current = { ...current, status: "cancelling" };
  try { await cancelCurrent(); }
  catch (error) {
    if (manifoldJobs.current?.id === current.id) manifoldJobs.current = { ...manifoldJobs.current, status: "running" };
    throw error;
  }
}

export function dismissManifoldJob(): void {
  if (manifoldJobs.current && !["running", "waiting", "cancelling"].includes(manifoldJobs.current.status)) manifoldJobs.current = null;
}
